import os
import re
import time
import random
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Базовый домен, относительно которого строим абсолютные URL
BASE = "https://ilibrary.ru"
# Страница со списком авторов (на ней много ссылок вида /author/<slug>/index.html)
AUTHORS_PAGE = f"{BASE}/author.html"

# Куда сохранять скачанные страницы
OUT_DIR = "./task1/dump"
# Файл, куда запишем подготовленный список ссылок
URLS_TXT = "./task1/urls.txt"
# Файл индекса: номер_файла -> URL
INDEX_TXT = "./task1/index.txt"

# Требование задания: скачать минимум 100 страниц
MIN_PAGES_TO_DOWNLOAD = 100

# "Вежливость" при обращении к сайту:
# - таймаут запроса (сек)
TIMEOUT = 25
# - случайная задержка между запросами (сек) — снижает риск блокировок и уважает нагрузку
MIN_DELAY = 0.6
MAX_DELAY = 1.6

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; StudyCrawler/1.0; +https://example.com/bot)"
}

# Ограничение на размер скачиваемой страницы (в байтах),
# чтобы случайно не скачать что-то очень большое
MAX_BYTES = 5_000_000  # 5 MB

# Ссылки на страницы текста на ilibrary обычно такие:
# https://ilibrary.ru/text/475/p.1/index.html
# Регулярка, описывающая "текстовую" страницу ilibrary:
TEXT_PAGE_RE = re.compile(r"^https?://(?:www\.)?ilibrary\.ru/text/\d+/p\.\d+/index\.html$")


def polite_sleep():
    """
    Делает случайную паузу между запросами.
    Это простой, но эффективный способ:
    - меньше нагрузка на сайт
    - меньше риск банов/ограничений
    """
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def fetch_html(session: requests.Session, url: str) -> str | None:
    """
    Скачивает HTML-страницу по URL и возвращает её как строку.
    Возвращает None, если:
    - запрос не удался
    - контент не text/html
    - контент слишком большой (> MAX_BYTES)

    - HTML НЕ очищаем от разметки — сохраняем "как есть"
    """
    polite_sleep()
    # Выполняем HTTP GET к серверу
    try:
        resp = session.get(url, timeout=TIMEOUT, allow_redirects=True, stream=True)
    except requests.RequestException as e:
        print(f"[SKIP] {url} -> request error: {e}")
        return None

    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "text/html" not in ctype:
        # иногда у простых страниц заголовок бывает странный, но чаще нормальный
        # если не html — пропускаем
        resp.close()
        print(f"[SKIP] {url} -> not html (Content-Type={ctype})")
        return None

    # Скачиваем контент "потоком" (stream=True),
    # чтобы можно было оборвать скачивание при превышении MAX_BYTES
    chunks = []
    total = 0
    try:
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_BYTES:
                print(f"[SKIP] {url} -> too large (> {MAX_BYTES} bytes)")
                return None
            chunks.append(chunk)
    finally:
        resp.close()

    raw = b"".join(chunks)
    encoding = resp.encoding or "utf-8"
    try:
        return raw.decode(encoding, errors="replace")
    except Exception:
        return raw.decode("utf-8", errors="replace")


def extract_links(html: str, base_url: str) -> list[str]:
    """
    Парсит HTML и вытаскивает все ссылки <a href="...">,
    возвращает список абсолютных URL.

    base_url нужен, чтобы urljoin корректно превратил относительные ссылки
    (например /author/...) в абсолютные (https://ilibrary.ru/author/...)
    """
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        out.append(abs_url)
    return out


def get_author_pages(session: requests.Session) -> list[str]:
    """
    Шаг 1 "подготовки списка страниц":
    1) Скачиваем AUTHORS_PAGE (/author.html)
    2) Вытаскиваем оттуда ссылки на страницы авторов вида:
       https://ilibrary.ru/author/<slug>/index.html

    Возвращаем список таких страниц (уникальный, с сохранением порядка).
    """
    html = fetch_html(session, AUTHORS_PAGE)
    if not html:
        raise RuntimeError("Не удалось скачать страницу author.html")

    links = extract_links(html, AUTHORS_PAGE)

    author_pages = []
    for u in links:
        # Фильтруем на страницы авторов
        # Обычно: https://ilibrary.ru/author/<slug>/index.html
        if re.match(r"^https?://(?:www\.)?ilibrary\.ru/author/[^/]+/index\.html$", u):
            author_pages.append(u)

    # Уникализация с сохранением порядка
    seen = set()
    uniq = []
    for u in author_pages:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    return uniq


def author_to_all_works_url(author_index_url: str) -> str:
    """
    Для каждого автора у ilibrary есть "полный список произведений":
      https://ilibrary.ru/author/<slug>/l.all/index.html

    Превращаем:
      .../author/<slug>/index.html
    в:
      .../author/<slug>/l.all/index.html
    """
    # .../author/chekhov/index.html -> .../author/chekhov/l.all/index.html
    return author_index_url.replace("/index.html", "/l.all/index.html")


def collect_text_page_urls(session: requests.Session, limit: int = 600) -> list[str]:
    """
    Шаг 2 "подготовки списка страниц":
    - Получаем список авторов
    - Для каждого автора идём на страницу l.all (все произведения)
    - Там ищем ссылки именно на страницы текста формата:
         /text/<id>/p.<n>/index.html

    limit — сколько ссылок собрать с запасом (лучше 1000+),
    потому что часть ссылок может не скачаться/быть короткой и т.д.
    """
    author_pages = get_author_pages(session)
    print(f"[INFO] authors found: {len(author_pages)}")

    collected = []
    seen = set()

    for i, author_url in enumerate(author_pages, start=1):
        if len(collected) >= limit:
            break

        all_works_url = author_to_all_works_url(author_url)
        html = fetch_html(session, all_works_url)
        if not html:
            continue

        links = extract_links(html, all_works_url)

        # В l.all обычно ссылки идут на p.1 (а иногда и на конкретные p.N)
        for u in links:
            if TEXT_PAGE_RE.match(u):
                if u not in seen:
                    seen.add(u)
                    collected.append(u)
                    if len(collected) >= limit:
                        break

        # Просто лог прогресса: каждые 10 авторов печатаем статистику по собранным авторам
        if i % 10 == 0:
            print(f"[INFO] processed authors: {i}, collected urls: {len(collected)}")

    return collected


def save_text(path: str, content: str) -> None:
    """
    Сохраняет строку content в файл (UTF-8).
    errors="replace" — чтобы даже при странных символах файл всё равно сохранился.
    """
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write(content)


def download_pages(session: requests.Session, urls: list[str], need: int) -> int:
    """
    Шаг 3 задания: скачать страницы по заранее подготовленному списку urls.
    - сохраняем каждую страницу в отдельный файл: 1.txt, 2.txt, ...
    - создаём index.txt: номер -> url

    need — сколько реально нужно сохранить (минимум 100 по заданию)
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    index_lines = []
    saved = 0

    for url in urls:
        if saved >= need:
            break

        html = fetch_html(session, url)
        if not html:
            continue

        # Мини-проверка, что страница не совсем пустая:
        # (иногда могут быть страницы-заглушки или очень короткие)
        if len(html) < 1000:
            print(f"[SKIP] {url} -> too small html")
            continue

        # Нумерация файлов начинается с 1
        num = saved + 1
        out_file = os.path.join(OUT_DIR, f"{num}.txt")
        # Сохраняем HTML "как есть" (НЕ очищаем от разметки)
        save_text(out_file, html)
        # Пишем строку индекса в финальный файл: "номер страницы из выкачки и url"
        index_lines.append(f"{num}\t{url}")
        saved += 1
        print(f"[OK] {num}: {url}")

    # После скачивания формируем index.txt
    # index.txt
    with open(INDEX_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines) + ("\n" if index_lines else ""))

    return saved


def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    # 1) Собираем "предварительно подготовленный список"
    # Берём с запасом, чтобы точно скачать 100 после пропусков
    urls = collect_text_page_urls(session, limit=1200)

    # Записываем urls.txt
    with open(URLS_TXT, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")

    print(f"[INFO] urls saved to {URLS_TXT}: {len(urls)}")

    # 2) Качаем минимум 100 страниц
    saved = download_pages(session, urls, MIN_PAGES_TO_DOWNLOAD)

    if saved < MIN_PAGES_TO_DOWNLOAD:
        print(f"[DONE] скачано {saved}, нужно {MIN_PAGES_TO_DOWNLOAD}. "
              f"Попробуйте увеличить limit в collect_text_page_urls() или повторить запуск.")
    else:
        print(f"[DONE] скачано {saved} страниц. HTML сохранён как есть в ./{OUT_DIR}, индекс: {INDEX_TXT}")


if __name__ == "__main__":
    main()
