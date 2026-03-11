import os
import re
import glob
import math
from collections import Counter, defaultdict

from bs4 import BeautifulSoup
import pymorphy3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Папка с HTML-выкачкой из задания 1
DUMP_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "task1", "dump"))

# Куда писать результаты
OUT_TERMS_DIR = os.path.join(BASE_DIR, "tfidf_terms")
OUT_LEMMAS_DIR = os.path.join(BASE_DIR, "tfidf_lemmas")

# Русские слова; допускаем один дефис внутри
WORD_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)?", re.IGNORECASE)

# Минимальная длина токена
MIN_LEN = 3

# Нормализация дефисных клитик: "тут-то" -> "тут"
CLITIC_RE = re.compile(r"^(?P<stem>[а-яё]+)-(то|де|ка|т)$", re.IGNORECASE)

# Служебные части речи/местоимения/числительные
DROP_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NUMR", "NPRO"}

# Пороги уверенности разборов (для мусора)
MIN_SCORE = 0.20
HAPAX_SCORE = 0.35


def html_to_text(html: str) -> str:
    """Вытаскиваем видимый текст из HTML."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def normalize_token(tok: str) -> str:
    """Нижний регистр + отрезаем дефисные клитики."""
    tok = tok.lower()
    m = CLITIC_RE.match(tok)
    if m:
        return m.group("stem")
    return tok


def is_clean_token(tok: str) -> bool:
    """Грубая чистка: длина и только кириллица (+ один дефис)."""
    tok = tok.lower()
    if len(tok) < MIN_LEN:
        return False
    return re.fullmatch(r"[а-яё]+(?:-[а-яё]+)?", tok) is not None


def file_id_from_path(path: str) -> int:
    """Берём номер документа из имени файла dump/N.txt."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if not re.fullmatch(r"\d+", stem):
        raise ValueError(f"Ожидался файл вида N.txt, а получили: {path}")
    return int(stem)


# 1) СЧИТАЕМ ЧАСТОТЫ ТЕРМИНОВ/ЛЕММ ПО ДОКУМЕНТАМ

def count_terms_in_doc(morph: pymorphy3.MorphAnalyzer, path: str):
    """
    Возвращает:
      term_counts: Counter(term -> count)  (term = токен)
      lemma_counts: Counter(lemma -> count)
      total_terms: int  (общее число терминов в документе после всех фильтров)
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()

    text = html_to_text(html)

    raw_freq = Counter()
    for m in WORD_RE.finditer(text.lower()):
        tok = normalize_token(m.group(0))
        if is_clean_token(tok):
            raw_freq[tok] += 1

    # Итоговые счётчики
    term_counts = Counter()
    lemma_counts = Counter()

    # total_terms = сумма всех прошедших токенов (с учётом повторов)
    total_terms = 0

    for tok, c in raw_freq.items():
        p = morph.parse(tok)[0]

        # неизвестные слова
        if hasattr(p, "is_known") and not p.is_known:
            continue

        # служебные части речи и пр.
        if p.tag.POS in DROP_POS:
            continue

        # низкая уверенность
        if p.score < MIN_SCORE:
            continue

        # hapax + низкая уверенность
        if c == 1 and p.score < HAPAX_SCORE:
            continue

        # если токен прошёл — добавляем его count раз
        term_counts[tok] = c
        total_terms += c

        # лемма этого токена, тоже count раз
        lemma = p.normal_form
        lemma_counts[lemma] += c

    return term_counts, lemma_counts, total_terms


# 2) DF/IDF + TF-IDF + ЗАПИСЬ ФАЙЛОВ

def compute_idf(df: int, n_docs: int) -> float:
    """
    IDF = ln(N / df)
    (натуральный логарифм)
    """
    return math.log(n_docs / df)


def write_tfidf_file(out_path: str, items):
    """
    items: iterable of (key, idf, tfidf)
    Формат строки: <key><пробел><idf><пробел><tf-idf>\n
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for key, idf, tfidf in items:
            f.write(f"{key} {idf:.8f} {tfidf:.8f}\n")


def main():
    if not os.path.isdir(DUMP_DIR):
        raise SystemExit(f"Не найдена папка dump из задания 1: {DUMP_DIR}")

    os.makedirs(OUT_TERMS_DIR, exist_ok=True)
    os.makedirs(OUT_LEMMAS_DIR, exist_ok=True)

    morph = pymorphy3.MorphAnalyzer()

    # Все документы dump/N.txt
    files = sorted(glob.glob(os.path.join(DUMP_DIR, "*.txt")))
    if not files:
        raise SystemExit(f"В папке {DUMP_DIR} нет *.txt файлов")

    # doc_id -> данные по документу
    doc_term_counts = {}
    doc_lemma_counts = {}
    doc_total_terms = {}

    # DF по коллекции
    df_terms = Counter()   # term -> number of docs containing term
    df_lemmas = Counter()  # lemma -> number of docs containing lemma

    # 1) Пробегаем все документы и собираем counts + df
    doc_ids = []
    for path in files:
        doc_id = file_id_from_path(path)
        doc_ids.append(doc_id)

        term_counts, lemma_counts, total_terms = count_terms_in_doc(morph, path)

        doc_term_counts[doc_id] = term_counts
        doc_lemma_counts[doc_id] = lemma_counts
        doc_total_terms[doc_id] = total_terms

        # DF: учитываем термин/лемму 1 раз на документ
        for t in term_counts.keys():
            df_terms[t] += 1
        for l in lemma_counts.keys():
            df_lemmas[l] += 1

    n_docs = len(doc_ids)

    # 2) Для каждого документа считаем TF-IDF и пишем в файлы
    for doc_id in sorted(doc_ids):
        total_terms = doc_total_terms[doc_id]

        # Если документ пустой после фильтров — создаём пустые файлы
        if total_terms == 0:
            open(os.path.join(OUT_TERMS_DIR, f"terms{doc_id}.txt"), "w", encoding="utf-8").close()
            open(os.path.join(OUT_LEMMAS_DIR, f"lemmas{doc_id}.txt"), "w", encoding="utf-8").close()
            continue

        # Термины
        term_items = []
        for term, cnt in doc_term_counts[doc_id].items():
            tf = cnt / total_terms
            idf = compute_idf(df_terms[term], n_docs)
            tfidf = tf * idf
            term_items.append((term, idf, tfidf))

        # сортируем для стабильного вывода
        term_items.sort(key=lambda x: x[0])

        write_tfidf_file(
            os.path.join(OUT_TERMS_DIR, f"terms{doc_id}.txt"),
            term_items
        )

        # Леммы
        lemma_items = []
        for lemma, cnt in doc_lemma_counts[doc_id].items():
            # tf леммы = (сумма вхождений её терминов) / (общее число терминов в документе)
            tf = cnt / total_terms
            idf = compute_idf(df_lemmas[lemma], n_docs)
            tfidf = tf * idf
            lemma_items.append((lemma, idf, tfidf))

        lemma_items.sort(key=lambda x: x[0])

        write_tfidf_file(
            os.path.join(OUT_LEMMAS_DIR, f"lemmas{doc_id}.txt"),
            lemma_items
        )

    print(f"[OK] Documents: {n_docs}")
    print(f"[OK] TF-IDF terms  -> {OUT_TERMS_DIR}")
    print(f"[OK] TF-IDF lemmas -> {OUT_LEMMAS_DIR}")


if __name__ == "__main__":
    main()