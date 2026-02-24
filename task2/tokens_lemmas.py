import os
import re
import glob
from collections import defaultdict, Counter

from bs4 import BeautifulSoup
import pymorphy3


# Пути к папкам
# Папка, где лежит этот скрипт (например .../task2)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Папка с HTML-дампами из task1: .../task1/dump
DUMP_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "task1", "dump"))

# Папки для результатов (создадим, если не существуют)
TOKENS_DIR = os.path.join(BASE_DIR, "tokens")
LEMMAS_DIR = os.path.join(BASE_DIR, "lemmas")


# Токенизация и фильтры
# Русские слова; допускаем один дефис внутри (например "северо-запад")
WORD_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)?", re.IGNORECASE)

# Минимальная длина токена, чтобы отсеять короткие служебные обломки
MIN_LEN = 3

# Нормализация дефисных клитик: "тут-то" -> "тут", "смотри-ка" -> "смотри"
CLITIC_RE = re.compile(r"^(?P<stem>[а-яё]+)-(то|де|ка|т)$", re.IGNORECASE)

# Части речи, которые выкидываем как "неинтересные" для словаря
# PREP предлог, CONJ союз, PRCL частица, INTJ междометие, NUMR числительное, NPRO местоимение
DROP_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NUMR", "NPRO"}

# Порог уверенности морфоразбора
MIN_SCORE = 0.20

# Более строгий порог для слов, встретившихся 1 раз в данном файле
HAPAX_SCORE = 0.35


def html_to_text(html: str) -> str:
    """
    Достаём видимый текст из HTML:
    - парсим HTML
    - удаляем script/style/noscript
    - возвращаем "плоский" текст
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def normalize_token(tok: str) -> str:
    """
    Нормализация токена:
    - нижний регистр
    - отрезаем дефисные клитики (тут-то -> тут)
    """
    tok = tok.lower()
    m = CLITIC_RE.match(tok)
    if m:
        return m.group("stem")
    return tok


def is_clean_token(tok: str) -> bool:
    """
    Грубый фильтр токена до морфологии:
    - длина >= MIN_LEN
    - только кириллица, допускаем один дефис внутри
    """
    tok = tok.lower()
    if len(tok) < MIN_LEN:
        return False
    return re.fullmatch(r"[а-яё]+(?:-[а-яё]+)?", tok) is not None


def file_id_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    if re.fullmatch(r"\d+", stem):
        return stem
    return stem


def process_one_file(morph: pymorphy3.MorphAnalyzer, path: str):
    """
    Обрабатываем один файл:
    1) вытаскиваем текст из HTML
    2) считаем частоты токенов
    3) фильтруем по морфологии/уверенности
    4) строим отображение лемма -> токены
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()

    text = html_to_text(html)

    # Частоты считаем в рамках одного файла, чтобы hapax-фильтр был "по файлу"
    freq = Counter()

    for m in WORD_RE.finditer(text.lower()):
        tok = normalize_token(m.group(0))
        if is_clean_token(tok):
            freq[tok] += 1

    tokens = set()

    for tok, count_in_file in freq.items():
        p = morph.parse(tok)[0]  # самый вероятный разбор

        # Неизвестные слова (часто мусор/обрывки)
        if hasattr(p, "is_known") and not p.is_known:
            continue

        # Служебные части речи/местоимения/числительные
        if p.tag.POS in DROP_POS:
            continue

        # Слишком низкая уверенность
        if p.score < MIN_SCORE:
            continue

        # Слово встретилось 1 раз и при этом разбор неуверенный — выбрасываем
        if count_in_file == 1 and p.score < HAPAX_SCORE:
            continue

        tokens.add(tok)

    tokens_sorted = sorted(tokens)

    lemma_to_tokens = defaultdict(set)
    for tok in tokens_sorted:
        lemma = morph.parse(tok)[0].normal_form
        lemma_to_tokens[lemma].add(tok)

    return tokens_sorted, lemma_to_tokens


def write_outputs(out_id: str, tokens_sorted, lemma_to_tokens):
    """
    Запись результатов в две отдельные папки:
      tokens/tokens{ID}.txt
      lemmas/lemmas{ID}.txt
    """
    tokens_path = os.path.join(TOKENS_DIR, f"tokens{out_id}.txt")
    lemmas_path = os.path.join(LEMMAS_DIR, f"lemmas{out_id}.txt")

    # tokens
    with open(tokens_path, "w", encoding="utf-8") as f:
        for t in tokens_sorted:
            f.write(t + "\n")

    # lemmas
    with open(lemmas_path, "w", encoding="utf-8") as f:
        for lemma in sorted(lemma_to_tokens.keys()):
            toks = sorted(lemma_to_tokens[lemma])
            f.write(lemma + " " + " ".join(toks) + "\n")

    print(f"[OK] {out_id}: tokens={len(tokens_sorted)} -> {tokens_path}")
    print(f"[OK] {out_id}: lemmas={len(lemma_to_tokens)} -> {lemmas_path}")


def main():
    # Создаём папки для результатов, если их нет
    os.makedirs(TOKENS_DIR, exist_ok=True)
    os.makedirs(LEMMAS_DIR, exist_ok=True)

    morph = pymorphy3.MorphAnalyzer()

    files = sorted(glob.glob(os.path.join(DUMP_DIR, "*.txt")))
    if not files:
        raise SystemExit(
            f"Не найдено файлов в папке '{DUMP_DIR}'. "
            f"Ожидаются {DUMP_DIR}/1.txt, {DUMP_DIR}/2.txt, ..."
        )

    for path in files:
        out_id = file_id_from_path(path)
        tokens_sorted, lemma_to_tokens = process_one_file(morph, path)
        write_outputs(out_id, tokens_sorted, lemma_to_tokens)


if __name__ == "__main__":
    main()