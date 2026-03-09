import os
import re
from collections import defaultdict

import pymorphy3


# Пути к папкам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEMMAS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "task2/" "lemmas"))     # из task2
INDEX_OUT = os.path.join(BASE_DIR, "inverted_index.txt")  # человекочитаемый вывод


# Индекс
def build_inverted_index(lemmas_dir: str):
    """
    Читает lemmas/lemmasN.txt и строит индекс:
      lemma -> set(doc_id)
    """
    index = defaultdict(set)

    # Берём файлы вида lemmas123.txt
    for fname in sorted(os.listdir(lemmas_dir)):
        m = re.fullmatch(r"lemmas(\d+)\.txt", fname)
        if not m:
            continue
        doc_id = int(m.group(1))
        path = os.path.join(lemmas_dir, fname)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # формат: "лемма токен1 токен2 ..."
                lemma = line.split()[0]
                index[lemma].add(doc_id)

    all_docs = set()
    for docs in index.values():
        all_docs |= docs

    return index, all_docs


def save_index(index, out_path: str):
    """
    Сохраняет индекс в текстовый файл:
      lemma: 1 2 5 10
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for term in sorted(index.keys()):
            docs = sorted(index[term])
            f.write(f"{term}: " + " ".join(map(str, docs)) + "\n")


# Булев поиск
TOKEN_RE = re.compile(
    r"\s*("                       # пропускаем пробелы
    r"\("                         # (
    r"|\)"                        # )
    r"|AND|OR|NOT"                # операторы
    r"|[а-яё]+(?:-[а-яё]+)?"      # слово (кириллица, один дефис)
    r")\s*",
    re.IGNORECASE
)

OP_PRECEDENCE = {
    "NOT": 3,
    "AND": 2,
    "OR": 1,
}


def tokenize_query(q: str):
    """
    Разбиваем строку запроса на токены:
    '(', ')', AND/OR/NOT, или слово
    """
    pos = 0
    tokens = []
    while pos < len(q):
        m = TOKEN_RE.match(q, pos)
        if not m:
            # непонятный символ/фрагмент
            raise ValueError(f"Некорректный запрос около: {q[pos:pos+20]!r}")
        tok = m.group(1)
        tokens.append(tok)
        pos = m.end()
    return tokens


def normalize_query_tokens(tokens, morph):
    """
    - операторы приводим к верхнему регистру
    - слова лемматизируем (чтобы 'Цезаря' искалось как 'цезарь')
    """
    norm = []
    for t in tokens:
        up = t.upper()
        if up in ("AND", "OR", "NOT", "(", ")"):
            norm.append(up)
        else:
            # слово -> лемма
            lemma = morph.parse(t.lower())[0].normal_form
            norm.append(lemma)
    return norm


def to_rpn(tokens):
    """
    Преобразует выражение из инфиксной формы (как мы пишем обычно):
        (a AND b) OR (c AND d)
    в обратную польскую запись (RPN), которую легко вычислять стеком:
        a b AND c d AND OR

    Используем алгоритм "shunting-yard" (Дейкстра):
    - output: выходной список (RPN)
    - ops   : стек операторов

    Скобки управляют порядком операций.
    Приоритеты задаются OP_PRECEDENCE.
    """
    output = []
    ops = []

    for t in tokens:
        if t == "(":
            ops.append(t)
        elif t == ")":
            while ops and ops[-1] != "(":
                output.append(ops.pop())
            if not ops:
                raise ValueError("Скобки несбалансированы: лишняя ')'")
            ops.pop()  # убрать '('
        elif t in OP_PRECEDENCE:
            while (
                ops and ops[-1] in OP_PRECEDENCE and
                OP_PRECEDENCE[ops[-1]] >= OP_PRECEDENCE[t]
            ):
                output.append(ops.pop())
            ops.append(t)
        else:
            output.append(t)

    while ops:
        if ops[-1] in ("(", ")"):
            raise ValueError("Скобки несбалансированы: лишняя '('")
        output.append(ops.pop())

    return output


def eval_rpn(rpn, index, all_docs):
    """
    Вычисляем RPN-выражение. На стеке — множества doc_id.
    """
    stack = []

    for t in rpn:
        if t == "NOT":
            if not stack:
                raise ValueError("NOT применяется без операнда")
            a = stack.pop()
            stack.append(all_docs - a)
        elif t == "AND":
            if len(stack) < 2:
                raise ValueError("AND требует 2 операнда")
            b = stack.pop()
            a = stack.pop()
            stack.append(a & b)
        elif t == "OR":
            if len(stack) < 2:
                raise ValueError("OR требует 2 операнда")
            b = stack.pop()
            a = stack.pop()
            stack.append(a | b)
        else:
            stack.append(set(index.get(t, set())))

    if len(stack) != 1:
        raise ValueError("Некорректное выражение: проверьте операторы/скобки")
    return stack[0]


def boolean_search(query: str, index, all_docs, morph):
    tokens = tokenize_query(query)
    tokens = normalize_query_tokens(tokens, morph)
    rpn = to_rpn(tokens)
    result_docs = eval_rpn(rpn, index, all_docs)
    return sorted(result_docs)


def main():
    if not os.path.isdir(LEMMAS_DIR):
        raise SystemExit(
            f"Не найдена папка с леммами: {LEMMAS_DIR}\n"
        )

    morph = pymorphy3.MorphAnalyzer()

    index, all_docs = build_inverted_index(LEMMAS_DIR)
    save_index(index, INDEX_OUT)

    print(f"[OK] Индекс построен. Терминов: {len(index)}. Документов: {len(all_docs)}.")
    print(f"[OK] Индекс сохранён в: {INDEX_OUT}")
    print()
    print("Введите запрос (AND/OR/NOT, скобки). Пример:")
    print("(Клеопатра AND Цезарь) OR (Антоний AND Цицерон) OR Помпей")
    print("Для выхода: пустая строка.")
    print()

    while True:
        q = input("QUERY> ").strip()
        if not q:
            break
        try:
            docs = boolean_search(q, index, all_docs, morph)
            if docs:
                print("Найдено в документах:", ", ".join(map(str, docs)))
            else:
                print("Совпадений нет.")
        except Exception as e:
            print("Ошибка:", e)
        print()


if __name__ == "__main__":
    main()