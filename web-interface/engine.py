import os
import re
import math
from collections import Counter, defaultdict

import pymorphy3


class VectorSearchEngine:
    """
    Векторный поиск на TF-IDF по ЛЕММАМ + cosine similarity.

    Реализация:
      - грузим TF-IDF и строим postings: lemma -> [(doc_id, tfidf)]
      - считаем norm(doc) для cosine
      - для запроса строим TF-IDF вектор: tf_query * idf
      - считаем dot через postings
      - strict=True: AND-фильтр по всем леммам запроса до ранжирования
    """

    LEMMA_FILE_RE = re.compile(r"^lemmas(\d+)\.txt$")

    # query tokenization (rus words + optional single hyphen)
    WORD_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)?", re.IGNORECASE)

    MIN_LEN = 3
    CLITIC_RE = re.compile(r"^(?P<stem>[а-яё]+)-(то|де|ка|т)$", re.IGNORECASE)

    DROP_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NUMR", "NPRO"}
    MIN_SCORE = 0.20

    def __init__(
        self,
        tfidf_lemmas_dir: str | None = None,
        index_txt: str | None = None,
    ):
        base = os.path.dirname(os.path.abspath(__file__))

        if tfidf_lemmas_dir is None:
            tfidf_lemmas_dir = os.path.normpath(os.path.join(base, "..", "task4", "tfidf_lemmas"))
        if index_txt is None:
            index_txt = os.path.normpath(os.path.join(base, "..", "task1", "index.txt"))

        self.tfidf_dir = tfidf_lemmas_dir
        self.index_txt = index_txt

        self.morph = pymorphy3.MorphAnalyzer()

        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.doc_norm: dict[int, float] = {}
        self.doc_ids: list[int] = []
        self.page_index: dict[int, str] = self._load_page_index(index_txt)

        self._load_tfidf()

    @staticmethod
    def _load_page_index(path: str) -> dict[int, str]:
        """
        task1/index.txt:
          1 https://...
          2 https://...
        """
        page_index: dict[int, str] = {}
        if not path or not os.path.isfile(path):
            return page_index

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                try:
                    doc_id = int(parts[0])
                except ValueError:
                    continue
                page_index[doc_id] = parts[1]
        return page_index

    @classmethod
    def _normalize_token(cls, tok: str) -> str:
        tok = tok.lower()
        m = cls.CLITIC_RE.match(tok)
        return m.group("stem") if m else tok

    @classmethod
    def _is_clean_token(cls, tok: str) -> bool:
        if len(tok) < cls.MIN_LEN:
            return False
        return re.fullmatch(r"[а-яё]+(?:-[а-яё]+)?", tok) is not None

    def _load_tfidf(self) -> None:
        if not os.path.isdir(self.tfidf_dir):
            raise FileNotFoundError(
                f"Не найдена папка TF-IDF: {self.tfidf_dir}."
            )

        sq = defaultdict(float)  # doc_id -> sum(w^2)
        files = 0

        for fname in sorted(os.listdir(self.tfidf_dir)):
            m = self.LEMMA_FILE_RE.fullmatch(fname)
            if not m:
                continue
            files += 1
            doc_id = int(m.group(1))
            self.doc_ids.append(doc_id)

            path = os.path.join(self.tfidf_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 3:
                        continue

                    lemma = parts[0]
                    idf_val = float(parts[1])
                    tfidf_val = float(parts[2])

                    if lemma not in self.idf:
                        self.idf[lemma] = idf_val

                    if tfidf_val != 0.0:
                        self.postings[lemma].append((doc_id, tfidf_val))
                        sq[doc_id] += tfidf_val * tfidf_val

        if files == 0:
            raise FileNotFoundError(f"В {self.tfidf_dir} нет файлов вида lemmasN.txt")

        self.doc_ids.sort()
        self.doc_norm = {doc_id: math.sqrt(v) for doc_id, v in sq.items()}

    def _query_lemmas(self, query: str) -> list[str]:
        raw = []
        for m in self.WORD_RE.finditer(query.lower()):
            t = self._normalize_token(m.group(0))
            if self._is_clean_token(t):
                raw.append(t)

        lemmas = []
        for t in raw:
            p = self.morph.parse(t)[0]

            if hasattr(p, "is_known") and not p.is_known:
                continue
            if p.tag.POS in self.DROP_POS:
                continue
            if p.score < self.MIN_SCORE:
                continue

            lemma = p.normal_form
            if lemma in self.idf:
                lemmas.append(lemma)

        return lemmas

    def _build_query_vector(self, query: str) -> dict[str, float]:
        """
        TF-IDF query vector:
          w_q = tf_q * idf
        """
        lemmas = self._query_lemmas(query)
        if not lemmas:
            return {}

        cnt = Counter(lemmas)
        total = sum(cnt.values())

        q_vec = {l: (c / total) * self.idf[l] for l, c in cnt.items()}
        return q_vec

    @staticmethod
    def _l2_norm(vec: dict[str, float]) -> float:
        return math.sqrt(sum(w * w for w in vec.values()))

    # поиск
    def search(self, query: str, top_n: int = 10, strict: bool = False) -> list[dict]:
        """
        Returns:
          [{"doc_id": int, "url": str, "score": float}, ...]
        """
        q_vec = self._build_query_vector(query)
        if not q_vec:
            return []

        q_norm = self._l2_norm(q_vec)
        if q_norm == 0.0:
            return []

        allowed_docs = None
        if strict:
            for lemma in q_vec.keys():
                docs = {doc_id for doc_id, _ in self.postings.get(lemma, [])}
                allowed_docs = docs if allowed_docs is None else (allowed_docs & docs)
            if not allowed_docs:
                return []

        dot = defaultdict(float)
        for lemma, wq in q_vec.items():
            for doc_id, wd in self.postings.get(lemma, []):
                if allowed_docs is not None and doc_id not in allowed_docs:
                    continue
                dot[doc_id] += wq * wd

        scored = []
        for doc_id, dp in dot.items():
            d_norm = self.doc_norm.get(doc_id, 0.0)
            if d_norm == 0.0:
                continue
            score = dp / (q_norm * d_norm)
            if score > 0:
                url = self.page_index.get(doc_id, f"dump/{doc_id}.txt")
                scored.append({"doc_id": doc_id, "url": url, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        scored = scored[:top_n]

        # округление только на вывод
        for r in scored:
            r["score"] = round(r["score"], 6)

        return scored