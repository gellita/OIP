from engine import VectorSearchEngine

TOP_N = 10
STRICT = False  # True = документ должен содержать все леммы запроса (AND-фильтр)

def main():
    print("=" * 70)
    print("  Vector Search (TF-IDF lemmas + cosine)")
    print("=" * 70)

    engine = VectorSearchEngine()  # грузит tfidf_lemmas и index.txt один раз при старте

    print("Введите запрос. Пустая строка — выход.")
    print(f"Параметры: TOP_N={TOP_N}, STRICT={STRICT}\n")

    while True:
        q = input("QUERY> ").strip()
        if not q:
            break

        results = engine.search(q, top_n=TOP_N, strict=STRICT)

        if not results:
            print("Совпадений нет.\n")
            continue

        for r in results:
            # r: {"doc_id": int, "url": str, "score": float}
            print(f"{r['doc_id']}\t{r['score']:.6f}\t{r['url']}")
        print()

if __name__ == "__main__":
    main()