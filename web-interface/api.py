from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from engine import VectorSearchEngine

app = FastAPI(title="Vector Search (TF-IDF)")
engine = VectorSearchEngine()

HTML = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Vector Search — TF-IDF</title>

  <!-- Bootstrap 5 (CDN) -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">

  <style>
    body { background: #f7f8fb; }
    .card { border: 0; box-shadow: 0 8px 24px rgba(16,24,40,.08); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .link-wrap { max-width: 520px; word-break: break-all; }
  </style>
</head>
<body>
  <div class="container py-5">
    <div class="row justify-content-center">
      <div class="col-lg-10 col-xl-9">

        <div class="mb-4">
          <h1 class="h3 mb-1">Векторный поиск для сайта ilibrary.ru</h1>
        </div>

        <div class="card p-4 mb-4">
          <div class="row g-3 align-items-end">
            <div class="col-12">
              <label for="q" class="form-label">Запрос</label>
              <input id="q" class="form-control form-control-lg" placeholder="Например: клеопатра цезарь" autocomplete="off">
              <div class="form-text">Введите фразу или несколько слов. Нажмите Enter для поиска.</div>
            </div>

            <div class="col-md-4">
              <label for="top" class="form-label">Top-K</label>
              <select id="top" class="form-select">
                <option value="5">5</option>
                <option value="10" selected>10</option>
                <option value="20">20</option>
              </select>
            </div>

            <div class="col-md-4">
              <label class="form-label d-block">Режим</label>
              <div class="form-check form-switch">
                <input class="form-check-input" type="checkbox" role="switch" id="strict">
                <label class="form-check-label" for="strict">Strict (документ содержит все леммы запроса)</label>
              </div>
            </div>

            <div class="col-md-4 d-grid">
              <button id="btn" class="btn btn-primary btn-lg" onclick="doSearch()">
                <span id="btnText">Поиск</span>
                <span id="spinner" class="spinner-border spinner-border-sm ms-2 d-none" role="status" aria-hidden="true"></span>
              </button>
            </div>
          </div>
        </div>

        <div id="alert" class="alert alert-danger d-none" role="alert"></div>

        <div class="card p-4">
          <div class="d-flex justify-content-between align-items-center mb-3">
            <div>
              <div class="h5 mb-0">Результаты</div>
              <div id="meta" class="text-muted small">—</div>
            </div>
            <button class="btn btn-outline-secondary btn-sm" onclick="clearResults()">Очистить</button>
          </div>

          <div class="table-responsive">
            <table class="table align-middle mb-0">
              <thead>
                <tr>
                  <th style="width: 56px;">#</th>
                  <th style="width: 90px;">Файл из выкачки</th>
                  <th style="width: 120px;">score</th>
                  <th>Ссылка</th>
                </tr>
              </thead>
              <tbody id="tbody">
                <tr><td colspan="4" class="text-muted">Пока нет результатов — выполните поиск.</td></tr>
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  </div>

  <script>
    const qInput = document.getElementById("q");
    qInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") doSearch();
    });

    function setLoading(isLoading) {
      const spinner = document.getElementById("spinner");
      const btn = document.getElementById("btn");
      const btnText = document.getElementById("btnText");
      spinner.classList.toggle("d-none", !isLoading);
      btn.disabled = isLoading;
      btnText.textContent = isLoading ? "Ищем..." : "Поиск";
    }

    function showError(msg) {
      const a = document.getElementById("alert");
      a.textContent = msg;
      a.classList.remove("d-none");
    }

    function hideError() {
      document.getElementById("alert").classList.add("d-none");
    }

    function clearResults() {
      document.getElementById("meta").textContent = "—";
      document.getElementById("tbody").innerHTML =
        '<tr><td colspan="4" class="text-muted">Пока нет результатов — выполните поиск.</td></tr>';
      hideError();
    }

    function render(results, query, strict, top_n) {
      const tbody = document.getElementById("tbody");
      const meta = document.getElementById("meta");

      meta.textContent = `Запрос: "${query}" · strict=${strict} · top=${top_n} · найдено: ${results.length}`;
      if (!results.length) {
        tbody.innerHTML = '<tr><td colspan="4"><span class="text-muted">Ничего не найдено.</span></td></tr>';
        return;
      }

      let html = "";
      results.forEach((r, i) => {
        const urlCell = (r.url && (r.url.startsWith("http://") || r.url.startsWith("https://")))
          ? `<a class="link-primary link-wrap" href="${r.url}" target="_blank" rel="noopener">${r.url}</a>`
          : `<span class="text-muted link-wrap mono">${r.url ?? ""}</span>`;

        html += `
          <tr>
            <td class="text-muted">${i+1}</td>
            <td class="mono">${r.doc_id}</td>
            <td class="mono">${Number(r.score).toFixed(6)}</td>
            <td>${urlCell}</td>
          </tr>
        `;
      });

      tbody.innerHTML = html;
    }

    async function doSearch() {
      hideError();
      const q = document.getElementById("q").value.trim();
      const strict = document.getElementById("strict").checked;
      const top_n = Number(document.getElementById("top").value);

      if (!q) {
        showError("Введите запрос.");
        return;
      }

      setLoading(true);
      try {
        const resp = await fetch(`/search?q=${encodeURIComponent(q)}&strict=${strict}&top_n=${top_n}`);
        if (!resp.ok) {
          const txt = await resp.text();
          throw new Error(txt || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        render(data.results, data.query, data.strict, data.top_n);
      } catch (e) {
        showError("Ошибка поиска: " + (e?.message ?? e));
      } finally {
        setLoading(false);
      }
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    strict: bool = False,
    top_n: int = 10
):
    top_n = max(1, min(int(top_n), 50))
    results = engine.search(q, top_n=top_n, strict=strict)
    return JSONResponse({"query": q, "strict": strict, "top_n": top_n, "results": results})