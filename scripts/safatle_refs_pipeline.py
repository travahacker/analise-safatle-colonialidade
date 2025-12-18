#!/usr/bin/env python3
"""Pipeline: download (optional) -> OCR -> extrair nomes -> enriquecer via Wikidata -> gerar web.

Este script foi escrito para ser reprodutível e auditável.
Ele NÃO inventa atributos sensíveis: só preenche quando há dado público explícito (Wikidata)
— caso contrário, marca como "desconhecido".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import pytesseract
import requests
from PIL import Image, ImageEnhance, ImageOps
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
OCR_DIR = DATA_DIR / "ocr"
DERIVED_DIR = DATA_DIR / "derived"
WEB_DIR = ROOT / "web"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"


@dataclass(frozen=True)
class OcrResult:
    text: str
    mean_conf: float
    method: str


def _iter_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files


def _preprocess(img: Image.Image, crop: bool) -> Image.Image:
    # Remove UI de screenshot (top/bottom) de forma conservadora
    if crop:
        w, h = img.size
        top = int(h * 0.13)
        bot = int(h * 0.87)
        img = img.crop((0, top, w, bot))

    # Up-scale + contraste ajuda bastante em texto de slide
    img = img.convert("RGB")
    img = img.resize((img.size[0] * 2, img.size[1] * 2), Image.Resampling.LANCZOS)
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = ImageEnhance.Sharpness(img).enhance(1.3)

    # grayscale + binarização leve
    gray = img.convert("L")
    # threshold adaptativo simples: usa percentil
    hist = gray.histogram()
    total = sum(hist)
    cum = 0
    thresh = 180
    for i, v in enumerate(hist):
        cum += v
        if cum / total >= 0.85:
            thresh = i
            break
    bw = gray.point(lambda x: 255 if x > thresh else 0)
    return bw


def _tesseract_data_conf(img: Image.Image, lang: str) -> tuple[str, float]:
    cfg = "--oem 1 --psm 6"
    data = pytesseract.image_to_data(img, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
    tokens = []
    confs = []
    for txt, conf in zip(data.get("text", []), data.get("conf", []), strict=False):
        t = (txt or "").strip()
        if not t:
            continue
        try:
            c = float(conf)
        except Exception:
            c = -1.0
        if c >= 0:
            confs.append(c)
        tokens.append(t)
    text = " ".join(tokens)
    mean_conf = (sum(confs) / len(confs)) if confs else -1.0
    return text, mean_conf


def ocr_image(path: Path, lang: str = "por", crop: bool = True) -> OcrResult:
    img = Image.open(path)

    # tenta 2 métodos e escolhe pelo mean_conf
    raw_text, raw_conf = _tesseract_data_conf(img, lang=lang)
    proc_img = _preprocess(img, crop=crop)
    proc_text, proc_conf = _tesseract_data_conf(proc_img, lang=lang)

    if proc_conf >= raw_conf:
        return OcrResult(text=proc_text, mean_conf=proc_conf, method="preprocess")
    return OcrResult(text=raw_text, mean_conf=raw_conf, method="raw")


NAME_PATTERNS: list[re.Pattern[str]] = [
    # SOBRENOME, Nome
    re.compile(r"\b([A-ZÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ][A-ZÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ\-]{2,})\s*,\s*([A-ZÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ][\wÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ\-']+(?:\s+[A-ZÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ][\wÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ\-']+)*)\b"),
    # Nome Sobrenome (bem conservador: 2-4 termos capitalizados)
    re.compile(r"\b([A-ZÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ][\wÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ\-']+\s+[A-ZÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ][\wÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ\-']+(?:\s+[A-ZÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ][\wÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ\-']+){0,2})\b"),
]


STOPWORDS = {
    "Post",
    "Instagram",
    "Filosofia",
    "FFLCH",
    "USP",
    "Teoria",
    "Ciências",
    "Humanas",
    "Primeiro",
    "Semestre",
    "Colonialidade",
}


def _clean_ocr_text(s: str) -> str:
    s = s.replace("\u00ad", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_names(text: str) -> list[str]:
    text = _clean_ocr_text(text)
    candidates: list[str] = []

    for pat in NAME_PATTERNS:
        for m in pat.finditer(text):
            if m.lastindex == 2:
                name = f"{m.group(2).strip()} {m.group(1).strip().title()}"
            else:
                name = m.group(1).strip()
            name = re.sub(r"\s{2,}", " ", name)
            if any(tok in STOPWORDS for tok in name.split()):
                continue
            if len(name) < 6:
                continue
            candidates.append(name)

    # Dedup aproximado
    dedup: list[str] = []
    for c in candidates:
        if any(fuzz.ratio(c.lower(), d.lower()) >= 92 for d in dedup):
            continue
        dedup.append(c)
    return dedup


def wikidata_search_person(name: str, session: requests.Session) -> dict | None:
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "pt",
        "uselang": "pt",
        "search": name,
        "limit": 5,
    }
    r = session.get(WIKIDATA_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = data.get("search", [])
    if not results:
        return None
    # heurística: pega o primeiro; guardamos score/descrição para auditoria
    return results[0]


def wikidata_get_claim_labels(qid: str, props: dict[str, str], session: requests.Session) -> dict[str, str | None]:
    url = WIKIDATA_ENTITY.format(qid=qid)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    entity = r.json().get("entities", {}).get(qid, {})
    claims = entity.get("claims", {})

    out: dict[str, str | None] = {}

    def _label(entity_id: str) -> str | None:
        # pega label pt, fallback en
        e = session.get(WIKIDATA_ENTITY.format(qid=entity_id), timeout=30).json().get("entities", {}).get(entity_id, {})
        labels = e.get("labels", {})
        return (labels.get("pt") or labels.get("en") or {}).get("value")

    for col, prop in props.items():
        values = []
        for c in claims.get(prop, [])[:3]:
            try:
                dv = c["mainsnak"]["datavalue"]["value"]
                if isinstance(dv, dict) and "id" in dv:
                    values.append(dv["id"])
            except Exception:
                continue
        if not values:
            out[col] = None
            continue
        labels = [x for x in (_label(v) for v in values) if x]
        out[col] = "; ".join(labels) if labels else None
    return out


def ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DIR.mkdir(parents=True, exist_ok=True)


def run_ocr(images: list[Path], lang: str, crop: bool) -> pd.DataFrame:
    rows = []
    for img_path in images:
        res = ocr_image(img_path, lang=lang, crop=crop)
        txt = _clean_ocr_text(res.text)
        out_txt = OCR_DIR / f"{img_path.stem}.txt"
        out_txt.write_text(txt + "\n", encoding="utf-8")
        rows.append(
            {
                "image": img_path.name,
                "ocr_method": res.method,
                "mean_conf": res.mean_conf,
                "text": txt,
            }
        )
    df = pd.DataFrame(rows)
    (DERIVED_DIR / "combined_ocr.txt").write_text("\n\n".join(df["text"].tolist()) + "\n", encoding="utf-8")
    df.drop(columns=["text"]).to_csv(DERIVED_DIR / "ocr_manifest.csv", index=False)
    return df


def run_name_extraction(ocr_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in ocr_df.iterrows():
        names = extract_names(r["text"])
        for n in names:
            # captura contexto
            ctx = r["text"]
            rows.append({"name": n, "image": r["image"], "context": ctx[:500]})

    df = pd.DataFrame(rows)
    if df.empty:
        out = pd.DataFrame(columns=["name", "occurrences", "images"])
        out.to_csv(DERIVED_DIR / "people_base.csv", index=False)
        return out

    # agregação
    agg = (
        df.groupby("name")
        .agg(
            occurrences=("image", "count"),
            images=("image", lambda s: "; ".join(sorted(set(s)))),
            sample_context=("context", "first"),
        )
        .reset_index()
        .sort_values(["occurrences", "name"], ascending=[False, True])
    )

    agg.to_csv(DERIVED_DIR / "people_base.csv", index=False)
    return agg


def run_wikidata_enrichment(people_df: pd.DataFrame, rate_limit_s: float = 0.5) -> pd.DataFrame:
    props = {
        "genero_wikidata": "P21",
        "grupo_etnico_wikidata": "P172",
        "orientacao_sexual_wikidata": "P91",
    }

    sess = requests.Session()
    sess.headers.update({"User-Agent": "safatle-course-data-viz/1.0 (educational; contact: local)"})

    rows = []
    for _, r in people_df.iterrows():
        name = r["name"]
        entry = None
        try:
            entry = wikidata_search_person(name, sess)
        except Exception:
            entry = None

        qid = entry.get("id") if entry else None
        label = entry.get("label") if entry else None
        desc = entry.get("description") if entry else None
        score = entry.get("match", {}).get("score") if entry else None

        claims = {k: None for k in props}
        if qid:
            try:
                claims = wikidata_get_claim_labels(qid, props, sess)
            except Exception:
                claims = {k: None for k in props}

        rows.append(
            {
                **r.to_dict(),
                "wikidata_qid": qid,
                "wikidata_label": label,
                "wikidata_description": desc,
                "wikidata_score": score,
                **claims,
                # colunas pedidas pela Verô (classe é MUITO inferencial → deixo como desconhecido)
                "raca": claims.get("grupo_etnico_wikidata") or "desconhecido",
                "classe": "desconhecido",
                "genero": claims.get("genero_wikidata") or "desconhecido",
                "orientacao_sexual": claims.get("orientacao_sexual_wikidata") or "desconhecido",
                "fontes": (f"https://www.wikidata.org/wiki/{qid}" if qid else ""),
            }
        )
        time.sleep(rate_limit_s)

    out = pd.DataFrame(rows)
    out.to_csv(DERIVED_DIR / "people_enriched.csv", index=False)
    out.to_json(DERIVED_DIR / "people_enriched.json", orient="records", force_ascii=False, indent=2)
    return out


def build_web(people_df: pd.DataFrame) -> None:
    WEB_DIR.mkdir(parents=True, exist_ok=True)

    # dados compactos para o front
    people = people_df.fillna("").to_dict(orient="records")
    (WEB_DIR / "people.json").write_text(json.dumps(people, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    html = """<!doctype html>
<html lang=\"pt-br\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Colonialidade como problema — distribuição de categorias (referências)</title>
  <style>
    :root { --bg:#0b0f14; --card:#121826; --text:#e6edf3; --muted:#9aa4af; --accent:#f97316; }
    body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial; background:var(--bg); color:var(--text); }
    .wrap { max-width:1100px; margin: 28px auto; padding: 0 16px; }
    h1 { font-size: 22px; margin: 0 0 8px; }
    p { color:var(--muted); line-height:1.45; margin: 6px 0 16px; }
    .grid { display:grid; grid-template-columns: repeat(12, 1fr); gap: 12px; }
    .card { background:var(--card); border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:14px; }
    .card h2 { font-size:14px; margin:0 0 8px; color: var(--text); }
    .span-6 { grid-column: span 6; }
    .span-12 { grid-column: span 12; }
    table { width:100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid rgba(255,255,255,0.08); padding: 8px 6px; text-align:left; vertical-align:top; }
    th { color: var(--muted); font-weight: 600; }
    a { color: #7dd3fc; text-decoration: none; }
    .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; background: rgba(249,115,22,0.12); border: 1px solid rgba(249,115,22,0.25); color: #fdba74; font-size: 12px; }
    @media (max-width: 900px){ .span-6{ grid-column: span 12; } }
  </style>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js\"></script>
</head>
<body>
  <div class=\"wrap\">
    <h1>Colonialidade como problema — distribuição de categorias sociais nas referências</h1>
    <p>
      Fonte primária: OCR das imagens do Drive. Atributos (gênero/grupo étnico/orientação sexual) só aparecem quando constam em fontes públicas (Wikidata). O restante fica como <span class=\"pill\">desconhecido</span>.
    </p>

    <div class=\"grid\">
      <div class=\"card span-6\"><h2>Gênero</h2><canvas id=\"cGenero\"></canvas></div>
      <div class=\"card span-6\"><h2>Raça / grupo étnico (Wikidata)</h2><canvas id=\"cRaca\"></canvas></div>
      <div class=\"card span-6\"><h2>Orientação sexual (Wikidata)</h2><canvas id=\"cOrient\"></canvas></div>
      <div class=\"card span-6\"><h2>Classe (não inferida automaticamente)</h2><canvas id=\"cClasse\"></canvas></div>

      <div class=\"card span-12\">
        <h2>Tabela (auditável)</h2>
        <div style=\"overflow:auto\">
          <table id=\"tPeople\">
            <thead>
              <tr>
                <th>Nome</th>
                <th>Ocorrências</th>
                <th>Imagens</th>
                <th>Gênero</th>
                <th>Raça/grupo étnico</th>
                <th>Orientação</th>
                <th>Classe</th>
                <th>Fonte</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

<script>
async function main(){
  const res = await fetch('./people.json');
  const people = await res.json();

  function norm(v){
    v = (v || '').trim();
    return v ? v : 'desconhecido';
  }
  function countBy(key){
    const m = new Map();
    for (const p of people){
      const k = norm(p[key]);
      m.set(k, (m.get(k) || 0) + 1);
    }
    // ordena: desconhecido no fim, resto desc
    const entries = Array.from(m.entries());
    entries.sort((a,b)=>{
      if (a[0]==='desconhecido' && b[0]!=='desconhecido') return 1;
      if (b[0]==='desconhecido' && a[0]!=='desconhecido') return -1;
      return b[1]-a[1];
    });
    return entries;
  }
  function makePie(canvasId, key){
    const entries = countBy(key);
    const labels = entries.map(e=>e[0]);
    const data = entries.map(e=>e[1]);
    const colors = labels.map((l,i)=> l==='desconhecido' ? 'rgba(148,163,184,0.55)' : `hsl(${(i*53)%360} 80% 60% / 0.75)`);

    new Chart(document.getElementById(canvasId), {
      type: 'doughnut',
      data: { labels, datasets: [{ data, backgroundColor: colors, borderColor: 'rgba(255,255,255,0.08)', borderWidth: 1 }] },
      options: {
        plugins: { legend: { labels: { color: '#e6edf3' } } },
        cutout: '55%'
      }
    });
  }

  makePie('cGenero', 'genero');
  makePie('cRaca', 'raca');
  makePie('cOrient', 'orientacao_sexual');
  makePie('cClasse', 'classe');

  const tbody = document.querySelector('#tPeople tbody');
  for (const p of people){
    const tr = document.createElement('tr');
    const src = p.fontes ? `<a href="${p.fontes}" target="_blank" rel="noreferrer">Wikidata</a>` : '';
    tr.innerHTML = `
      <td>${p.name || ''}</td>
      <td>${p.occurrences ?? ''}</td>
      <td>${p.images || ''}</td>
      <td>${norm(p.genero)}</td>
      <td>${norm(p.raca)}</td>
      <td>${norm(p.orientacao_sexual)}</td>
      <td>${norm(p.classe)}</td>
      <td>${src}</td>
    `;
    tbody.appendChild(tr);
  }
}
main();
</script>
</body>
</html>
"""

    (WEB_DIR / "index.html").write_text(html, encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="por", help="idioma do OCR (tesseract)")
    parser.add_argument("--no-crop", action="store_true", help="não recortar topo/rodapé")
    parser.add_argument("--skip-ocr", action="store_true", help="pula OCR (usa arquivos existentes em data/ocr)")
    parser.add_argument("--skip-wikidata", action="store_true", help="pula enriquecimento via Wikidata")
    args = parser.parse_args(argv)

    ensure_dirs()

    images = _iter_images(IMAGES_DIR)
    if not images:
        print(f"Nenhuma imagem encontrada em {IMAGES_DIR}", file=sys.stderr)
        return 2

    ocr_rows = []
    if args.skip_ocr:
        # reconstitui textos a partir dos .txt existentes
        for img_path in images:
            txt_path = OCR_DIR / f"{img_path.stem}.txt"
            if not txt_path.exists():
                print(f"Faltando OCR: {txt_path}", file=sys.stderr)
                return 3
            ocr_rows.append({"image": img_path.name, "ocr_method": "existing", "mean_conf": None, "text": txt_path.read_text(encoding='utf-8')})
        ocr_df = pd.DataFrame(ocr_rows)
    else:
        ocr_df = run_ocr(images, lang=args.lang, crop=not args.no_crop)

    base_people = run_name_extraction(ocr_df)

    if base_people.empty:
        # ainda assim gera web vazia
        empty = pd.DataFrame(columns=["name", "occurrences", "images", "genero", "raca", "classe", "orientacao_sexual", "fontes"])
        empty.to_json(WEB_DIR / "people.json", orient="records", force_ascii=False, indent=2)
        build_web(empty)
        return 0

    if args.skip_wikidata:
        # cria colunas esperadas sem inventar
        out = base_people.copy()
        out["genero"] = "desconhecido"
        out["raca"] = "desconhecido"
        out["classe"] = "desconhecido"
        out["orientacao_sexual"] = "desconhecido"
        out["fontes"] = ""
        out.to_csv(DERIVED_DIR / "people_enriched.csv", index=False)
        out.to_json(DERIVED_DIR / "people_enriched.json", orient="records", force_ascii=False, indent=2)
        build_web(out)
        return 0

    enriched = run_wikidata_enrichment(base_people)
    build_web(enriched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
