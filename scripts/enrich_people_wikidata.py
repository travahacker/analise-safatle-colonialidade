#!/usr/bin/env python3

"""Enriquece `data/derived/people.csv` com dados públicos (Wikidata/Wikipedia).

Objetivo: preencher colunas pedidas para visualização (com fontes):
- genero (Wikidata P21)
- raca (aqui: "grupo étnico" do Wikidata P172; *não* é inferência visual)
- orientacao_sexual (Wikidata P91, quando existir)
- classe: não existe de forma confiável/estruturada → usamos duas colunas:
  - classe: permanece "desconhecido" (não inferida)
  - classe_ocupacional: um bucket simples baseado em ocupações (P106)

O script nunca inventa atributos sensíveis: se não houver dado, marca "desconhecido".

Saídas:
- data/derived/people_enriched.csv
- data/derived/people_enriched.json
- web/data/people.json (atualizado para o dashboard)

"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DERIVED = ROOT / "data" / "derived"
WEB_DATA = ROOT / "web" / "data"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
WIKIDATA_PAGE = "https://www.wikidata.org/wiki/{qid}"
WIKIPEDIA_PAGE = "https://{lang}.wikipedia.org/wiki/{title}"


ROMAN_RE = re.compile(r"^(?=[IVXLCDM])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", re.I)


@dataclass(frozen=True)
class WDHit:
    qid: str
    label: str | None
    description: str | None
    score: float | None


def _wd_search(name: str, session: requests.Session, lang: str) -> list[WDHit]:
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": lang,
        "uselang": lang,
        "search": name,
        "limit": 5,
    }
    r = session.get(WIKIDATA_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = data.get("search", [])
    hits: list[WDHit] = []
    for item in results:
        qid = item.get("id")
        if not qid:
            continue
        hits.append(
            WDHit(
                qid=qid,
                label=item.get("label"),
                description=item.get("description"),
                score=(item.get("match", {}) or {}).get("score"),
            )
        )
    return hits


def _entity_json(qid: str, session: requests.Session) -> dict:
    r = session.get(WIKIDATA_ENTITY.format(qid=qid), timeout=30)
    r.raise_for_status()
    return r.json().get("entities", {}).get(qid, {})


def _claim_entity_ids(entity: dict, prop: str, limit: int = 5) -> list[str]:
    claims = (entity.get("claims", {}) or {}).get(prop, [])
    ids: list[str] = []
    for c in claims[:limit]:
        try:
            dv = c["mainsnak"]["datavalue"]["value"]
            if isinstance(dv, dict) and "id" in dv:
                ids.append(dv["id"])
        except Exception:
            continue
    return ids


def _labels_for_qids(qids: list[str], session: requests.Session, lang: str) -> list[str]:
    out: list[str] = []
    for qid in qids:
        try:
            ent = _entity_json(qid, session)
            labels = ent.get("labels", {}) or {}
            val = (labels.get(lang) or labels.get("en") or {}).get("value")
            if val:
                out.append(val)
        except Exception:
            continue
    return out


def _sitelink(entity: dict, langwiki: str) -> str | None:
    # langwiki ex: "ptwiki", "enwiki"
    sl = (entity.get("sitelinks", {}) or {}).get(langwiki)
    if not sl:
        return None
    title = sl.get("title")
    if not title:
        return None
    return WIKIPEDIA_PAGE.format(lang=langwiki.replace("wiki", ""), title=title.replace(" ", "_"))


def _is_human(entity: dict) -> bool:
    # instance of (P31) includes human (Q5)
    return "Q5" in set(_claim_entity_ids(entity, "P31", limit=10))


def _normalize_spaces(name: str) -> str:
    n = (name or "").strip()
    n = re.sub(r"\s+", " ", n)
    return n


def _alternate_queries(name: str) -> list[str]:
    original = _normalize_spaces(name)
    alts = [original]

    # casos conhecidos de OCR/abreviação (checa antes de mexer em iniciais)
    overrides = {
        "G Hegel": "Georg Wilhelm Friedrich Hegel",
        "S Eisenstadt": "Shmuel Eisenstadt",
        "Vladimir Ilitch Lenin": "Vladimir Lenin",
        "Vladimir Lenin": "Vladimir Lenin",
        "Edward W Said": "Edward Said",
        "Edouard Glissant": "Édouard Glissant",
        "Jose Carlos Mariategui": "José Carlos Mariátegui",
        "Bjorn Wittrock": "Björn Wittrock",
        "Homi K Bhabha": "Homi K. Bhabha",
        "Nascimento Abdias": "Abdias do Nascimento",
    }
    if original in overrides:
        alts.insert(0, overrides[original])

    # variação com pontos em iniciais (ajuda desambiguação)
    if re.search(r"\b[A-Z]\b", original):
        alts.append(re.sub(r"\b([A-Z])\b", r"\1.", original))

    # variação removendo iniciais isoladas (só como fallback)
    no_single = re.sub(r"\b([A-Z])\b", "", original)
    no_single = re.sub(r"\s+", " ", no_single).strip()
    if no_single and no_single != original:
        alts.append(no_single)

    # evita numerais romanos como nome
    alts = [a for a in alts if a and not ROMAN_RE.match(a)]

    # dedup mantendo ordem
    seen = set()
    uniq = []
    for a in alts:
        if a.lower() in seen:
            continue
        seen.add(a.lower())
        uniq.append(a)
    return uniq


def pick_best(name: str, session: requests.Session, lang: str) -> tuple[WDHit | None, dict | None, str]:
    best_hit: WDHit | None = None
    best_entity: dict | None = None
    best_query = ""
    best_score = -1e18

    orig_tokens = [t for t in re.split(r"[^\\wÀ-ÖØ-öø-ÿ]+", _normalize_spaces(name).lower()) if len(t) >= 3]

    for q in _alternate_queries(name):
        try:
            hits = _wd_search(q, session, lang=lang)
        except Exception:
            hits = []

        for hit in hits:
            try:
                ent = _entity_json(hit.qid, session)
            except Exception:
                continue

            if not _is_human(ent):
                continue

            sc = float(hit.score) if isinstance(hit.score, (int, float)) else 0.0

            # penaliza quando o rótulo não contém nenhum token do nome (reduz falsos positivos tipo \"JD Vance\")
            label_l = (hit.label or "").lower()
            if orig_tokens and not any(tok in label_l for tok in orig_tokens):
                sc -= 15.0

            # bonus se tem Wikipedia (auditável)
            if _sitelink(ent, "ptwiki") or _sitelink(ent, "enwiki"):
                sc += 10.0

            desc = (hit.description or "").lower()
            # penaliza entradas obviamente fora do tema (heurística simples)
            if any(
                k in desc
                for k in [
                    "football",
                    "footballer",
                    "futebol",
                    "futebolista",
                    "city",
                    "cidade",
                    "state capital",
                    "actor",
                    "actress",
                    "film",
                    "television",
                    "tv",
                    "singer",
                    "rapper",
                    "musician",
                ]
            ):
                sc -= 12.0
            # bonus se parece \"autor/intelectual\" (só por texto público do description)
            if any(k in desc for k in ["philos", "filó", "sociolog", "antrop", "histori", "writer", "escritor", "poeta", "theor", "crític", "professor", "acadêm", "political theorist"]):
                sc += 6.0

            if sc > best_score:
                best_score = sc
                best_hit = hit
                best_entity = ent
                best_query = q

        if best_hit and best_score >= 12.0:
            break

    # Se ficou muito fraco/ambíguo, é melhor não chutar.
    if best_hit and best_score < 5.0:
        return None, None, ""

    return best_hit, best_entity, best_query


def bucket_classe_ocupacional(occupations: list[str]) -> str:
    occ = " ".join(o.lower() for o in occupations)
    if not occ.strip():
        return "desconhecido"

    # heurística simples e transparente
    if any(k in occ for k in ["philosopher", "filóso", "sociologist", "antrop", "historian", "historiador", "professor", "academic", "acadêm", "scientist", "pesquisador", "theorist"]):
        return "academia/intelectual"
    if any(k in occ for k in ["writer", "escritor", "poet", "poeta", "novelist", "romancista", "essayist", "ensaísta"]):
        return "artes/letras"
    if any(k in occ for k in ["politician", "político", "revolutionary", "revolucion", "activist", "ativista", "militant", "militante", "union", "sindical"]):
        return "política/militância"
    if any(k in occ for k in ["economist", "economista"]):
        return "economia"
    if any(k in occ for k in ["lawyer", "jurist", "jurista", "judge", "juiz"]):
        return "direito"

    return "outros"


def enrich(input_csv: Path, out_csv: Path, out_json: Path, web_json: Path, lang: str, rate_limit_s: float) -> None:
    df = pd.read_csv(input_csv).fillna("")

    sess = requests.Session()
    sess.headers.update({"User-Agent": "safatle-course-data-viz/1.0 (educational; contact: local)"})

    rows = []
    for _, r in df.iterrows():
        name = str(r.get("name", "")).strip()

        hit, entity, used_query = pick_best(name, sess, lang=lang)
        qid = hit.qid if hit else None

        genero = "desconhecido"
        raca = "desconhecido"
        orientacao = "desconhecido"
        ocupacoes: list[str] = []

        fonte_genero = ""
        fonte_raca = ""
        fonte_orientacao = ""

        if entity:
            # P21 gender, P172 ethnic group, P91 sexual orientation, P106 occupation
            genero_vals = _labels_for_qids(_claim_entity_ids(entity, "P21"), sess, lang)
            raca_vals = _labels_for_qids(_claim_entity_ids(entity, "P172"), sess, lang)
            orient_vals = _labels_for_qids(_claim_entity_ids(entity, "P91"), sess, lang)
            ocupacoes = _labels_for_qids(_claim_entity_ids(entity, "P106"), sess, lang)

            if genero_vals:
                genero = "; ".join(genero_vals)
                fonte_genero = WIKIDATA_PAGE.format(qid=qid)
            if raca_vals:
                raca = "; ".join(raca_vals)
                fonte_raca = WIKIDATA_PAGE.format(qid=qid)
            if orient_vals:
                orientacao = "; ".join(orient_vals)
                fonte_orientacao = WIKIDATA_PAGE.format(qid=qid)

        classe_ocup = bucket_classe_ocupacional(ocupacoes)

        ptwiki = _sitelink(entity, "ptwiki") if entity else None
        enwiki = _sitelink(entity, "enwiki") if entity else None

        rows.append(
            {
                **r.to_dict(),
                "wikidata_qid": qid or "",
                "wikidata_label": (hit.label if hit else "") or "",
                "wikidata_description": (hit.description if hit else "") or "",
                "wikidata_score": (hit.score if hit and hit.score is not None else ""),
                "wikidata_url": (WIKIDATA_PAGE.format(qid=qid) if qid else ""),
                "wikipedia_pt": ptwiki or "",
                "wikipedia_en": enwiki or "",
                "wikidata_query": used_query or "",
                "ocupacoes_wikidata": "; ".join(ocupacoes) if ocupacoes else "",
                # campos pedidos
                "raca": raca,
                "classe": str(r.get("classe", "desconhecido")) if "classe" in r else "desconhecido",
                "classe_ocupacional": classe_ocup,
                "genero": genero,
                "orientacao_sexual": orientacao,
                "fonte_raca": fonte_raca,
                "fonte_genero": fonte_genero,
                "fonte_orientacao_sexual": fonte_orientacao,
            }
        )

        time.sleep(rate_limit_s)

    out = pd.DataFrame(rows).fillna("")
    out.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(out.to_dict(orient="records"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    web_json.write_text(json.dumps(out.to_dict(orient="records"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA_DERIVED / "people.csv"))
    ap.add_argument("--out-csv", default=str(DATA_DERIVED / "people_enriched.csv"))
    ap.add_argument("--out-json", default=str(DATA_DERIVED / "people_enriched.json"))
    ap.add_argument("--web-json", default=str(WEB_DATA / "people.json"))
    ap.add_argument("--lang", default="pt")
    ap.add_argument("--rate-limit", type=float, default=0.35)
    args = ap.parse_args()

    enrich(
        input_csv=Path(args.input),
        out_csv=Path(args.out_csv),
        out_json=Path(args.out_json),
        web_json=Path(args.web_json),
        lang=args.lang,
        rate_limit_s=float(args.rate_limit),
    )


if __name__ == "__main__":
    main()
