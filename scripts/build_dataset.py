#!/usr/bin/env python3

"""Build dataset from course reference images.

Steps:
- OCR Instagram screenshots (crop slide area, enhance contrast)
- Extract cited names (authors) from OCR text
- Output:
  - data/ocr/*.txt (per image)
  - data/derived/people.csv (deduped people + evidence)
  - data/derived/people.json (same as JSON)

Notes on sensitive attributes:
This script intentionally does NOT infer race/class/sexual orientation.
Only fields explicitly available in the source material should be filled.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from rapidfuzz import fuzz


ORANGE_HUE_MIN = 10 / 360  # approx
ORANGE_HUE_MAX = 55 / 360


@dataclass
class Mention:
    image_file: str
    line: str


@dataclass
class PersonRow:
    name: str
    mentions: int
    images: list[str]
    evidence_lines: list[str]
    # Sensitive-ish categories: do not infer. Keep unknown unless explicitly sourced.
    raca: str = "desconhecido"
    classe: str = "desconhecido"
    genero: str = "desconhecido"
    orientacao_sexual: str = "desconhecido"
    fonte_raca: str = ""
    fonte_classe: str = ""
    fonte_genero: str = ""
    fonte_orientacao_sexual: str = ""


def rgb_to_hsv01(r: int, g: int, b: int) -> tuple[float, float, float]:
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    mx = max(rf, gf, bf)
    mn = min(rf, gf, bf)
    diff = mx - mn

    if diff == 0:
        h = 0.0
    elif mx == rf:
        h = ((gf - bf) / diff) % 6
    elif mx == gf:
        h = (bf - rf) / diff + 2
    else:
        h = (rf - gf) / diff + 4
    h /= 6.0

    s = 0.0 if mx == 0 else diff / mx
    v = mx
    return h, s, v


def is_orange_pixel(r: int, g: int, b: int) -> bool:
    h, s, v = rgb_to_hsv01(r, g, b)
    if v < 0.15:
        return False
    if s < 0.20:
        return False
    # handle wrap-around not needed for orange
    return ORANGE_HUE_MIN <= h <= ORANGE_HUE_MAX


def detect_orange_crop(img: Image.Image) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) bounding box likely containing the orange slide.

    Heuristic: compute orange density per row and find the longest contiguous band.
    """

    rgb = img.convert("RGB")
    w, h = rgb.size
    px = rgb.load()

    # Sample columns to keep this fast
    sample_cols = list(range(0, w, max(1, w // 120)))

    densities: list[float] = []
    for y in range(h):
        orange = 0
        for x in sample_cols:
            r, g, b = px[x, y]
            if is_orange_pixel(r, g, b):
                orange += 1
        densities.append(orange / max(1, len(sample_cols)))

    # Rows with enough orange (tolerant)
    threshold = 0.20
    mask = [d >= threshold for d in densities]

    # Find longest contiguous True run
    best = (0, -1)  # (start, end)
    start = None
    for i, ok in enumerate(mask + [False]):
        if ok and start is None:
            start = i
        elif (not ok) and start is not None:
            end = i - 1
            if end - start > best[1] - best[0]:
                best = (start, end)
            start = None

    top, bottom = best
    if bottom < top:
        # fallback: crop out typical Instagram chrome
        top = int(h * 0.18)
        bottom = int(h * 0.78)

    pad = int(h * 0.02)
    top = max(0, top - pad)
    bottom = min(h - 1, bottom + pad)

    # Horizontal crop: most slides span almost full width; cut small margins
    left = int(w * 0.05)
    right = int(w * 0.95)

    return left, top, right, bottom


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    # Upscale a bit
    w, h = img.size
    scale = 2
    img = img.resize((w * scale, h * scale))

    # Convert to grayscale
    img = img.convert("L")

    # Increase contrast
    img = ImageEnhance.Contrast(img).enhance(2.2)
    img = ImageEnhance.Sharpness(img).enhance(1.7)

    # Reduce small noise and sharpen edges
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.SHARPEN)

    return img


def ocr_image(img_path: Path, ocr_lang: str) -> tuple[str, Image.Image, tuple[int, int, int, int]]:
    img = Image.open(img_path)
    crop = detect_orange_crop(img)
    cropped = img.crop(crop)
    pre = preprocess_for_ocr(cropped)

    config = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
    text = pytesseract.image_to_string(pre, lang=ocr_lang, config=config)
    # Normalize whitespace lightly
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text, cropped, crop


AUTHOR_SEGMENT_RE = re.compile(
    r"^(?P<surname>[A-ZÀ-ÖØ-ÝÇÃÕÜÑ][A-ZÀ-ÖØ-ÝÇÃÕÜÑ\- ]{1,60}),\s*(?P<given>[^.]{1,80}?)(?:\.|$)",
    re.UNICODE,
)


def clean_name_segment(seg: str) -> str:
    seg = seg.strip()
    seg = re.sub(r"\s+", " ", seg)
    seg = seg.strip(" ;,\t")
    return seg


def extract_people_from_text(text: str, image_file: str) -> list[Mention]:
    mentions: list[Mention] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Break multi-author lines
        parts = [clean_name_segment(p) for p in re.split(r";", line) if p.strip()]
        for part in parts:
            m = AUTHOR_SEGMENT_RE.match(part)
            if not m:
                continue
            surname_raw = m.group("surname").strip()
            # Guardrails: avoid false positives where OCR turns prose into "SURNAME, given"
            # Typical bibliographic surnames are short (often 1-3 tokens, sometimes with connectors).
            if len(surname_raw.split()) > 4:
                continue
            # Drop roman numerals like "XIX," accidentally parsed as surname
            if re.fullmatch(r"[IVXLCDM]{2,}", surname_raw):
                continue

            surname = surname_raw.title().strip()
            given = m.group("given").strip()
            # Remove role markers / editor notes (OCR sometimes truncates the closing ')')
            given = re.sub(r"\(Org.*$", "", given, flags=re.I).strip()
            given = re.sub(r"\(Ed.*$", "", given, flags=re.I).strip()
            given = re.sub(r"\(Org\.?\)\s*$", "", given, flags=re.I).strip()
            # Drop any remaining trailing parenthetical
            given = re.sub(r"\s*\(.*$", "", given).strip()
            # Some OCR glitches: stray commas
            given = given.strip(" ,")
            # Normalize initials spacing
            given = re.sub(r"\s+", " ", given)
            # Given names in bibliographic entries normally start with a capital letter
            if given and not re.match(r"^[A-ZÀ-ÖØ-Ý]", given):
                continue
            # Another guardrail: given names shouldn't look like a long sentence fragment
            if len(given.split()) > 7:
                continue
            full = f"{given} {surname}".strip()
            # Fix ordering when given is empty (rare)
            if not given:
                full = surname
            mentions.append(Mention(image_file=image_file, line=full))

        # Handle lines like "SUBCOMANDANTE INSURGENTE GALEANO." without comma
        if re.match(r"^[A-ZÀ-ÖØ-ÝÇÃÕÜÑ][A-ZÀ-ÖØ-ÝÇÃÕÜÑ\- ]{6,}$", line):
            # Avoid headers like "MÓDULO" / "REFERÊNCIAS"
            if any(tok in line for tok in ["MÓDULO", "REFERÊNCIAS"]):
                continue
            if len(line.split()) < 2:
                continue
            full = line.title().strip()
            mentions.append(Mention(image_file=image_file, line=full))

    return mentions


def dedupe_people(names: list[str]) -> list[str]:
    """Dedupe names using fuzzy matching (conservative)."""
    canon: list[str] = []

    def similar(a: str, b: str) -> bool:
        score = fuzz.token_sort_ratio(a, b)
        return score >= 93

    for n in sorted(set(names), key=lambda s: (len(s), s)):
        found = False
        for c in canon:
            if similar(n, c):
                found = True
                break
        if not found:
            canon.append(n)
    return sorted(canon)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="/workspace/data/images")
    ap.add_argument("--ocr-dir", default="/workspace/data/ocr")
    ap.add_argument("--crops-dir", default="/workspace/data/derived/crops")
    ap.add_argument("--out-csv", default="/workspace/data/derived/people.csv")
    ap.add_argument("--out-json", default="/workspace/data/derived/people.json")
    ap.add_argument("--ocr-lang", default="por+eng")
    ap.add_argument("--force-ocr", action="store_true")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    ocr_dir = Path(args.ocr_dir)
    crops_dir = Path(args.crops_dir)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    ocr_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
        key=lambda p: p.name,
    )
    if not image_files:
        raise SystemExit(f"Nenhuma imagem encontrada em {images_dir}")

    all_mentions: list[Mention] = []

    for img_path in image_files:
        out_txt = ocr_dir / (img_path.stem + ".txt")
        if out_txt.exists() and not args.force_ocr:
            text = out_txt.read_text(encoding="utf-8", errors="replace")
        else:
            text, cropped, _ = ocr_image(img_path, args.ocr_lang)
            out_txt.write_text(text, encoding="utf-8")
            # Save cropped preview for auditing
            crop_path = crops_dir / (img_path.stem + "__crop.png")
            cropped.save(crop_path)

        all_mentions.extend(extract_people_from_text(text, img_path.name))

    if not all_mentions:
        raise SystemExit("Não consegui extrair nenhum nome (OCR pode ter falhado).")

    # Build initial table (mentions)
    names = [m.line for m in all_mentions]
    canon = dedupe_people(names)

    # Map each mention to nearest canonical name (best match)
    mapped: list[tuple[str, str, str]] = []  # (canonical, image_file, raw)
    for m in all_mentions:
        best = None
        best_score = -1
        for c in canon:
            score = fuzz.token_sort_ratio(m.line, c)
            if score > best_score:
                best_score = score
                best = c
        mapped.append((best or m.line, m.image_file, m.line))

    rows: list[PersonRow] = []
    for person in canon:
        subset = [x for x in mapped if x[0] == person]
        images = sorted(set(x[1] for x in subset))
        evidence = []
        # keep up to 8 evidence lines
        for _, _, raw in subset:
            if raw not in evidence:
                evidence.append(raw)
            if len(evidence) >= 8:
                break

        rows.append(
            PersonRow(
                name=person,
                mentions=len(subset),
                images=images,
                evidence_lines=evidence,
            )
        )

    df = pd.DataFrame([asdict(r) for r in rows]).sort_values(by=["mentions", "name"], ascending=[False, True])
    df.to_csv(out_csv, index=False)

    out_json.write_text(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: OCR em {len(image_files)} imagens")
    print(f"OK: {len(rows)} pessoas (dedupe)")
    print(f"CSV: {out_csv}")
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()
