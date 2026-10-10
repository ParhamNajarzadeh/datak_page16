"""CPU-only, provenance-preserving exploratory analysis of Persian newspapers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import platform
import random
import re
import statistics
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path

import psutil
import pymupdf as fitz
from PIL import Image, ImageDraw

DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
LETTERS = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک"})
NUMBER_WORDS = {"اول": 1, "یک": 1, "دوم": 2, "دو": 2, "سوم": 3,
                "سه": 3, "چهارم": 4, "چهار": 4, "چهاردهم": 14,
                "شانزدهم": 16, "آخر": None, "مقابل": None}
PAGE_TOKEN = r"(?:\d{1,3}|اول|یک|دوم|دو|سوم|سه|چهارم|چهار|چهاردهم|شانزدهم|آخر|مقابل)"
MARKER = re.compile(
    rf"(?P<kind>ادامه|بقیه|باقی)\s*(?:[‌\s]*ی)?\s*"
    rf"(?P<direction>در|از)?\s*صفحه(?:[‌\s]*[یۀه])?\s*"
    rf"(?P<page>{PAGE_TOKEN})(?!\d)")
PREFIX_MARKER = re.compile(
    r"(?P<page>\d{1,4})\s+(?P<kind>ادامه|بقیه|باقی)\s+"
    r"(?P<direction>در|از)\s+صفحه(?:[‌\s]*[یۀه])?")
DATE = re.compile(r"(?<!\d)(?:13|14)\d{2}\s*[/ـ.\-]\s*\d{1,2}\s*[/ـ.\-]\s*\d{1,2}(?!\d)")


def normalize(text: str) -> str:
    # Raw text is always saved separately. No spelling or word repair.
    return unicodedata.normalize("NFC", text).translate(LETTERS).translate(DIGITS)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


class MemorySampler:
    """Sample current process tree RSS; this is an approximation, not OS peak RSS."""
    def __enter__(self):
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        parent = psutil.Process()
        while not self.stop.is_set():
            total = 0
            for proc in [parent, *parent.children(recursive=True)]:
                try:
                    total += proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.peak = max(self.peak, total)
            self.stop.wait(0.05)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def coverage(rects, width, height, grid=100):
    """Approximate union area on a grid; overlapping boxes are not double counted."""
    occupied = set()
    for x0, y0, x1, y1 in rects:
        left, right = max(0, int(x0 / width * grid)), min(grid, math.ceil(x1 / width * grid))
        top, bottom = max(0, int(y0 / height * grid)), min(grid, math.ceil(y1 / height * grid))
        occupied.update((x, y) for x in range(left, right) for y in range(top, bottom))
    return len(occupied) / grid ** 2


def text_quality(text):
    chars = [c for c in text if not c.isspace()]
    n = len(chars)
    arabic = sum("ARABIC" in unicodedata.name(c, "") and c.isalpha() for c in chars)
    bad = sum(c == "�" or unicodedata.category(c) in {"Co", "Cs"} or
              (unicodedata.category(c) == "Cc") for c in chars)
    return {"characters": len(text), "nonspace_characters": n,
            "arabic_script_ratio": arabic / n if n else 0,
            "suspicious_character_ratio": bad / n if n else 0,
            "replacement_characters": text.count("�"),
            "zwnj_count": text.count("\u200c")}


def inspect_page(page, issue_id, number):
    start = time.perf_counter()
    # TEXT excludes image payloads: do not inflate memory with embedded image bytes.
    data = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT, sort=False)
    blocks, lines, fonts, markers = [], [], Counter(), []
    for bi, block in enumerate(data["blocks"]):
        if block["type"] != 0:
            continue
        block_id = f"{issue_id}-p{number:03d}-b{bi:04d}"
        block_lines = []
        for li, line in enumerate(block["lines"]):
            raw = "".join(span["text"] for span in line["spans"])
            line_id = f"{block_id}-l{li:03d}"
            sizes = [span["size"] for span in line["spans"]]
            record = {"line_id": line_id, "block_id": block_id, "pdf_page": number,
                      "bbox_pdf_pt": list(line["bbox"]), "text_raw": raw,
                      "text_normalized": normalize(raw), "direction": list(line["dir"]),
                      "font_size_max": max(sizes, default=0), "spans": line["spans"]}
            lines.append(record)
            block_lines.append(raw)
            for span in line["spans"]:
                fonts[(span["font"], round(span["size"], 2))] += len(span["text"])
        raw = "\n".join(block_lines)
        blocks.append({"block_id": block_id, "pdf_page": number,
                       "bbox_pdf_pt": list(block["bbox"]), "text_raw": raw,
                       "text_normalized": normalize(raw), "line_count": len(block_lines),
                       "quality": text_quality(raw), "type": "unclassified_text"})
        # Across lines within one PDF block; cross-block matches are deliberately not inferred.
        normalized = normalize(raw)
        matches = [(m, "following_number") for m in MARKER.finditer(normalized)]
        matches += [(m, "preceding_number_needs_review") for m in PREFIX_MARKER.finditer(normalized)
                    if not any(m.start() < old.end() and old.start() < m.end() for old, _ in matches)]
        for match, pattern in matches:
            token = match["page"]
            target = int(token) if token.isdigit() else NUMBER_WORDS.get(token)
            # Some PDF spans duplicate printed digits (14 -> 1414). Preserve ambiguity.
            number_candidates = []
            if target is not None and target > 200:
                if len(token) % 2 == 0 and token[:len(token)//2] == token[len(token)//2:]:
                    number_candidates.append(int(token[:len(token)//2]))
                target = None
            markers.append({"marker_id": f"{block_id}-m{len(markers):03d}",
                            "source_file": page.parent.name, "pdf_page": number,
                            "block_id": block_id, "bbox_pdf_pt": list(block["bbox"]),
                            "matched_normalized_text": match.group(), "block_text_raw": raw,
                            "direction": {"در": "to", "از": "from"}.get(match["direction"], "unknown"),
                            "referenced_printed_page": target, "page_token": token,
                            "number_candidates_needs_review": number_candidates,
                            "extraction_pattern": pattern,
                            "editorial_or_advertisement": "needs_review"})
    extract_seconds = time.perf_counter() - start
    image_start = time.perf_counter()
    images = []
    for ii, image in enumerate(page.get_image_info(xrefs=True)):
        bbox = list(image["bbox"])
        # transform vectors capture rotation/skew; bbox alone would understate effective DPI.
        a, b, c, d, _, _ = image["transform"]
        placed_width, placed_height = math.hypot(a, b), math.hypot(c, d)
        images.append({"image_id": f"{issue_id}-p{number:03d}-i{ii:03d}",
                       "pdf_page": number, "xref": image.get("xref", 0),
                       "bbox_pdf_pt": bbox, "width_px": image["width"],
                       "height_px": image["height"],
                       "effective_dpi_x": image["width"] * 72 / placed_width if placed_width else None,
                       "effective_dpi_y": image["height"] * 72 / placed_height if placed_height else None,
                       "area_fraction_bbox": max(0, bbox[2]-bbox[0]) * max(0, bbox[3]-bbox[1]) / page.rect.get_area()})
    raw_text = "\n\n".join(b["text_raw"] for b in blocks)
    quality = text_quality(raw_text)
    # Triage only: these heuristics cannot measure CER or certify a usable text layer.
    if quality["nonspace_characters"] < 50:
        triage = "little_or_no_text"
    elif quality["suspicious_character_ratio"] > 0.01 or quality["arabic_script_ratio"] < 0.2:
        triage = "inspect_text_layer"
    else:
        triage = "text_present_needs_gold_check"
    border_lines = [line for line in lines if line["bbox_pdf_pt"][1] < page.rect.height * .12 or
                    line["bbox_pdf_pt"][3] > page.rect.height * .9]
    printed_candidates = []
    for line in border_lines:
        token = normalize(line["text_raw"]).strip()
        if re.fullmatch(r"\d{1,3}", token) and 1 <= int(token) <= 200:
            printed_candidates.append({"number": int(token), "line_id": line["line_id"],
                                       "bbox_pdf_pt": line["bbox_pdf_pt"], "verified": False})
    row = {"pdf_page": number, "width_pt": page.rect.width, "height_pt": page.rect.height,
           "rotation": page.rotation, **quality, "text_layer_triage": triage,
           "text_blocks": len(blocks), "text_lines": len(lines), "image_placements": len(images),
           "text_bbox_coverage_approx": coverage([b["bbox_pdf_pt"] for b in blocks], page.rect.width, page.rect.height),
           "image_bbox_coverage_approx": coverage([i["bbox_pdf_pt"] for i in images], page.rect.width, page.rect.height),
           "continuation_markers": len(markers), "printed_page_verified": None,
           "printed_page_candidates": printed_candidates,
           "numeric_date_candidates": DATE.findall(normalize(raw_text)),
           "extract_seconds": extract_seconds,
           "image_inspection_seconds": time.perf_counter() - image_start}
    return row, blocks, lines, images, fonts, markers, raw_text


def render_overlay(page, blocks, images, markers, out: Path, max_edge):
    scale = max_edge / max(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    canvas = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    canvas.save(out / "page.jpg", quality=88)
    draw = ImageDraw.Draw(canvas)
    # page.get_text uses unrotated coordinates; pixmap uses rotated page coordinates.
    def coords(bbox):
        r = fitz.Rect(bbox) * page.rotation_matrix
        return (r.x0*scale, r.y0*scale, r.x1*scale, r.y1*scale)
    marked = {m["block_id"] for m in markers}
    for item in images:
        draw.rectangle(coords(item["bbox_pdf_pt"]), outline="#eea000", width=2)
    for block in blocks:
        rect = coords(block["bbox_pdf_pt"])
        color = "#ef3340" if block["block_id"] in marked else "#00a0ff"
        draw.rectangle(rect, outline=color, width=2)
        label = block["block_id"].split("-")[-1]
        draw.text((rect[0]+2, rect[1]+2), label, fill=color, stroke_width=1, stroke_fill="white")
    canvas.save(out / "overlay.jpg", quality=90)


def marker_candidates(markers, verified_map):
    """Generate reciprocal candidates, never assign articles or assume PDF=print page."""
    result = []
    for source in markers:
        if source["direction"] != "to" or source["referenced_printed_page"] is None:
            continue
        source_printed = verified_map.get(source["pdf_page"])
        if source_printed is None:
            continue
        for target in markers:
            if target["direction"] == "from" and target["referenced_printed_page"] == source_printed and \
                    verified_map.get(target["pdf_page"]) == source["referenced_printed_page"]:
                result.append({"from_marker": source["marker_id"], "to_marker": target["marker_id"],
                               "from_pdf_page": source["pdf_page"], "to_pdf_page": target["pdf_page"],
                               "evidence": "reciprocal_page_markers_only", "status": "candidate_needs_review"})
    return result


def analyze_file(path, output, args, page_maps):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    issue_id = f"{path.stem}-{digest[:8]}"
    directory = output / issue_id
    directory.mkdir(parents=True, exist_ok=True)
    rows, all_markers, all_images, font_counts, annotation = [], [], [], Counter(), []
    rng = random.Random(args.seed)
    verified_map = {int(k): int(v) for k, v in page_maps.get(path.name, {}).items()}
    with MemorySampler() as memory:
        started = time.perf_counter()
        with fitz.open(path) as doc:
            if doc.needs_pass:
                raise ValueError(f"Encrypted PDF requires password: {path}")
            for number, page in enumerate(doc, 1):
                page_started = time.perf_counter()
                pdir = directory / f"page-{number:03d}"
                pdir.mkdir(exist_ok=True)
                row, blocks, lines, images, fonts, markers, raw = inspect_page(page, issue_id, number)
                row["printed_page_verified"] = verified_map.get(number)
                write_json(pdir / "blocks.json", blocks)
                write_json(pdir / "lines.json", lines)
                (pdir / "text-raw.txt").write_text(raw, encoding="utf-8")
                (pdir / "text-normalized.txt").write_text(normalize(raw), encoding="utf-8")
                render_start = time.perf_counter()
                if not args.no_render:
                    render_overlay(page, blocks, images, markers, pdir, args.render_max_edge)
                row["render_seconds"] = time.perf_counter() - render_start
                eligible = [line for line in lines if len(line["text_raw"].strip()) >= 15]
                sampled = rng.sample(eligible, min(args.sample_lines, len(eligible)))
                for index, line in enumerate(sampled):
                    crop_relative = ""
                    if not args.no_render:
                        rect = (fitz.Rect(line["bbox_pdf_pt"]) + (-2, -2, 2, 2)) & page.rect
                        if not rect.is_empty:
                            crop_relative = f"page-{number:03d}/sample-{index:02d}.png"
                            page.get_pixmap(matrix=fitz.Matrix(args.crop_dpi/72, args.crop_dpi/72),
                                            clip=rect, alpha=False).save(directory / crop_relative)
                    annotation.append({"source_file": path.name, "pdf_page": number,
                                       "line_id": line["line_id"], "bbox_pdf_pt": line["bbox_pdf_pt"],
                                       "image_path": crop_relative, "prediction": line["text_raw"],
                                       "reference": "", "reviewed": "", "notes": ""})
                row["page_wall_seconds_including_exports"] = time.perf_counter() - page_started
                rows.append(row)
                all_markers.extend(markers)
                all_images.extend(images)
                font_counts.update(fonts)
            pdf_metadata = doc.metadata
        measured_seconds = time.perf_counter() - started
    write_csv(directory / "pages.csv", rows)
    write_json(directory / "pages.json", rows)
    write_csv(directory / "images.csv", all_images,
              ["image_id", "pdf_page", "xref", "bbox_pdf_pt", "width_px", "height_px",
               "effective_dpi_x", "effective_dpi_y", "area_fraction_bbox"])
    write_json(directory / "markers.json", all_markers)
    write_csv(directory / "markers.csv", all_markers,
              ["marker_id", "source_file", "pdf_page", "block_id", "bbox_pdf_pt",
               "matched_normalized_text", "block_text_raw", "direction",
               "referenced_printed_page", "page_token", "number_candidates_needs_review",
               "extraction_pattern", "editorial_or_advertisement"])
    candidates = marker_candidates(all_markers, verified_map)
    write_json(directory / "reciprocal-candidates.json", candidates)
    write_csv(directory / "annotation-lines.csv", annotation,
              ["source_file", "pdf_page", "line_id", "bbox_pdf_pt", "image_path",
               "prediction", "reference", "reviewed", "notes"])
    write_csv(directory / "fonts.csv", [{"font": name, "size_pt": size, "characters": count}
                                         for (name, size), count in font_counts.most_common()])
    times = sorted(row["page_wall_seconds_including_exports"] for row in rows)
    summary = {"source_file": str(path.resolve()), "sha256": digest, "issue_id": issue_id,
               "directory": issue_id, "page_count": len(rows), "pdf_metadata": pdf_metadata,
               "text_characters": sum(r["characters"] for r in rows),
               "triage_counts": dict(Counter(r["text_layer_triage"] for r in rows)),
               "continuation_markers": len(all_markers), "reciprocal_candidates": len(candidates),
               "verified_page_mapping_count": len(verified_map),
               "inspection_export_wall_seconds": measured_seconds,
               "page_seconds_p50": statistics.median(times) if times else None,
               "page_seconds_p95": times[max(0, math.ceil(.95*len(times))-1)] if times else None,
               "sampled_process_tree_peak_rss_mb": memory.peak / 1024**2,
               "timing_scope": "PDF inspection + render/crops + page exports; excludes OCR/layout model/stitching and final reports"}
    write_json(directory / "summary.json", summary)
    return summary, rows, all_markers


def report(output, results, environment):
    esc = lambda v: html.escape(str(v))
    sections = []
    for summary, pages, markers in results:
        folder = summary["directory"]
        cards = []
        for page in pages:
            number = page["pdf_page"]
            image = f"{folder}/page-{number:03d}/overlay.jpg"
            preview = f'<a href="{image}"><img loading="lazy" src="{image}"></a>' if (output / image).exists() else ""
            page_markers = [m for m in markers if m["pdf_page"] == number]
            labels = "<br>".join(esc(m["matched_normalized_text"]) for m in page_markers) or "—"
            cards.append(f'''<article><h3>صفحهٔ PDF {number}</h3>{preview}
              <p>متن: {page['characters']:,} نویسه؛ بلوک: {page['text_blocks']}؛ تصویر: {page['image_placements']}</p>
              <p>وضعیت اولیه: <code>{esc(page['text_layer_triage'])}</code></p>
              <p>پوشش تقریبی جعبه‌های متن: {page['text_bbox_coverage_approx']:.1%}</p>
              <p>نشانگرها: {labels}</p>
              <a href="{folder}/page-{number:03d}/text-raw.txt">متن خام</a> ·
              <a href="{folder}/page-{number:03d}/blocks.json">بلوک‌ها و مختصات</a></article>''')
        sections.append(f'''<section><h2>{esc(Path(summary['source_file']).name)}</h2>
          <p>{summary['page_count']} صفحه؛ {summary['text_characters']:,} نویسه؛
          {summary['continuation_markers']} نشانگر ادامه؛ زمان بررسی و خروجی:
          {summary['inspection_export_wall_seconds']:.2f} ثانیه؛ RSS نمونه‌برداری‌شده:
          {summary['sampled_process_tree_peak_rss_mb']:.1f} MB</p>
          <p><a href="{folder}/pages.csv">آمار صفحات CSV</a> ·
          <a href="{folder}/images.csv">تصاویر و DPI</a> ·
          <a href="{folder}/markers.csv">نشانگرها</a> ·
          <a href="{folder}/annotation-lines.csv">نمونهٔ رونویسی</a></p>
          <div class="grid">{''.join(cards)}</div></section>''')
    (output / "report.html").write_text(f'''<!doctype html><html lang="fa" dir="rtl">
      <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
      <title>EDA — Page 16</title><style>
      body{{font-family:Tahoma,Arial,sans-serif;background:#f2f5f9;color:#192536;margin:28px;line-height:1.8}}
      .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}}
      article{{background:white;padding:16px;border-radius:12px;overflow-wrap:anywhere}}
      img{{width:100%;max-height:600px;object-fit:contain}} code{{direction:ltr;unicode-bidi:embed}}
      a{{color:#1261ad}} .note{{background:#fff3d6;padding:16px;border-radius:10px}}
      </style><h1>بررسی اکتشافی آرشیو روزنامه — Page 16</h1>
      <div class="note">وجود متن معادل صحت متن، ترتیب خواندن یا مقالهٔ کامل نیست.
      این بررسی OCR و مدل layout اجرا نمی‌کند و زمان آن benchmark سامانهٔ نهایی نیست.
      CER، IoU و دقت دوختن بدون برچسب انسانی محاسبه نمی‌شوند.
      شماره‌های چاپی حدس‌زده‌شده تأیید نشده‌اند؛ تطبیق ادامه‌ها فقط با نگاشت تأییدشده تولید می‌شود.
      آبی: بلوک متن خام PDF؛ قرمز: بلوک دارای نشانگر؛ نارنجی: تصویر.
      بلوک PDF الزاماً بلوک معنایی یا مقاله نیست.</div>
      {''.join(sections)}<h2>محیط اجرا</h2><pre dir="ltr">{esc(json.dumps(environment, ensure_ascii=False, indent=2))}</pre>
      </html>''', encoding="utf-8")


def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(char_a != char_b)))
        previous = current
    return previous[-1]


def cer_command(args):
    totals = {"strict": {"edits": 0, "reference_characters": 0},
              "normalized": {"edits": 0, "reference_characters": 0}}
    count = 0
    with args.annotations.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("reviewed", "").strip().lower() not in {"1", "true", "yes"}:
                continue
            reference, prediction = row["reference"], row["prediction"]
            count += 1
            for mode in totals:
                ref, pred = (normalize(reference), normalize(prediction)) if mode == "normalized" else (reference, prediction)
                totals[mode]["edits"] += edit_distance(ref, pred)
                totals[mode]["reference_characters"] += len(ref)
    for values in totals.values():
        n = values["reference_characters"]
        values["cer"] = values["edits"] / n if n else None
    result = {"reviewed_lines": count, "metrics": totals,
              "normalized_policy": "NFC; Arabic/Persian yeh/kaf; digit shapes; spaces and ZWNJ preserved"}
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="Inspect PDFs and export HTML/CSV/JSON")
    analyze.add_argument("pdfs", nargs="+", type=Path)
    analyze.add_argument("--output", type=Path, default=Path("outputs"))
    analyze.add_argument("--page-map", type=Path, help="JSON: file name -> PDF page -> verified printed page")
    analyze.add_argument("--render-max-edge", type=int, default=1800)
    analyze.add_argument("--crop-dpi", type=int, default=200)
    analyze.add_argument("--sample-lines", type=int, default=5)
    analyze.add_argument("--seed", type=int, default=16)
    analyze.add_argument("--no-render", action="store_true")
    cer = sub.add_parser("cer", help="Compute CER on human-reviewed annotation CSV")
    cer.add_argument("annotations", type=Path)
    cer.add_argument("--output", type=Path, default=Path("cer.json"))
    args = parser.parse_args()
    if args.command == "cer":
        cer_command(args)
        return
    if args.render_max_edge <= 0 or args.crop_dpi <= 0 or args.sample_lines < 0:
        parser.error("Render size/DPI must be positive; sample count must be non-negative")
    for path in args.pdfs:
        if not path.is_file():
            parser.error(f"PDF not found: {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    page_maps = json.loads(args.page_map.read_text(encoding="utf-8-sig")) if args.page_map else {}
    environment = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "python": platform.python_version(), "platform": platform.platform(),
                   "processor": platform.processor(), "physical_cores": psutil.cpu_count(logical=False),
                   "logical_cores": psutil.cpu_count(), "ram_gb": psutil.virtual_memory().total / 1024**3,
                   "pymupdf": fitz.VersionBind, "psutil": psutil.__version__,
                   "configuration": {k: [str(p) for p in v] if k == "pdfs" else
                                     str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   "coordinate_system": "PDF points, unrotated page, top-left origin; 72 pt/inch",
                   "memory_sampling_interval_seconds": .05}
    results = []
    for path in args.pdfs:
        print(f"Analyzing {path.name} ...", flush=True)
        result = analyze_file(path, args.output, args, page_maps)
        results.append(result)
        print(f"  {result[0]['page_count']} pages, {result[0]['inspection_export_wall_seconds']:.2f}s", flush=True)
    write_json(args.output / "environment.json", environment)
    write_json(args.output / "summary.json", [r[0] for r in results])
    report(args.output, results, environment)
    print(f"Report: {(args.output / 'report.html').resolve()}")


if __name__ == "__main__":
    main()
