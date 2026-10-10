# EDA for Page 16

This CPU-only tool is for preliminary inspection of a Persian newspaper PDF. Raw text, coordinates, and provenance are preserved. PDF output is not presented as an article or correct reading order.

## English notebook for presentation

`Page16_EDA.ipynb` contains 16 sections, 33 cells, executed charts and images, computed insights, and architecture/experiment suggestions. Saved outputs are available for viewing; to re-run, the `outputs` folder must be next to the notebook.

```powershell
python -m pip install -r requirements-notebook.txt
python -m jupyterlab Page16_EDA.ipynb
```

In Jupyter, run `Run All Cells`. The data path can be changed via the `OUTPUT_DIR` variable. `build_notebook.py` is the source for reproducing the notebook; running it rebuilds the ipynb file as a version without outputs, so if you have manually edited the notebook, save your changes first.

## Installation and execution — PowerShell

```powershell
python -m pip install -r requirements.txt
python eda.py analyze "C:\Users\Mandish\Desktop\1-7.pdf" "C:\Users\Mandish\Desktop\1-8.pdf" "C:\Users\Mandish\Desktop\1-9.pdf" --output outputs
Start-Process outputs\report.html
```

For a quick check without images and crops:

```powershell
python eda.py analyze "C:\Users\Mandish\Desktop\1-8.pdf" --no-render --output outputs-fast
```

## What is inspected?

- Page count, dimensions, rotation, metadata, and SHA256 of the input file.
- Character/line/block counts, share of Arabic-script characters, suspicious characters, and ZWNJ; these are only **triage**, not CER or quality assurance.
- Blocks, lines, spans, fonts, and bboxes without artificial whole-page sorting.
- Images actually placed on the page, pixel dimensions, and effective DPI from placement transform; including inline images. Effective DPI is not equivalent to scanner DPI. Placement count differs from unique image file count.
- Approximate coverage of the union of bboxes on a 100×100 grid; box coverage is not equivalent to ink coverage, and image coverage alone does not prove scan type.
- Indicators of "continued/remainder/rest in/from page ...", including Persian/Arabic digits and some written-out numbers. The indicator is checked at the block level; other patterns and indicators broken across blocks may be missed.
- The number before the indicator is also checked. In samples, extraction of `1414 بقیه در صفحه` was observed; for repeated numbers outside the range, the definitive destination is `null` and the possible number is recorded in `number_candidates_needs_review`. This case must be confirmed with the image.
- Print-number candidates from standalone numeric lines near the top/bottom of the page; number and date are only suggestions. The newspaper name is separate from possible metadata and is not automatically confirmed.
- Extraction time, image inspection, render, and total page processing/output time; p50/p95 and sampled RSS of the process tree.
- Five reproducible random lines from each page along with crops for human transcription. For fine/hard text analysis, also add targeted samples.

## Output structure

```text
outputs/
  report.html
  environment.json
  summary.json
  <file-stem>-<hash>/
    summary.json
    pages.csv / pages.json
    images.csv
    fonts.csv
    markers.csv / markers.json
    reciprocal-candidates.json
    annotation-lines.csv
    page-001/
      page.jpg / overlay.jpg
      blocks.json / lines.json
      text-raw.txt / text-normalized.txt
      sample-00.png ...
```

Overlay colors: blue = raw PDF text, red = block containing continuation indicator, orange = image. The ID on the block can be matched with `blocks.json`. Data coordinates are in PDF points, with top-left origin and the unrotated page coordinate system; rotation is applied for the display image.

## Inspecting two independent continuations 2 ↔ 14

First review `markers.csv` and the overlays of pages 2 and 14 of `1-8.pdf`. To build reciprocal candidates, a **confirmed** mapping from PDF number to print number is needed. For example, after manual confirmation:

```json
{
  "1-8.pdf": {"2": 2, "9": 9, "11": 11, "14": 14}
}
```

Save it in `page-map.json` and run:

```powershell
python eda.py analyze "C:\Users\Mandish\Desktop\1-8.pdf" --page-map page-map.json --output outputs-mapped
```

`reciprocal-candidates.json` generates all pairs compatible with reciprocal numbers, **not article stitching**. Two continuations to one destination can produce four candidate pairs; this output is useful for ambiguity review. Requirement/advertisement indicators are also recorded and labeled `needs_review`; no indicator is automatically converted into news. "Opposite page" has no numeric destination so that page-layout rules can be independently confirmed.

## Measuring real CER

In `annotation-lines.csv`, enter independent transcription from the crop into `reference` and set `reviewed` to `1`. The `prediction` column is the extracted PDF text; for OCR comparison you can create a separate CSV with the same reference and the OCR engine's prediction. To avoid bias, hide the prediction column while transcribing.

```powershell
python eda.py cer "outputs\<file-stem>-<hash>\annotation-lines.csv" --output cer.json
```

CER is computed with Levenshtein edit distance and total errors divided by total reference length, not the average of line CERs. Strict and NFC + y/k and digit-shape normalization versions are reported; spaces, ZWNJ, and punctuation are not removed. An empty confirmed reference is allowed; if total reference length is zero, CER is undefined and `null`. CER can exceed 1.

## Limitations and timing interpretation

This EDA does not include OCR, detector, article grouping, stitching, or a compressed model. Its timing **does not prove the 2–3 minute system budget**. Inspection/export time includes image generation, crop, and page files; initial hashing and final report generation are outside it. Run `--no-render` to separate visualization cost. Memory is sampled at 50 ms intervals and may miss very short peaks. Runs are sequential and single-worker.

For CER/IoU/stitching accuracy and before/after quantization reporting, human gold and real model runs are needed. Data lacking text is marked in the output but OCR is not run on it. Good text extraction is not necessarily correct order or spelling; fonts with broken Unicode mappings may pass the heuristic. text-normalized is only NFC and letter/digit normalization and does not perform spelling correction or generative rewriting.

PyMuPDF is licensed under AGPL or commercial terms; the product distribution method must be compatible with the chosen license.
