"""PDF -> structured blocks: multi-column reading order, heading-path tracking, table extraction.

Heuristic-based (font size relative to the page's dominant body size decides headings;
horizontal page midpoint decides column). Good enough for typical two-column D&D rulebook
layouts; irregular 3+ column pages or overlapping sidebars may read out of strict order —
acceptable since each block is stored as its own unit, so a misordering doesn't corrupt
retrieval, only strict document-flow order.
"""

import hashlib
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class ExtractedBlock:
    page: int  # 1-indexed
    heading_path: str
    text: str
    block_type: str  # "prose" | "table"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _body_font_size(doc: fitz.Document, sample_pages: int = 20) -> float:
    sizes: dict[float, int] = {}
    for page in doc[: min(sample_pages, doc.page_count)]:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    size = round(span["size"], 1)
                    sizes[size] = sizes.get(size, 0) + len(text)
    if not sizes:
        return 10.0
    return max(sizes, key=lambda s: sizes[s])


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned = [[(cell or "").strip() for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""
    header, *body = cleaned
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _overlaps(bbox_a: tuple, bbox_b: tuple, threshold: float = 0.5) -> bool:
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter_area = (ix1 - ix0) * (iy1 - iy0)
    a_area = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
    return (inter_area / a_area) > threshold


def extract_pdf(path: str) -> tuple[str, int, list[ExtractedBlock], list[int]]:
    """Returns (sha256, page_count, blocks_in_reading_order, low_text_page_numbers)."""
    doc = fitz.open(path)
    sha256 = _sha256_file(path)
    body_size = _body_font_size(doc)

    blocks: list[ExtractedBlock] = []
    low_text_pages: list[int] = []
    heading_major: str | None = None
    heading_minor: str | None = None

    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_num = page_index + 1
        mid_x = page.rect.width / 2

        table_bboxes: list[tuple] = []
        try:
            found = page.find_tables()
            for table in found.tables:
                table_bboxes.append(tuple(table.bbox))
                md = _table_to_markdown(table.extract())
                if md.strip():
                    heading_path = " > ".join(p for p in (heading_major, heading_minor) if p)
                    blocks.append(ExtractedBlock(page_num, heading_path, md, "table"))
        except Exception:
            table_bboxes = []

        positioned: list[tuple[int, float, str, float]] = []
        page_text_len = 0

        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            bbox = tuple(block["bbox"])
            if any(_overlaps(bbox, tb) for tb in table_bboxes):
                continue

            text_parts = []
            max_size = 0.0
            for line in block["lines"]:
                for span in line["spans"]:
                    text_parts.append(span["text"])
                    max_size = max(max_size, span["size"])
            text = "".join(text_parts).strip()
            if not text:
                continue

            page_text_len += len(text)
            column = 0 if (bbox[0] + bbox[2]) / 2 < mid_x else 1
            positioned.append((column, bbox[1], text, max_size))

        if page_text_len < 20:
            low_text_pages.append(page_num)

        positioned.sort(key=lambda t: (t[0], t[1]))

        for _, _, text, max_size in positioned:
            if max_size >= body_size * 1.3:
                heading_major = text
                heading_minor = None
                continue
            if max_size >= body_size * 1.15:
                heading_minor = text
                continue

            heading_path = " > ".join(p for p in (heading_major, heading_minor) if p)
            blocks.append(ExtractedBlock(page_num, heading_path, text, "prose"))

    page_count = doc.page_count
    doc.close()
    return sha256, page_count, blocks, low_text_pages
