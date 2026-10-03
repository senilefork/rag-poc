"""PDF -> LlamaIndex nodes via docling.

Docling recovers reading order, section headers, table structure (TableFormer) and
caption/footnote attachment. We walk its ``DoclingDocument`` once in reading order,
accumulate prose under the current section header, and emit one node per table.
"""

import re
from pathlib import Path

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, TableItem
from docling_core.types.doc.labels import DocItemLabel
from llama_index.core.schema import TextNode

CHUNK_WORDS = 300
CHUNK_OVERLAP = 50

TEXT_LABELS = frozenset(
    {
        DocItemLabel.TEXT,
        DocItemLabel.PARAGRAPH,
        DocItemLabel.LIST_ITEM,
    }
)
SECTION_LABELS = frozenset({DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE})

_converter: DocumentConverter | None = None


def build_converter() -> DocumentConverter:
    """Converter using the pypdfium PDF backend.

    The default docling-parse backend mis-OCRs small cells whose text runs close
    to table rules -- in avino.pdf it turns the "Gold (%)" recovery label into
    "G0%) (%)". pypdfium reads it correctly and restores word spacing that
    docling-parse drops ("andpresentedintherecent" -> "and presented in ...").
    """
    options = PdfPipelineOptions()
    options.do_table_structure = True
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options, backend=PyPdfiumDocumentBackend
            )
        }
    )


def load_document(pdf_path: str) -> DoclingDocument:
    """Convert a PDF once per process and reuse the converter's cached models."""
    global _converter
    if _converter is None:
        _converter = build_converter()
    return _converter.convert(pdf_path).document


def page_of(item) -> int:
    """1-based page number of an item's first provenance record."""
    prov = getattr(item, "prov", None)
    if not prov:
        return 1
    return int(prov[0].page_no)


def table_to_text(item: TableItem, doc: DoclingDocument) -> str:
    """Markdown body, then any footnotes docling attached to the table.

    `export_to_markdown` already renders the caption, so it is only prepended
    when the export omits it.
    """
    body = item.export_to_markdown(doc).strip()
    caption = item.caption_text(doc).strip()

    parts = []
    if caption and caption not in body:
        parts.append(caption)
    if body:
        parts.append(body)
    for ref in item.footnotes:
        note = ref.resolve(doc).text.strip()
        if note:
            parts.append(note)
    return "\n\n".join(parts)


def table_rows_to_texts(item: TableItem, doc: DoclingDocument) -> list[str]:
    """Natural-language row nodes for better retrieval.

    Break the table into one self-contained line per data row. Each line
    contains the table caption, the current group header and row label
    (joined by an em dash if they differ meaningfully), and period=value
    pairs for every column that has a value. Group headers are identified
    when all cells in a row are identical to the first cell text and the
    row has no data variation; those rows are consumed as context, not emitted.

    This is tuned to the Avino table structure; it's a pragmatic step toward
    scalable table chunking (constraint 3 in the design discussion).
    """
    grid = getattr(getattr(item, "data", None), "grid", None)
    if not grid:
        return []
    if not grid or len(grid) == 0:
        return []

    caption = item.caption_text(doc).strip()

    def clean_header(h: str) -> str:
        if not h:
            return ""
        return h.replace("*", "").strip()

    header_row = grid[0]
    periods = [clean_header(getattr(c, "text", "")) for c in header_row[1:]]

    nodes: list[str] = []
    group = ""
    for row in grid[1:]:
        if len(row) == 0:
            continue
        label = getattr(row[0], "text", "").strip()
        if not label:
            continue
        vals = [getattr(c, "text", "").strip() for c in row[1:]]

        all_same = len(vals) > 0 and all(v == label for v in vals)
        if all_same:
            group = label
            continue

        if group:
            label_lower = label.split()[0].lower() if label else ""
            group_lower = group.lower() if group else ""
            if group_lower and label_lower and label_lower in group_lower:
                title = label
            else:
                title = f"{group} — {label}"
        else:
            title = label

        parts = []
        if caption:
            parts.append(f"{caption}.")
        parts.append(f"{title}.")
        # One period per line: unambiguous to parse and easier for the LLM to
        # match a question's period against a single value.
        for period, value in zip(periods, vals):
            if value:
                parts.append(f"{period} = {value}")
        nodes.append("\n".join(parts))
    return nodes


def table_consumed_refs(doc: DoclingDocument) -> set[str]:
    """Refs already inlined into a table node, so the walk does not repeat them."""
    consumed: set[str] = set()
    for table in doc.tables:
        for ref in list(table.captions) + list(table.footnotes):
            consumed.add(ref.cref)
    return consumed


def split_with_word_overlap(text: str, chunk_words: int, overlap: int) -> list[str]:
    """Slide a word window across text so each chunk repeats the prior chunk's tail."""
    words = text.split()
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if not 0 <= overlap < chunk_words:
        raise ValueError("overlap must be >= 0 and smaller than chunk_words")
    if len(words) <= chunk_words:
        return [" ".join(words)] if words else []

    step = chunk_words - overlap
    chunks = []
    start = 0
    while start < len(words):
        chunk = words[start : start + chunk_words]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks


def collect_blocks(doc: DoclingDocument) -> list[dict]:
    """One reading-order pass -> ordered text and table blocks.

    Captions and footnotes attached to a table are skipped here because
    `table_to_text` inlines them. Footnotes that belong to no table are kept as
    prose so their content is not lost.
    """
    blocks: list[dict] = []
    consumed = table_consumed_refs(doc)
    section = ""
    buffer: list[str] = []
    buffer_page = 1

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = "\n\n".join(buffer).strip()
        buffer = []
        if text:
            blocks.append(
                {"text": text, "page": buffer_page, "section": section, "type": "text"}
            )

    for item, _level in doc.iterate_items():
        label = item.label
        if label in SECTION_LABELS:
            flush()
            section = getattr(item, "text", "").strip()
            continue
        if label == DocItemLabel.TABLE:
            flush()
            table_text = table_to_text(item, doc)
            if table_text:
                blocks.append(
                    {
                        "text": table_text,
                        "page": page_of(item),
                        "section": section,
                        "type": "table",
                    }
                )
            for row_text in table_rows_to_texts(item, doc):
                if row_text:
                    blocks.append(
                        {
                            "text": row_text,
                            "page": page_of(item),
                            "section": section,
                            "type": "table_row",
                        }
                    )
            continue
        if label == DocItemLabel.CAPTION and item.self_ref in consumed:
            continue
        if label == DocItemLabel.FOOTNOTE and item.self_ref in consumed:
            continue
        if label in TEXT_LABELS or label == DocItemLabel.FOOTNOTE:
            text = getattr(item, "text", "").strip()
            if not text:
                continue
            if not buffer:
                buffer_page = page_of(item)
            buffer.append(text)

    flush()
    return blocks


def build_nodes(
    pdf_path: str,
    *,
    chunk_words: int = CHUNK_WORDS,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[TextNode]:
    """Parse a PDF into overlapping text chunks, whole-table nodes, and table row nodes."""
    source = Path(pdf_path).name
    doc = load_document(pdf_path)

    nodes: list[TextNode] = []
    ordinals: dict[tuple[str, str], int] = {}

    for block in collect_blocks(doc):
        key = (block["section"], block["type"])
        pieces = (
            [block["text"]]
            if block["type"].startswith("table")
            else split_with_word_overlap(block["text"], chunk_words, chunk_overlap)
        )
        for piece in pieces:
            ordinal = ordinals.get(key, 0)
            ordinals[key] = ordinal + 1
            scope = re.sub(r"[^0-9A-Za-z]+", "_", block["section"]).strip("_") or "root"
            chunk_id = (
                f"{source}::p{block['page']}::{scope}::{block['type']}::{ordinal}"
            )
            # Row nodes already carry their own caption and group context, so
            # prepending the section heading would only add noise.
            body = (
                piece
                if block["type"].startswith("table")
                else f"{block['section']}\n\n{piece}"
                if block["section"]
                else piece
            )
            nodes.append(
                TextNode(
                    text=body,
                    metadata={
                        "source": source,
                        "page": block["page"],
                        "type": block["type"],
                        "section": block["section"],
                        "chunk_id": chunk_id,
                    },
                )
            )

    return nodes
