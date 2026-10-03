"""Tests for markdown_nodes: word-overlap chunking plus ground-truth checks
against the Avino metallurgical table.

The table's values are OCR-derived, so numeric corruption is a real risk. The
`test_avino_table_*` tests assert every cell against questions.json, which holds
independently-verified values for all six periods.
"""

import json
import re
from pathlib import Path

import pytest

from markdown_nodes import CHUNK_WORDS, build_nodes, split_with_word_overlap


def _repo_root():
    """Locate the repo root by walking up until pdfs/ and questions.json exist."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pdfs").is_dir() and (candidate / "questions.json").is_file():
            return candidate
    pytest.skip("could not locate repo root (expected pdfs/ and questions.json)")


REPO_ROOT = _repo_root()
AVINO_PDF = REPO_ROOT / "pdfs" / "avino.pdf"
QUESTIONS = REPO_ROOT / "questions.json"

# Maps the table's (group row, row label) pair onto a questions.json field.
ROW_TO_FIELD = {
    ("Feed Tonnage", "Tonnes Milled (dry t)"): ("tonnes_milled",),
    ("Feed Grade", "Silver (g/t)"): ("grade", "silver_gt"),
    ("Feed Grade", "Gold (g/t)"): ("grade", "gold_gt"),
    ("Feed Grade", "Copper (%)"): ("grade", "copper_percent"),
    ("Recovery", "Silver (%)"): ("recovery_percent", "silver"),
    ("Recovery", "Gold (%)"): ("recovery_percent", "gold"),
    ("Recovery", "Copper (%)"): ("recovery_percent", "copper"),
    ("Total Metal Produced", "Silver Produced (oz)"): ("production", "silver_oz"),
    ("Total Metal Produced", "Gold Produced (oz)"): ("production", "gold_oz"),
    ("Total Metal Produced", "Copper Produced (lbs)"): ("production", "copper_lbs"),
}

FOOTNOTE_MARKER = re.compile(r"[*+º]+\s*$")


# --------------------------------------------------------------------------
# split_with_word_overlap
# --------------------------------------------------------------------------


def _words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_passes_through_unsplit():
    assert split_with_word_overlap("one two three", 300, 50) == ["one two three"]


def test_text_shorter_than_window_yields_single_chunk():
    chunks = split_with_word_overlap(_words(299), 300, 50)
    assert len(chunks) == 1
    assert len(chunks[0].split()) == 299


def test_text_exactly_at_window_yields_single_chunk():
    assert len(split_with_word_overlap(_words(300), 300, 50)) == 1


def test_text_one_word_over_window_yields_two_chunks():
    chunks = split_with_word_overlap(_words(301), 300, 50)
    assert [len(c.split()) for c in chunks] == [300, 51]


def test_no_word_is_lost_across_chunks():
    text = _words(1000)
    chunks = split_with_word_overlap(text, 300, 50)
    # Overlap repeats words by design; walk each chunk from where the last ended.
    cursor = 0
    for chunk in chunks:
        chunk_words = chunk.split()
        assert chunk_words[:1] == text.split()[cursor : cursor + 1]
        cursor += len(chunk_words) - (50 if chunk is not chunks[-1] else 0)
    assert cursor == 1000


def test_consecutive_chunks_overlap_by_expected_word_count():
    chunks = split_with_word_overlap(_words(1000), 300, 50)
    first_tail = chunks[0].split()[-50:]
    second_head = chunks[1].split()[:50]
    assert first_tail == second_head


def test_zero_overlap_produces_disjoint_chunks():
    chunks = split_with_word_overlap(_words(1000), 300, 0)
    assert [len(c.split()) for c in chunks] == [300, 300, 300, 100]
    assert chunks[0].split()[-1] != chunks[1].split()[0]


def test_chunk_order_is_preserved():
    text = _words(1000)
    chunks = split_with_word_overlap(text, 300, 50)
    assert chunks[0].split() == text.split()[:300]


def test_whitespace_only_text_yields_no_chunks():
    assert split_with_word_overlap("   \n\t ", 300, 50) == []


@pytest.mark.parametrize("chunk_words", [0, -1])
def test_non_positive_chunk_words_raises(chunk_words):
    with pytest.raises(ValueError):
        split_with_word_overlap("some words here", chunk_words, 0)


@pytest.mark.parametrize("overlap", [-1, 300, 301])
def test_overlap_must_be_smaller_than_chunk_words(overlap):
    with pytest.raises(ValueError):
        split_with_word_overlap(_words(1000), 300, overlap)


# --------------------------------------------------------------------------
# Avino table ground truth
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def avino_nodes():
    if not AVINO_PDF.exists():
        pytest.skip(f"missing PDF: {AVINO_PDF}")
    return build_nodes(str(AVINO_PDF))


@pytest.fixture(scope="module")
def avino_table_node(avino_nodes):
    tables = [n for n in avino_nodes if n.metadata["type"] == "table"]
    assert tables, "expected at least one table node from avino.pdf"
    return tables[0]


@pytest.fixture(scope="module")
def avino_periods():
    if not QUESTIONS.exists():
        pytest.skip(f"missing ground truth: {QUESTIONS}")
    data = json.loads(QUESTIONS.read_text())
    prop = next(p for p in data["properties"] if p["name"] == "Avino")
    return {p["period"]: p for p in prop["performance_data"]}


def _parse_markdown_table(text):
    """Return (header_cells, data_rows) from the pipe-table in a node's text."""
    lines = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    rows = [
        [c.strip() for c in ln.strip().strip("|").split("|")]
        for ln in lines
    ]
    # Drop the |---|---| separator row.
    rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r if c)]
    return rows[0], rows[1:]


def _to_float(cell):
    return float(cell.replace(",", "").strip())


def test_table_period_columns_match_ground_truth(avino_table_node, avino_periods):
    header, _ = _parse_markdown_table(avino_table_node.get_content())
    columns = [FOOTNOTE_MARKER.sub("", c) for c in header[1:]]
    assert set(columns) == set(avino_periods)


def test_gold_recovery_row_label_is_not_ocr_corrupted(avino_table_node):
    _, rows = _parse_markdown_table(avino_table_node.get_content())
    labels = [r[0] for r in rows]
    assert "Gold (%)" in labels
    assert "G0%) (%)" not in labels


# --------------------------------------------------------------------------
# Node structure
# --------------------------------------------------------------------------


def test_every_node_carries_required_metadata(avino_nodes):
    required = {"source", "page", "type", "section", "chunk_id"}
    for node in avino_nodes:
        assert required <= set(node.metadata)
        # ChromaDB rejects None metadata values.
        assert all(value is not None for value in node.metadata.values())


def test_chunk_ids_are_unique(avino_nodes):
    ids = [n.metadata["chunk_id"] for n in avino_nodes]
    assert len(ids) == len(set(ids))


def test_page_numbers_are_one_based(avino_nodes):
    for node in avino_nodes:
        assert isinstance(node.metadata["page"], int)
        assert node.metadata["page"] >= 1


def test_text_chunks_respect_configured_word_limit(avino_nodes):
    limit = CHUNK_WORDS + 8  # + section header prepended to each chunk
    for node in avino_nodes:
        if node.metadata["type"] == "text":
            assert len(node.get_content().split()) <= limit


def test_text_chunks_carry_their_section_header(avino_nodes):
    with_section = [n for n in avino_nodes if n.metadata["type"] == "text"]
    assert with_section
    assert all(
        n.get_content().startswith(n.metadata["section"])
        for n in with_section
        if n.metadata["section"]
    )


def test_table_node_preserves_every_period_column(avino_table_node, avino_periods):
    header, _ = _parse_markdown_table(avino_table_node.get_content())
    assert len(header) - 1 == len(avino_periods)


def test_every_table_value_matches_ground_truth(avino_table_node, avino_periods):
    header, rows = _parse_markdown_table(avino_table_node.get_content())
    columns = [FOOTNOTE_MARKER.sub("", c) for c in header[1:]]

    group = None
    checked = 0
    for row in rows:
        label, values = row[0], row[1:]
        if values and all(v == label for v in values):
            group = label
            continue

        key = (group, label)
        if key not in ROW_TO_FIELD:
            continue

        path = ROW_TO_FIELD[key]
        for period, cell in zip(columns, values):
            expected = avino_periods[period]
            for part in path:
                expected = expected[part]
            assert _to_float(cell) == pytest.approx(float(expected)), (
                f"{period} / {'.'.join(path)}: parsed {cell!r}, expected {expected!r}"
            )
            checked += 1

    assert checked == 60, f"expected 60 checked values, got {checked}"


# --------------------------------------------------------------------------
# Table row nodes
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def avino_row_nodes(avino_nodes):
    rows = [n for n in avino_nodes if n.metadata["type"] == "table_row"]
    assert rows, "expected table_row nodes from avino.pdf"
    return rows


def _parse_row_node(text):
    """Return (caption, title, {period: value}) from a table_row node's text.

    Shape: "<caption>." / "<title>." / one "<period> = <value>" line per column.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    assert len(lines) >= 2, f"row node too short: {text[:120]!r}"

    caption = lines[0].removesuffix(".").strip()
    title = lines[1].removesuffix(".").strip()

    values = {}
    for line in lines[2:]:
        assert " = " in line, f"expected '<period> = <value>', got {line!r}"
        period, value = line.split(" = ", 1)
        values[period.strip()] = value.strip()
    return caption, title, values


def test_table_and_row_nodes_both_exist(avino_nodes, avino_row_nodes):
    assert any(n.metadata["type"] == "table" for n in avino_nodes)
    assert avino_row_nodes


def test_row_node_count_matches_data_rows(avino_row_nodes):
    # 15 grid rows minus 1 header minus 4 group headers.
    assert len(avino_row_nodes) == 10


def test_row_nodes_omit_the_section_heading_prefix(avino_row_nodes):
    # Row nodes carry their own caption/title context; prepending the section
    # heading would add noise without adding meaning.
    for node in avino_row_nodes:
        section = node.metadata["section"]
        if section:
            assert not node.get_content().startswith(section)


def test_group_headers_are_consumed_not_emitted(avino_row_nodes):
    titles = [_parse_row_node(n.get_content())[1] for n in avino_row_nodes]
    for group in ("Feed Tonnage", "Feed Grade", "Recovery", "Total Metal Produced"):
        assert group not in titles, f"group header leaked as its own node: {group!r}"


def test_row_titles_join_group_and_label_without_duplication(avino_row_nodes):
    titles = {_parse_row_node(n.get_content())[1] for n in avino_row_nodes}
    assert "Feed Grade — Silver (g/t)" in titles
    assert "Recovery — Silver (%)" in titles
    assert "Feed Tonnage — Tonnes Milled (dry t)" in titles
    # "Feed Tonnage of Tonnes Milled" style duplication must not appear.
    assert not any(" of " in t for t in titles), titles


def test_row_nodes_cover_every_period_column(avino_row_nodes, avino_periods):
    for node in avino_row_nodes:
        _, _, values = _parse_row_node(node.get_content())
        assert set(values) == set(avino_periods), node.get_content()[:120]


def test_row_node_values_match_ground_truth(avino_row_nodes, avino_periods):
    """Every one of the 60 OCR-derived values must be right in the row nodes too."""
    checked = 0
    for node in avino_row_nodes:
        _, title, values = _parse_row_node(node.get_content())
        group = title.split(" — ")[0] if " — " in title else None
        label = title.split(" — ")[-1]
        key = (group, label)
        if key not in ROW_TO_FIELD:
            continue
        for period, cell in values.items():
            expected = avino_periods[period]
            for part in ROW_TO_FIELD[key]:
                expected = expected[part]
            assert _to_float(cell) == pytest.approx(float(expected)), (
                f"{period} / {title}: row node parsed {cell!r}, expected {expected!r}"
            )
            checked += 1

    assert checked == 60, f"expected 60 checked values in row nodes, got {checked}"


def test_row_node_feed_grade_silver_2025_is_59(avino_row_nodes):
    """Guards the question that exposed the whole-table node's poor ranking."""
    matches = [
        n for n in avino_row_nodes
        if _parse_row_node(n.get_content())[1] == "Feed Grade — Silver (g/t)"
    ]
    assert len(matches) == 1
    assert _parse_row_node(matches[0].get_content())[2]["2025"] == "59"


def test_row_node_values_agree_with_whole_table_node(avino_table_node, avino_row_nodes):
    """Row nodes must not become a second, divergent source of truth."""
    header, rows = _parse_markdown_table(avino_table_node.get_content())
    columns = [FOOTNOTE_MARKER.sub("", c) for c in header[1:]]

    table_values = {}
    group = None
    for row in rows:
        label, values = row[0], row[1:]
        if values and all(v == label for v in values):
            group = label
            continue
        for period, cell in zip(columns, values):
            table_values[(group, label, period)] = cell

    for node in avino_row_nodes:
        _, title, values = _parse_row_node(node.get_content())
        group = title.split(" — ")[0] if " — " in title else None
        label = title.split(" — ")[-1]
        for period, cell in values.items():
            key = (group, label, period)
            if key in table_values:
                assert _to_float(cell) == pytest.approx(_to_float(table_values[key])), (
                    f"{period} / {title}: row node {cell!r} != table {table_values[key]!r}"
                )