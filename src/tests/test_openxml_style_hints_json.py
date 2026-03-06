"""
Unit tests for OpenXML style hints sourced from JSON tree + format ID caches.
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.alignment_service import AlignmentService


def test_json_only_style_hints():
    service = AlignmentService()
    json_tree = {
        "type": "paragraph",
        "content": [
            {
                "type": "text",
                "value": "contoh",
                "fontFamily": "Courier New",
                "styleName": "MyCustomStyle"
            }
        ]
    }
    hints = service._extract_openxml_style_hints(json_tree)

    assert "courier new" in hints["font_families"]
    assert "mycustomstyle" in hints["style_ids"]
    assert hints["is_code_font"] is True
    assert hints["is_code_like_openxml"] is True


def test_format_id_style_hints():
    service = AlignmentService()
    json_tree = {
        "dfp_id": 2001,
        "content": [
            {"type": "text", "dftx_id": 1001, "value": "kode"}
        ]
    }
    text_cache = {
        1001: {
            "dftx_font_ascii": "Times New Roman",
            "dftx_bold": 0,
            "dftx_italic": 0,
            "dftx_underline": "none",
        }
    }
    paragraph_cache = {
        2001: {
            "dfp_p_style_id": "STTSSegmenProgramContent",
        }
    }

    hints = service._extract_openxml_style_hints(
        json_tree,
        text_format_cache=text_cache,
        paragraph_format_cache=paragraph_cache,
    )

    assert "times new roman" in hints["font_families"]
    assert "sttssegmenprogramcontent" in hints["style_ids"]
    assert hints["is_code_font"] is False
    assert hints["is_code_style"] is True
    assert hints["is_code_like_openxml"] is True


def test_missing_format_ids_graceful_fallback():
    service = AlignmentService()
    json_tree = {
        "dfp_id": 999999,
        "content": [
            {"type": "text", "dftx_id": 888888, "value": "fallback"}
        ]
    }
    hints = service._extract_openxml_style_hints(
        json_tree,
        text_format_cache={},
        paragraph_format_cache={},
    )

    assert isinstance(hints["font_families"], list)
    assert isinstance(hints["style_ids"], list)
    assert hints["is_code_font"] is False
    assert hints["is_code_style"] is False


def test_build_openxml_units_without_db_session():
    service = AlignmentService()
    element = SimpleNamespace(
        delemen_id=1,
        delemen_sequence=1,
        delemen_type="paragraph",
        delemen_json_tree={
            "dfp_id": 2002,
            "content": [{"type": "text", "dftx_id": 1002, "value": "unit text"}],
        },
    )
    format_cache = {
        "text": {1002: {"dftx_font_ascii": "Courier New"}},
        "paragraph": {2002: {"dfp_p_style_id": "BodyText"}},
    }

    units, table_debug = service._build_openxml_units(
        [element],
        page_seq_range=None,
        db_session=None,
        format_cache=format_cache,
    )

    assert len(units) == 1
    assert table_debug == []
    assert units[0]["font_families"] == ["courier new"]
    assert "bodytext" in units[0]["style_ids"]
    assert units[0]["is_code_font"] is True
    assert units[0]["is_code_like_openxml"] is True


if __name__ == "__main__":
    test_json_only_style_hints()
    test_format_id_style_hints()
    test_missing_format_ids_graceful_fallback()
    test_build_openxml_units_without_db_session()
    print("All OpenXML style hint tests passed.")
