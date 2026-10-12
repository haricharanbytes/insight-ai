import io

import pandas as pd
import pytest

from src.data.loader import load_dataset
from src.data.validation import (
    EmptyFileError,
    FileTooLargeError,
    MalformedFileError,
    SheetSelectionRequired,
    TooManyRowsError,
    UnsupportedFileTypeError,
    normalize_columns,
)

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 1000


def _load(name, content, **kw):
    return load_dataset(name, content, MAX_BYTES, kw.pop("max_rows", MAX_ROWS), **kw)


def _xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


# CSV

def test_load_basic_csv():
    r = _load("data.csv", b"a,b\n1,2\n3,4\n")
    assert r.df.shape == (2, 2)
    assert r.file_type == "csv"
    assert list(r.df.columns) == ["a", "b"]


def test_csv_semicolon_delimiter():
    r = _load("data.csv", b"a;b;c\n1;2;3\n4;5;6\n")
    assert r.df.shape == (2, 3)
    assert r.delimiter == ";"


def test_csv_latin1_encoding():
    content = "name,city\nJosé,São Paulo\n".encode("latin-1")
    r = _load("data.csv", content)
    assert r.df.iloc[0]["name"] == "José"
    assert any("not UTF-8" in w for w in r.warnings)


def test_csv_utf8_bom():
    content = b"\xef\xbb\xbfa,b\n1,2\n"
    r = _load("data.csv", content)
    assert list(r.df.columns) == ["a", "b"]  # no BOM in first column name


def test_csv_does_not_modify_values():
    r = _load("data.csv", b"a,b\n,0\n-5,\n")
    assert r.df["a"].isna().sum() == 1  # missing stays missing
    assert (r.df["b"] == 0).sum() == 1  # zero stays zero
    assert (r.df["a"] < 0).sum() == 1  # negative stays negative


# Validation

def test_unsupported_extension():
    with pytest.raises(UnsupportedFileTypeError):
        _load("data.txt", b"a,b\n1,2\n")


def test_old_xls_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        _load("data.xls", b"whatever")


def test_empty_file():
    with pytest.raises(EmptyFileError):
        _load("data.csv", b"")


def test_header_only_csv():
    with pytest.raises(EmptyFileError):
        _load("data.csv", b"a,b,c\n")


def test_file_too_large():
    with pytest.raises(FileTooLargeError):
        load_dataset("data.csv", b"a,b\n1,2\n", max_bytes=3, max_rows=MAX_ROWS)


def test_too_many_rows():
    content = b"a\n" + b"1\n" * 50
    with pytest.raises(TooManyRowsError):
        _load("data.csv", content, max_rows=10)


def test_binary_file_renamed_to_csv():
    with pytest.raises(MalformedFileError):
        _load("data.csv", b"\x00\x01\x02\x03binary")


def test_fake_xlsx():
    with pytest.raises(MalformedFileError):
        _load("data.xlsx", b"this is not a zip file")


def test_filename_path_is_stripped():
    r = _load("../../etc/data.csv", b"a,b\n1,2\n")
    assert r.filename == "data.csv"


# Excel

def test_excel_single_sheet_auto_selected():
    content = _xlsx_bytes({"Sheet1": pd.DataFrame({"a": [1, 2], "b": [3, 4]})})
    r = _load("book.xlsx", content)
    assert r.df.shape == (2, 2)
    assert r.sheet_name == "Sheet1"


def test_excel_multi_sheet_requires_selection():
    content = _xlsx_bytes({
        "Sales": pd.DataFrame({"x": [1]}),
        "Costs": pd.DataFrame({"y": [2]}),
    })
    with pytest.raises(SheetSelectionRequired) as exc:
        _load("book.xlsx", content)
    assert exc.value.sheet_names == ["Sales", "Costs"]


def test_excel_choose_sheet():
    content = _xlsx_bytes({
        "Sales": pd.DataFrame({"x": [1, 2]}),
        "Costs": pd.DataFrame({"y": [9]}),
    })
    r = _load("book.xlsx", content, sheet_name="Costs")
    assert list(r.df.columns) == ["y"]
    assert r.sheet_name == "Costs"


def test_excel_unknown_sheet():
    content = _xlsx_bytes({"Sales": pd.DataFrame({"x": [1]})})
    with pytest.raises(MalformedFileError):
        _load("book.xlsx", content, sheet_name="Nope")


# Column names

def test_normalize_columns_blank_duplicate_spaces():
    new, renames = normalize_columns(["  Order   Date ", "Sales", "Sales", "", "Unnamed: 4"])
    assert new == ["Order Date", "Sales", "Sales_2", "column_4", "column_5"]
    assert "Order Date" not in renames  # unchanged-after-strip handled via original key
    assert renames["Sales"] == "Sales_2" or "Sales_2" in renames.values()


def test_normal_columns_untouched():
    new, renames = normalize_columns(["Order Date", "Sub-Category", "Sales"])
    assert new == ["Order Date", "Sub-Category", "Sales"]
    assert renames == {}