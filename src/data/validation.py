"""Upload validation, custom errors, and column-name normalization."""
from pathlib import Path

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


class DataLoadError(Exception):
    """Base class for all user-facing data loading errors."""

    code = "data_load_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class UnsupportedFileTypeError(DataLoadError):
    code = "unsupported_file_type"


class FileTooLargeError(DataLoadError):
    code = "file_too_large"


class EmptyFileError(DataLoadError):
    code = "empty_file"


class MalformedFileError(DataLoadError):
    code = "malformed_file"


class TooManyRowsError(DataLoadError):
    code = "too_many_rows"


class SheetSelectionRequired(DataLoadError):
    """Raised when an Excel workbook has several sheets and none was chosen."""

    code = "sheet_selection_required"

    def __init__(self, sheet_names: list[str]):
        super().__init__("This workbook has multiple sheets. Please choose one.")
        self.sheet_names = sheet_names


def validate_upload(filename: str, content: bytes, max_bytes: int) -> tuple[str, str]:
    """Validate name, extension, size, and basic content signature.

    Returns (safe_filename, extension).
    """
    safe_name = Path(filename or "").name.strip() or "uploaded_file"
    ext = Path(safe_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext or 'none'}'. Please upload a .csv or .xlsx file."
        )
    if len(content) == 0:
        raise EmptyFileError("The uploaded file is empty.")
    if len(content) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise FileTooLargeError(f"File is too large. The limit is {limit_mb:.0f} MB.")

    # Cheap content checks so a renamed file is not parsed blindly.
    if ext == ".xlsx" and not content.startswith(b"PK\x03\x04"):
        raise MalformedFileError("This file is not a valid .xlsx workbook.")
    if ext == ".csv" and b"\x00" in content[:8192]:
        raise MalformedFileError(
            "This file looks binary (or UTF-16 encoded), not a text CSV. "
            "Please re-save it as UTF-8 CSV."
        )

    return safe_name, ext


def normalize_columns(columns) -> tuple[list[str], dict[str, str]]:
    """Make column names safe without changing normal ones.

    - converts to string and trims/collapses whitespace
    - empty or pandas 'Unnamed: N' headers become 'column_N'
    - duplicates get a numeric suffix (_2, _3, ...)

    Returns (new_names, renames) where renames maps original -> new for
    every column that was changed.
    """
    new_names: list[str] = []
    renames: dict[str, str] = {}
    seen: dict[str, int] = {}

    for i, col in enumerate(columns):
        original = str(col)
        name = " ".join(original.split())
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{i + 1}"

        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1

        # Edge case: a generated name may collide with a later real one.
        while name in new_names:
            name = f"{name}_dup"

        new_names.append(name)
        if name != original:
            renames[original] = name

    return new_names, renames