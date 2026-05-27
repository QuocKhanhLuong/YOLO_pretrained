#!/usr/bin/env python3
"""Streaming parser for PostgreSQL plain SQL dump COPY blocks."""

from __future__ import annotations

import csv
import gzip
import io
import re
import argparse
import sys
from collections.abc import Iterator
from pathlib import Path


COPY_RE = re.compile(
    r"^COPY\s+([^\s(]+)\s*\((?P<columns>.*?)\)\s+FROM\s+stdin;",
    re.IGNORECASE,
)


def normalize_table_name(table_name: str) -> str:
    """Normalize a dump table name while preserving the optional schema prefix."""
    return table_name.strip().replace('"', "")


def table_matches(table_name: str, target_names: set[str]) -> bool:
    """Return true when the normalized full or unqualified name is targeted."""
    normalized = normalize_table_name(table_name)
    unqualified = normalized.rsplit(".", 1)[-1]
    return normalized in target_names or unqualified in target_names


def parse_columns(columns_text: str) -> list[str]:
    """Parse the column list from a COPY header."""
    reader = csv.reader(
        io.StringIO(columns_text),
        delimiter=",",
        quotechar='"',
        skipinitialspace=True,
    )
    return [column.strip().strip('"') for column in next(reader)]


def parse_copy_fields(line: str) -> list[str | None]:
    """Split one COPY data line into decoded fields."""
    line = line.rstrip("\n")
    if line.endswith("\r"):
        line = line[:-1]

    reader = csv.reader(
        io.StringIO(line),
        delimiter="\t",
        quotechar=None,
        escapechar=None,
        quoting=csv.QUOTE_NONE,
    )
    return [decode_copy_value(value) for value in next(reader)]


def decode_copy_value(value: str) -> str | None:
    """Decode PostgreSQL text COPY escapes, keeping non-null values as strings."""
    if value == r"\N":
        return None
    if "\\" not in value:
        return value

    decoded: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char != "\\" or index == len(value) - 1:
            decoded.append(char)
            index += 1
            continue

        index += 1
        escaped = value[index]
        if escaped == "n":
            decoded.append("\n")
        elif escaped == "r":
            decoded.append("\r")
        elif escaped == "t":
            decoded.append("\t")
        elif escaped == "b":
            decoded.append("\b")
        elif escaped == "f":
            decoded.append("\f")
        elif escaped == "v":
            decoded.append("\v")
        elif escaped in {"\\", "'"}:
            decoded.append(escaped)
        elif escaped == "x":
            hex_digits = []
            lookahead = index + 1
            while lookahead < len(value) and len(hex_digits) < 2:
                if value[lookahead] not in "0123456789abcdefABCDEF":
                    break
                hex_digits.append(value[lookahead])
                lookahead += 1
            if hex_digits:
                decoded.append(chr(int("".join(hex_digits), 16)))
                index = lookahead - 1
            else:
                decoded.append("x")
        elif escaped in "01234567":
            octal_digits = [escaped]
            lookahead = index + 1
            while lookahead < len(value) and len(octal_digits) < 3:
                if value[lookahead] not in "01234567":
                    break
                octal_digits.append(value[lookahead])
                lookahead += 1
            decoded.append(chr(int("".join(octal_digits), 8)))
            index = lookahead - 1
        else:
            decoded.append(escaped)
        index += 1
    return "".join(decoded)


def iter_copy_rows(
    sql_gz_path: Path,
    table_names: set[str],
) -> Iterator[tuple[str, dict[str, str | None]]]:
    """Yield rows from targeted COPY blocks in a PostgreSQL .sql.gz dump."""
    targets = {normalize_table_name(table_name) for table_name in table_names}

    with gzip.open(sql_gz_path, "rt", encoding="utf-8", errors="replace", newline="") as file_obj:
        current_table: str | None = None
        current_columns: list[str] = []
        selected = False

        for line_number, line in enumerate(file_obj, start=1):
            if current_table is None:
                match = COPY_RE.match(line)
                if not match:
                    continue

                current_table = normalize_table_name(match.group(1))
                current_columns = parse_columns(match.group("columns"))
                selected = table_matches(current_table, targets)
                continue

            if line == "\\.\n" or line == "\\.\r\n" or line.rstrip("\r\n") == r"\.":
                current_table = None
                current_columns = []
                selected = False
                continue

            if not selected:
                continue

            fields = parse_copy_fields(line)
            if len(fields) != len(current_columns):
                raise ValueError(
                    f"{sql_gz_path}:{line_number}: COPY row for {current_table} has "
                    f"{len(fields)} fields, expected {len(current_columns)}"
                )
            yield current_table, dict(zip(current_columns, fields))


def load_tables(
    sql_gz_path: Path,
    table_names: set[str],
) -> dict[str, list[dict[str, str | None]]]:
    """Load selected COPY tables from a PostgreSQL .sql.gz dump into memory."""
    tables: dict[str, list[dict[str, str | None]]] = {}
    for table_name, row in iter_copy_rows(sql_gz_path, table_names):
        tables.setdefault(table_name, []).append(row)
    return tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect row counts for selected COPY tables in a PostgreSQL .sql.gz dump."
    )
    parser.add_argument("sql_gz_path", nargs="?", help="Path to a plain SQL .gz dump.")
    parser.add_argument(
        "--table",
        action="append",
        dest="tables",
        default=[],
        help="Table to count. Can be repeated, for example public.frames.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.sql_gz_path:
        return 0
    if not args.tables:
        print("ERROR: provide at least one --table", file=sys.stderr)
        return 1

    counts: dict[str, int] = {}
    try:
        for table_name, _row in iter_copy_rows(Path(args.sql_gz_path), set(args.tables)):
            counts[table_name] = counts.get(table_name, 0) + 1
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for table_name in sorted(counts):
        print(f"{table_name}: {counts[table_name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
