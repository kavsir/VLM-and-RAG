"""Deterministic conversion of observed table HTML into the v1 logical grid."""

import re
from html.parser import HTMLParser

from pydantic import ValidationError

from vlm_rag.physical_ir.v1 import TableCell, TableStructure


class TableHTMLStructureError(ValueError):
    """Raised when observed HTML does not describe a defensible logical table."""


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list[TableCell] = []
        self.occupied: set[tuple[int, int]] = set()
        self.row_index = -1
        self.in_row = False
        self.cell_start: tuple[int, int, int, int, bool] | None = None
        self.cell_text: list[str] = []
        self.saw_table = False
        self.table_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        if name == "table":
            if self.saw_table and self.table_depth == 0:
                raise TableHTMLStructureError("multiple table elements are ambiguous")
            self.table_depth += 1
            if self.table_depth > 1:
                raise TableHTMLStructureError("nested tables are not supported")
            self.saw_table = True
            return
        if self.table_depth != 1:
            return
        if name == "tr":
            if self.in_row or self.cell_start is not None:
                raise TableHTMLStructureError("nested or unclosed table row")
            self.row_index += 1
            self.in_row = True
        elif name in {"td", "th"}:
            if not self.in_row or self.cell_start is not None:
                raise TableHTMLStructureError("table cell must occur directly within a row")
            attributes = {key.lower(): value for key, value in attrs}
            row_span = self._positive_span(attributes.get("rowspan"), "rowspan")
            column_span = self._positive_span(attributes.get("colspan"), "colspan")
            column = 0
            while any(
                (row, occupied_column) in self.occupied
                for row in range(self.row_index, self.row_index + row_span)
                for occupied_column in range(column, column + column_span)
            ):
                column += 1
            self.cell_start = (self.row_index, column, row_span, column_span, name == "th")
            self.cell_text = []
        elif name == "br" and self.cell_start is not None:
            self.cell_text.append("\n")

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name == "table":
            if self.table_depth != 1 or self.in_row or self.cell_start is not None:
                raise TableHTMLStructureError("table ended with an open row or cell")
            self.table_depth = 0
            return
        if self.table_depth != 1:
            return
        if name in {"td", "th"}:
            if self.cell_start is None:
                raise TableHTMLStructureError("table cell end has no matching start")
            row, column, row_span, column_span, is_header = self.cell_start
            for occupied_row in range(row, row + row_span):
                for occupied_column in range(column, column + column_span):
                    position = (occupied_row, occupied_column)
                    if position in self.occupied:
                        raise TableHTMLStructureError(
                            f"HTML table cells overlap at logical position {position}"
                        )
                    self.occupied.add(position)
            text = re.sub(r"\s+", " ", "".join(self.cell_text)).strip()
            self.cells.append(
                TableCell(
                    row_start=row,
                    column_start=column,
                    row_span=row_span,
                    column_span=column_span,
                    text=text,
                    is_header=is_header,
                )
            )
            self.cell_start = None
            self.cell_text = []
        elif name == "tr":
            if not self.in_row or self.cell_start is not None:
                raise TableHTMLStructureError("table row ended in an invalid state")
            self.in_row = False

    def handle_data(self, data: str) -> None:
        if self.cell_start is not None:
            self.cell_text.append(data)

    @staticmethod
    def _positive_span(value: str | None, field: str) -> int:
        if value is None:
            return 1
        try:
            parsed = int(value)
        except ValueError as exc:
            raise TableHTMLStructureError(f"invalid {field} value {value!r}") from exc
        if parsed < 1:
            raise TableHTMLStructureError(f"{field} must be positive")
        return parsed


def table_structure_from_html(html: str) -> TableStructure | None:
    """Return a logical table grid, or ``None`` when no table element is present."""
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
        parser.close()
    except (TableHTMLStructureError, ValidationError) as exc:
        raise TableHTMLStructureError(f"cannot recover table structure: {exc}") from exc
    if not parser.saw_table:
        return None
    if parser.table_depth or parser.in_row or parser.cell_start is not None:
        raise TableHTMLStructureError("unterminated table HTML")
    row_count = max(
        [parser.row_index + 1, *(cell.row_start + cell.row_span for cell in parser.cells)]
    )
    column_count = max([0, *(cell.column_start + cell.column_span for cell in parser.cells)])
    return TableStructure(row_count=row_count, column_count=column_count, cells=tuple(parser.cells))


__all__ = ["TableHTMLStructureError", "table_structure_from_html"]
