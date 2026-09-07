"""Parser-neutral line events retaining exact Physical IR character offsets."""

from dataclasses import dataclass

from vlm_rag.physical_ir.v1 import BlockKindV1, PhysicalBlockV1, PhysicalDocumentV1


@dataclass(frozen=True, slots=True)
class PhysicalLineEvent:
    """One exact LF-delimited segment from a Physical IR text block."""

    block_id: str
    page_index: int
    block_kind: BlockKindV1
    document_order: int
    line_order: int
    char_start: int
    char_end: int
    text: str


def iter_block_lines(block: PhysicalBlockV1, document_order: int) -> tuple[PhysicalLineEvent, ...]:
    """Split on LF without changing or dropping any source character."""
    if not block.text:
        return ()
    events: list[PhysicalLineEvent] = []
    start = 0
    line_order = 0
    while start < len(block.text):
        newline = block.text.find("\n", start)
        end = len(block.text) if newline < 0 else newline + 1
        events.append(
            PhysicalLineEvent(
                block_id=block.id,
                page_index=block.page_index,
                block_kind=block.kind,
                document_order=document_order,
                line_order=line_order,
                char_start=start,
                char_end=end,
                text=block.text[start:end],
            )
        )
        start = end
        line_order += 1
    return tuple(events)


def physical_text_events(document: PhysicalDocumentV1) -> tuple[PhysicalLineEvent, ...]:
    """Return all non-empty CONTENT text events in physical document order."""
    from vlm_rag.physical_ir.models import BlockDisposition

    events: list[PhysicalLineEvent] = []
    document_order = 0
    for page in document.pages:
        for block in page.blocks:
            if block.disposition == BlockDisposition.CONTENT and block.text:
                events.extend(iter_block_lines(block, document_order))
            document_order += 1
    return tuple(events)


__all__ = ["PhysicalLineEvent", "iter_block_lines", "physical_text_events"]
