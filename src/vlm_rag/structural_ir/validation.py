"""Structural-to-physical integrity and exact content-partition validation."""

import hashlib
import re
import unicodedata
from collections import defaultdict

from vlm_rag.physical_ir.models import BlockDisposition
from vlm_rag.physical_ir.serialization_v1 import physical_document_v1_to_json
from vlm_rag.physical_ir.v1 import PhysicalBlockV1, PhysicalDocumentV1
from vlm_rag.structural_ir.models import AnchorRole, PhysicalAnchor, StructuralDocument


class StructuralPhysicalIntegrityError(ValueError):
    """Raised when Structural IR is inconsistent with its bound Physical IR input."""


def _all_anchors(document: StructuralDocument) -> list[tuple[str, PhysicalAnchor]]:
    return [
        (node.id, anchor)
        for node in document.nodes
        for anchor in (*node.heading_anchors, *node.direct_content_anchors)
    ]


def validate_against_physical(structural: StructuralDocument, physical: PhysicalDocumentV1) -> None:
    """Validate identity, physical SHA, anchor bounds, ownership, and exact coverage."""
    identity_pairs = (
        ("document_id", structural.document_id, physical.document_id),
        ("version_id", structural.version_id, physical.version_id),
        (
            "source_artifact_sha256",
            structural.source_artifact_sha256,
            physical.source_artifact_sha256,
        ),
    )
    for field, actual, expected in identity_pairs:
        if actual != expected:
            raise StructuralPhysicalIntegrityError(
                f"{field} mismatch: structural={actual!r}, physical={expected!r}"
            )
    physical_sha = hashlib.sha256(
        physical_document_v1_to_json(physical).encode("utf-8")
    ).hexdigest()
    if structural.source_physical_ir_sha256 != physical_sha:
        raise StructuralPhysicalIntegrityError("source Physical IR SHA-256 mismatch")

    blocks: dict[str, PhysicalBlockV1] = {
        block.id: block for page in physical.pages for block in page.blocks
    }
    anchors_by_block: dict[str, list[tuple[str, PhysicalAnchor]]] = defaultdict(list)
    for node_id, anchor in _all_anchors(structural):
        block = blocks.get(anchor.block_id)
        if block is None:
            raise StructuralPhysicalIntegrityError(
                f"anchor from {node_id!r} references missing block {anchor.block_id!r}"
            )
        if anchor.page_index != block.page_index:
            raise StructuralPhysicalIntegrityError(
                f"anchor page disagrees with block {anchor.block_id!r}"
            )
        if block.disposition != BlockDisposition.CONTENT:
            raise StructuralPhysicalIntegrityError(
                f"non-CONTENT block {anchor.block_id!r} must remain unanchored"
            )
        if anchor.role == AnchorRole.OBJECT:
            if block.text:
                raise StructuralPhysicalIntegrityError(
                    f"OBJECT anchor requires an empty-text block: {anchor.block_id!r}"
                )
        else:
            if not block.text:
                raise StructuralPhysicalIntegrityError(
                    f"text anchor references empty block {anchor.block_id!r}"
                )
            assert anchor.char_start is not None and anchor.char_end is not None
            if anchor.char_end > len(block.text):
                raise StructuralPhysicalIntegrityError(
                    f"anchor exceeds text bounds for block {anchor.block_id!r}"
                )
        anchors_by_block[anchor.block_id].append((node_id, anchor))

    for block in blocks.values():
        attached = anchors_by_block.get(block.id, [])
        if block.disposition != BlockDisposition.CONTENT:
            if attached:
                raise StructuralPhysicalIntegrityError(
                    f"discarded/unknown block {block.id!r} unexpectedly has anchors"
                )
            continue
        if not block.text:
            object_anchors = [anchor for _, anchor in attached if anchor.role == AnchorRole.OBJECT]
            if len(object_anchors) != 1 or len(attached) != 1:
                raise StructuralPhysicalIntegrityError(
                    f"empty CONTENT block {block.id!r} requires exactly one OBJECT owner"
                )
            continue
        text_anchors = [
            anchor
            for _, anchor in attached
            if anchor.role in {AnchorRole.MARKER, AnchorRole.TITLE, AnchorRole.BODY}
        ]
        spans = sorted((anchor.char_start, anchor.char_end) for anchor in text_anchors)
        cursor = 0
        for start, end in spans:
            assert start is not None and end is not None
            if start != cursor:
                problem = "overlap" if start < cursor else "gap"
                raise StructuralPhysicalIntegrityError(
                    f"text anchors have a {problem} in block {block.id!r} at {cursor}"
                )
            cursor = end
        if cursor != len(block.text):
            raise StructuralPhysicalIntegrityError(
                f"text anchors do not cover block {block.id!r} through {len(block.text)}"
            )

    for node in structural.nodes[1:]:
        marker_anchors = [
            anchor for anchor in node.heading_anchors if anchor.role == AnchorRole.MARKER
        ]
        if len(marker_anchors) != 1:
            raise StructuralPhysicalIntegrityError(
                f"node {node.id!r} requires exactly one MARKER anchor"
            )
        marker = marker_anchors[0]
        block = blocks[marker.block_id]
        assert marker.char_start is not None and marker.char_end is not None
        observed = block.text[marker.char_start : marker.char_end].strip()
        if observed != node.marker_text:
            raise StructuralPhysicalIntegrityError(
                f"node {node.id!r} marker_text is not backed by its marker anchor"
            )
        if node.recognition_evidence.matched_text != node.marker_text:
            raise StructuralPhysicalIntegrityError(
                f"node {node.id!r} recognition evidence disagrees with marker_text"
            )
        title_anchors = [
            anchor for anchor in node.heading_anchors if anchor.role == AnchorRole.TITLE
        ]
        if (node.title is None) != (not title_anchors):
            raise StructuralPhysicalIntegrityError(
                f"node {node.id!r} title and TITLE anchors must be present together"
            )
        if title_anchors:
            title_text = "".join(
                blocks[anchor.block_id].text[anchor.char_start : anchor.char_end]
                for anchor in title_anchors
                if anchor.char_start is not None and anchor.char_end is not None
            )
            normalized_title = re.sub(
                r"[\t\f\v ]+",
                " ",
                unicodedata.normalize("NFKC", title_text).replace("\u00a0", " "),
            ).strip()
            if normalized_title != node.title:
                raise StructuralPhysicalIntegrityError(
                    f"node {node.id!r} title is not backed by its TITLE anchors"
                )


def reconstruction_by_block(
    structural: StructuralDocument, physical: PhysicalDocumentV1
) -> dict[str, str]:
    """Reconstruct each anchored text block from its ordered Structural IR spans."""
    block_text = {block.id: block.text for page in physical.pages for block in page.blocks}
    anchors: dict[str, list[PhysicalAnchor]] = defaultdict(list)
    for _, anchor in _all_anchors(structural):
        if anchor.role != AnchorRole.OBJECT:
            anchors[anchor.block_id].append(anchor)
    result: dict[str, str] = {}
    for block_id, values in anchors.items():
        values.sort(key=lambda anchor: (anchor.char_start or 0, anchor.char_end or 0))
        result[block_id] = "".join(
            block_text[block_id][anchor.char_start : anchor.char_end]
            for anchor in values
            if anchor.char_start is not None and anchor.char_end is not None
        )
    return result


__all__ = [
    "StructuralPhysicalIntegrityError",
    "reconstruction_by_block",
    "validate_against_physical",
]
