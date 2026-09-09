"""Deterministic Semantic IR construction from frozen Physical and Structural IR."""

import hashlib

from vlm_rag.physical_ir.serialization_v1 import physical_document_v1_to_json
from vlm_rag.physical_ir.v1 import PhysicalDocumentV1
from vlm_rag.semantic_ir.extraction import extract_mention_candidates
from vlm_rag.semantic_ir.models import (
    SemanticDocument,
    SemanticMention,
    SemanticProvenanceKind,
    SemanticStatement,
    TextSemanticAnchor,
    VisualObservation,
)
from vlm_rag.structural_ir import (
    AnchorRole,
    StructuralDocument,
    structural_document_to_json,
    validate_against_physical,
)


class SemanticSourceIntegrityError(ValueError):
    """Raised when Semantic construction inputs do not share exact identity and bytes."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_semantic_document(
    physical: PhysicalDocumentV1,
    structural: StructuralDocument,
    *,
    visual_observations: tuple[VisualObservation, ...] = (),
) -> SemanticDocument:
    """Build statements and mentions without modifying either input IR."""
    validate_against_physical(structural, physical)
    physical_sha = _sha256_text(physical_document_v1_to_json(physical))
    structural_sha = _sha256_text(structural_document_to_json(structural))
    blocks = {block.id: block for page in physical.pages for block in page.blocks}
    statements: list[SemanticStatement] = []
    mentions: list[SemanticMention] = []
    statement_index = 0
    mention_index = 0
    for node in structural.nodes:
        for source_anchor in node.direct_content_anchors:
            if source_anchor.role != AnchorRole.BODY:
                continue
            block = blocks[source_anchor.block_id]
            assert source_anchor.char_start is not None and source_anchor.char_end is not None
            text = block.text[source_anchor.char_start : source_anchor.char_end]
            if not text.strip():
                continue
            statement_id = f"statement-{statement_index:08d}"
            statement_index += 1
            statement_anchor = TextSemanticAnchor(
                block_id=block.id,
                page_index=block.page_index,
                char_start=source_anchor.char_start,
                char_end=source_anchor.char_end,
            )
            statement_mention_ids: list[str] = []
            for candidate in extract_mention_candidates(text):
                mention_id = f"mention-{mention_index:08d}"
                mention_index += 1
                statement_mention_ids.append(mention_id)
                mentions.append(
                    SemanticMention(
                        id=mention_id,
                        kind=candidate.kind,
                        statement_id=statement_id,
                        raw_text=candidate.raw_text,
                        normalized_value=candidate.normalized_value,
                        evidence_anchors=(
                            TextSemanticAnchor(
                                block_id=block.id,
                                page_index=block.page_index,
                                char_start=source_anchor.char_start + candidate.start,
                                char_end=source_anchor.char_start + candidate.end,
                            ),
                        ),
                        provenance=SemanticProvenanceKind.RULE_BASED_TEXT,
                        legal_reference=candidate.legal_reference,
                        quantity=candidate.quantity,
                    )
                )
            statements.append(
                SemanticStatement(
                    id=statement_id,
                    structural_node_id=node.id,
                    structural_canonical_path=node.canonical_path,
                    text=text,
                    evidence_anchors=(statement_anchor,),
                    mention_ids=tuple(statement_mention_ids),
                    provenance=SemanticProvenanceKind.RULE_BASED_TEXT,
                )
            )
    node_by_id = {node.id: node for node in structural.nodes}
    for observation in visual_observations:
        if (
            observation.document_id != physical.document_id
            or observation.version_id != physical.version_id
            or observation.source_artifact_sha256 != physical.source_artifact_sha256
            or observation.evidence_anchor.source_artifact_sha256 != physical.source_artifact_sha256
            or observation.source_physical_ir_sha256 != physical_sha
            or observation.source_structural_ir_sha256 != structural_sha
        ):
            raise SemanticSourceIntegrityError("VLM observation source identity mismatch")
        if (
            observation.provenance != SemanticProvenanceKind.VLM_TRANSCRIPTION
            or observation.text is None
        ):
            continue
        owner = node_by_id.get(observation.structural_node_id)
        if owner is None or owner.canonical_path != observation.structural_canonical_path:
            raise SemanticSourceIntegrityError("VLM observation structural context mismatch")
        owned_block_ids = {anchor.block_id for anchor in owner.direct_content_anchors}
        if any(block_id not in owned_block_ids for block_id in observation.source_block_ids):
            raise SemanticSourceIntegrityError(
                "VLM observation source blocks are incompatible with structural context"
            )
        if any(block_id not in blocks for block_id in observation.source_block_ids):
            raise SemanticSourceIntegrityError("VLM observation references missing physical block")
        statement_id = f"statement-{statement_index:08d}"
        statement_index += 1
        statement_mention_ids = []
        for candidate in extract_mention_candidates(observation.text):
            mention_id = f"mention-{mention_index:08d}"
            mention_index += 1
            statement_mention_ids.append(mention_id)
            mentions.append(
                SemanticMention(
                    id=mention_id,
                    kind=candidate.kind,
                    statement_id=statement_id,
                    raw_text=candidate.raw_text,
                    normalized_value=candidate.normalized_value,
                    evidence_anchors=(observation.evidence_anchor,),
                    provenance=SemanticProvenanceKind.VLM_TRANSCRIPTION,
                    source_visual_observation_id=observation.id,
                    legal_reference=candidate.legal_reference,
                    quantity=candidate.quantity,
                )
            )
        statements.append(
            SemanticStatement(
                id=statement_id,
                structural_node_id=owner.id,
                structural_canonical_path=owner.canonical_path,
                text=observation.text,
                evidence_anchors=(observation.evidence_anchor,),
                mention_ids=tuple(statement_mention_ids),
                provenance=SemanticProvenanceKind.VLM_TRANSCRIPTION,
                source_visual_observation_id=observation.id,
            )
        )
    return SemanticDocument(
        document_id=physical.document_id,
        version_id=physical.version_id,
        source_artifact_sha256=physical.source_artifact_sha256,
        source_physical_ir_sha256=physical_sha,
        source_structural_ir_sha256=structural_sha,
        statements=tuple(statements),
        mentions=tuple(mentions),
        visual_observations=visual_observations,
    )


def validate_semantic_sources(
    semantic: SemanticDocument,
    physical: PhysicalDocumentV1,
    structural: StructuralDocument,
) -> None:
    """Validate exact identities, canonical source hashes, nodes, paths, and evidence spans."""
    validate_against_physical(structural, physical)
    expected = {
        "document_id": physical.document_id,
        "version_id": physical.version_id,
        "source_artifact_sha256": physical.source_artifact_sha256,
        "source_physical_ir_sha256": _sha256_text(physical_document_v1_to_json(physical)),
        "source_structural_ir_sha256": _sha256_text(structural_document_to_json(structural)),
    }
    for field, value in expected.items():
        if getattr(semantic, field) != value:
            raise SemanticSourceIntegrityError(f"{field} mismatch")
    blocks = {block.id: block for page in physical.pages for block in page.blocks}
    nodes = {node.id: node for node in structural.nodes}
    for statement in semantic.statements:
        node = nodes.get(statement.structural_node_id)
        if node is None or node.canonical_path != statement.structural_canonical_path:
            raise SemanticSourceIntegrityError("statement structural identity mismatch")
        reconstructed = ""
        for anchor in statement.evidence_anchors:
            if not isinstance(anchor, TextSemanticAnchor):
                continue
            block = blocks.get(anchor.block_id)
            if block is None or block.page_index != anchor.page_index:
                raise SemanticSourceIntegrityError("statement references missing physical evidence")
            if anchor.char_end > len(block.text):
                raise SemanticSourceIntegrityError("statement evidence span exceeds physical text")
            reconstructed += block.text[anchor.char_start : anchor.char_end]
        if reconstructed and reconstructed != statement.text:
            raise SemanticSourceIntegrityError("statement text differs from exact evidence")
    for mention in semantic.mentions:
        for anchor in mention.evidence_anchors:
            if not isinstance(anchor, TextSemanticAnchor):
                continue
            block = blocks.get(anchor.block_id)
            if block is None or anchor.char_end > len(block.text):
                raise SemanticSourceIntegrityError("mention references invalid physical evidence")
            if block.text[anchor.char_start : anchor.char_end] != mention.raw_text:
                raise SemanticSourceIntegrityError("mention raw_text differs from exact evidence")


__all__ = [
    "SemanticSourceIntegrityError",
    "build_semantic_document",
    "validate_semantic_sources",
]
