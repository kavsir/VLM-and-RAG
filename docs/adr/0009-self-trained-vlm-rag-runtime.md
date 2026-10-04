# ADR 0009: Self-trained VLM and evidence-backed RAG runtime

- Status: Proposed; implemented for review, live model validation pending

The user requested a usable VLM/RAG flow, then selected self-training with LoRA/QLoRA on a
future GPU host. Existing IR and evaluation truth remain frozen. The current computer runs
only CPU validation and retrieval; it is not the training target.

Keep training dependencies in an isolated Python 3.12 project. The application owns dataset
validation, source isolation and the prompt/response contract. Supervision must be human-checked
and document-disjoint, and cannot consume the six held-out reference sources. Pin the base-model
revision and the training dependency resolution. Record manifests and hardware with checkpoints.

Serve an adapter behind a replaceable Chat Completions boundary. The application retains
verified image bytes, raw completion evidence and IR lineage. Shared prompts prevent training
and inference schema drift. Requests require explicit opt-in and a finite per-process cap.

Use SQLite FTS5 as the initial retrieval implementation, with domain-owned evidence chunks.
Each chunk is tied to the exact Semantic document hash, statement and source anchors. Choose
one representation per document/version. Citation verification checks IDs and literal quotes;
it does not certify semantic entailment or factual completeness.

This establishes an executable local boundary and GPU handoff. Dense retrieval, automatic
hierarchy expansion, deployment/authentication and model-quality claims require separate work.
No database/framework owns the domain model; no training or service dependency enters core IR.
