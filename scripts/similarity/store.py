"""Chroma-backed vector store for ComponentFingerprints.

The store is a thin wrapper around ``chromadb.PersistentClient`` that:
- embeds each fingerprint's ``text_for_embedding`` with MiniLM-L6-v2 (lazy-loaded)
- persists to a local directory (no server, no network after model download)
- supports upsert / get / query_similar / all_fingerprints
- stores structural metadata (category_root, value_keys, template_signature) as
  Chroma metadata so we can do post-filter cosine queries later if needed

Embedding model is loaded lazily on first use to keep import time fast for
CLI tools that may not need embeddings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from scripts.similarity.fingerprint import ComponentFingerprint

COLLECTION_NAME = "component_fingerprints"
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Lazy module-level cache for the embedding model. We don't use functools.cache
# because sentence_transformers.SentenceTransformer caches are stateful and
# we want explicit control over the singleton.
_MODEL = None


def _get_model(model_name: str = DEFAULT_MODEL):
    """Lazy-load the sentence-transformers model. Idempotent."""
    global _MODEL
    if _MODEL is None or getattr(_MODEL, "_model_name", None) != model_name:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(model_name)
        _MODEL._model_name = model_name  # type: ignore[attr-defined]
    return _MODEL


def _metadata_from_fp(fp: ComponentFingerprint) -> dict:
    """Convert a ComponentFingerprint to a flat dict for Chroma metadata.

    Chroma metadata values must be str/int/float/bool. We comma-join list-typed
    fields. Missing fields use empty-string / 0 defaults so query() never
    returns 'metadata is None' errors.
    """
    return {
        "category_root": fp.category_root or "",
        "category_code": fp.category_code or "",
        "value_keys": ",".join(fp.value_keys or []),
        "attribute_keys": ",".join(fp.attribute_keys or []),
        "template_signature": fp.template_signature or "",
        "variant_count": int(fp.variant_count or 0),
        "name": fp.name or "",
    }


def _fp_from_metadata(component_id: str, document: str, meta: dict) -> ComponentFingerprint:
    """Inverse of _metadata_from_fp — used by get() and all_fingerprints()."""
    return ComponentFingerprint(
        component_id=component_id,
        name=meta.get("name", component_id),
        description="",  # we don't store description in metadata; use document if needed
        category_code=meta.get("category_code", ""),
        category_root=meta.get("category_root", ""),
        attribute_keys=[k for k in (meta.get("attribute_keys", "") or "").split(",") if k],
        value_keys=[k for k in (meta.get("value_keys", "") or "").split(",") if k],
        variant_count=int(meta.get("variant_count", 0) or 0),
        template_signature=(meta.get("template_signature") or None) or None,
        text_for_embedding=document,
    )


class FingerprintStore:
    """Persistent Chroma collection of ComponentFingerprints.

    Parameters
    ----------
    persist_dir:
        Directory where Chroma persists the collection. Created if missing.
    collection_name:
        Defaults to ``"component_fingerprints"``. Override for testing isolation.
    model_name:
        Sentence-transformers model name. Defaults to ``"all-MiniLM-L6-v2"``;
        pass a smaller model like ``"all-MiniLM-L6-v1"`` for tests if needed.

    Notes
    -----
    The store is **in-process** and uses Chroma's PersistentClient (no server).
    Embeddings are computed in-memory by sentence-transformers on the local CPU.
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = COLLECTION_NAME,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.model_name = model_name

        # Lazy import: chromadb is heavy and we don't want to require it for
        # the lightweight fingerprint builder.
        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._coll = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ---- write path ----------------------------------------------------------

    def upsert(self, fp: ComponentFingerprint) -> None:
        """Add or update a single fingerprint."""
        model = _get_model(self.model_name)
        vec = model.encode(fp.text_for_embedding).tolist()
        self._coll.upsert(
            ids=[fp.component_id],
            embeddings=[vec],
            documents=[fp.text_for_embedding],
            metadatas=[_metadata_from_fp(fp)],
        )

    def upsert_many(self, fps: list[ComponentFingerprint]) -> None:
        """Batch upsert — much faster than per-item for N>5."""
        if not fps:
            return
        model = _get_model(self.model_name)
        ids = [fp.component_id for fp in fps]
        docs = [fp.text_for_embedding for fp in fps]
        vecs = model.encode(docs).tolist()
        metas = [_metadata_from_fp(fp) for fp in fps]
        self._coll.upsert(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)

    # ---- read path -----------------------------------------------------------

    def get(self, component_id: str) -> Optional[ComponentFingerprint]:
        """Return a single fingerprint by id, or None if not found."""
        res = self._coll.get(ids=[component_id], include=["documents", "metadatas"])
        if not res["ids"]:
            return None
        return _fp_from_metadata(
            component_id=res["ids"][0],
            document=res["documents"][0],
            meta=res["metadatas"][0],
        )

    def all_fingerprints(self) -> list[ComponentFingerprint]:
        """Return every fingerprint in the collection."""
        res = self._coll.get(include=["documents", "metadatas"])
        return [
            _fp_from_metadata(cid, doc, meta)
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
        ]

    def query_similar(
        self, fp: ComponentFingerprint, n: int = 10
    ) -> list[tuple[ComponentFingerprint, float]]:
        """Return the N nearest neighbors of ``fp`` by cosine distance.

        Returns a list of ``(fingerprint, distance)`` tuples. The query itself
        will appear with distance 0 — callers should filter it out if they only
        want neighbors.
        """
        model = _get_model(self.model_name)
        vec = model.encode(fp.text_for_embedding).tolist()
        res = self._coll.query(
            query_embeddings=[vec],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        out: list[tuple[ComponentFingerprint, float]] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append((_fp_from_metadata(cid, doc, meta), float(dist)))
        return out

    def count(self) -> int:
        return self._coll.count()

    def reset(self) -> None:
        """Drop the collection. Used by tests; never call from prod code."""
        self._client.delete_collection(self.collection_name)
        self._coll = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
