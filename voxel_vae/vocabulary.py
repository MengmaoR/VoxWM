"""Deterministic training vocabulary and per-root semantic token remapping."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_local_vocabulary(root):
    path = Path(root) / "vocabulary.json"
    raw = json.loads(path.read_text())
    tokens = raw.get("tokens")
    if not isinstance(tokens, list) or raw.get("K", len(tokens)) != len(tokens):
        raise ValueError(f"Invalid vocabulary: {path}")
    pairs = []
    for expected, row in enumerate(tokens):
        if row.get("id") != expected:
            raise ValueError(f"Non-contiguous token IDs in {path}")
        param2 = int(row["param2"])
        if not 0 <= param2 <= 255:
            raise ValueError(f"Invalid raw param2 in {path}")
        pairs.append((str(row["node_name"]), param2))
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"Duplicate semantic token in {path}")
    return pairs, sha256_file(path)


@dataclass(frozen=True)
class RootVocabulary:
    root: str
    local_k: int
    local_sha256: str
    manifest_sha256: str
    local_to_canonical: tuple[int, ...]
    identity_mapping: bool


class CanonicalVocabulary:
    """Exact ``(Node Name, raw param2)`` pairs, with optional published ID order."""

    VERSION = 1

    def __init__(self, pairs, roots=(), *, preserve_ids=False):
        self.pairs = tuple((str(name), int(param2)) for name, param2 in pairs)
        if not preserve_ids and tuple(sorted(self.pairs)) != self.pairs:
            raise ValueError("Canonical vocabulary must be sorted by (node_name, param2)")
        if len(self.pairs) != len(set(self.pairs)):
            raise ValueError("Canonical vocabulary contains duplicates")
        if len(self.pairs) > np.iinfo(np.uint16).max + 1:
            raise ValueError("Canonical vocabulary does not fit uint16")
        self.index = {pair: token for token, pair in enumerate(self.pairs)}
        self.roots = tuple(roots)
        self.preserve_ids = preserve_ids
        self.sha256 = hashlib.sha256(_json_bytes(self.semantic_payload())).hexdigest()

    @property
    def K(self):
        return len(self.pairs)

    @property
    def air_ids(self):
        return tuple(index for index, (name, _) in enumerate(self.pairs) if name == "air")

    def semantic_payload(self):
        return [{"id": index, "node_name": name, "param2": param2}
                for index, (name, param2) in enumerate(self.pairs)]

    def to_dict(self):
        return {
            "version": self.VERSION,
            "representation": "CanonicalID(Node Name, raw param2)",
            "K": self.K,
            "sha256": self.sha256,
            "tokens": self.semantic_payload(),
            **({"id_order": "explicit"} if self.preserve_ids else {}),
            "roots": [{
                "root": root.root,
                "local_K": root.local_k,
                "local_sha256": root.local_sha256,
                "manifest_sha256": root.manifest_sha256,
                "identity_mapping": root.identity_mapping,
                "new_tokens_relative_to_root": self.K - root.local_k,
                "local_to_canonical": list(root.local_to_canonical),
            } for root in self.roots],
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("." + path.name + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text())
        pairs = [(row["node_name"], int(row["param2"])) for row in raw["tokens"]]
        roots = tuple(RootVocabulary(
            root=str(row["root"]),
            local_k=int(row["local_K"]),
            local_sha256=str(row["local_sha256"]),
            manifest_sha256=str(row["manifest_sha256"]),
            local_to_canonical=tuple(map(int, row["local_to_canonical"])),
            identity_mapping=bool(row["identity_mapping"]),
        ) for row in raw.get("roots", []))
        result = cls(pairs, roots, preserve_ids=raw.get("id_order") == "explicit")
        if (raw.get("K") != result.K or raw.get("sha256") != result.sha256
                or raw["tokens"] != result.semantic_payload()):
            raise ValueError("Canonical vocabulary K/hash mismatch")
        return result


def build_canonical_vocabulary(dataset_roots):
    roots = [Path(root).expanduser().resolve() for root in dataset_roots]
    if not roots or len(roots) != len(set(roots)):
        raise ValueError("At least one unique --dataset-root is required")
    locals_ = []
    union = set()
    for root in roots:
        manifest = root / "dataset_manifest.json"
        if not manifest.is_file():
            raise ValueError(f"Missing dataset manifest: {manifest}")
        pairs, local_hash = _read_local_vocabulary(root)
        locals_.append((root, pairs, local_hash, sha256_file(manifest)))
        union.update(pairs)
    canonical_pairs = tuple(sorted(union))
    canonical_index = {pair: index for index, pair in enumerate(canonical_pairs)}
    root_records = []
    for root, pairs, local_hash, manifest_hash in locals_:
        mapping = tuple(canonical_index[pair] for pair in pairs)
        root_records.append(RootVocabulary(
            root=str(root), local_k=len(pairs), local_sha256=local_hash,
            manifest_sha256=manifest_hash, local_to_canonical=mapping,
            identity_mapping=mapping == tuple(range(len(mapping))),
        ))
    return CanonicalVocabulary(canonical_pairs, root_records)


def mapping_array(vocabulary, root_index):
    mapping = vocabulary.roots[int(root_index)].local_to_canonical
    if not mapping:
        raise ValueError("Empty local vocabulary")
    return np.asarray(mapping, dtype=np.uint16)
