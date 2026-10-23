"""Shared index integrity checks used at startup and by offline validation."""
from pathlib import Path

import faiss
import numpy as np

from .common import MODEL, REVISION, digest, read_json, require, sha256


def validate_index(data: Path, *, chunks=None, index=None, full: bool=True) -> dict:
    chunks=read_json(data/"chunks.json") if chunks is None else chunks
    manifest=read_json(data/"validation"/"manifest.json")
    require(manifest["embedding_model"]==MODEL and manifest["embedding_revision"]==REVISION,"Embedding model/revision mismatch")
    require(manifest["document_normalization"]==manifest["query_normalization"]=="L2","Normalization mismatch")
    runtime={"chunks.json","index.faiss","validation/vector_mapping.json","policy_conflicts.json","directory.json"}
    require(runtime<=set(manifest["files"]),"Manifest missing runtime checksums")
    for filename, expected in manifest["files"].items():
        require((data/filename).resolve().is_relative_to(data.resolve()),"Manifest path escapes data directory")
        if full or filename in runtime:
            require(sha256(data/filename)==expected,f"Checksum mismatch: {filename}")
    ix=faiss.read_index(str(data/"index.faiss")) if index is None else index
    require(type(ix).__name__=="IndexFlatIP" and ix.metric_type==faiss.METRIC_INNER_PRODUCT,"Incorrect index type")
    require(ix.ntotal==len(chunks)==manifest["vector_count"],"Vector count mismatch")
    require(ix.d==manifest["dimension"]==768,"Dimension mismatch")
    mapping=read_json(data/"validation"/"vector_mapping.json")
    require(mapping==[{"vector_position":i,"chunk_id":c["chunk_id"],"content_sha256":digest(c["content"])} for i,c in enumerate(chunks)],"Invalid vector mapping")
    if not full:
        return {"vector_count":ix.ntotal,"dimension":ix.d,"mapping_valid":True,"runtime_checksums_valid":True}
    v=ix.reconstruct_n(0,ix.ntotal)
    require(np.isfinite(v).all(),"Nonfinite vectors")
    norms=np.linalg.norm(v,axis=1)
    require(np.allclose(norms,1,atol=2e-5,rtol=0),"Unnormalized embeddings")
    scores,positions=ix.search(v[:min(10,len(v))],1)
    require(np.allclose(scores,1,atol=2e-5),"Reload/self-search failure")
    require(all(0<=p<len(chunks) for p in positions[:,0]),"Search returned invalid row")
    return {"index_reloaded":True,"vector_count":ix.ntotal,"dimension":ix.d,"norm_min":float(norms.min()),"norm_max":float(norms.max()),"finite":True,"mapping_valid":True,"self_search_valid":True}


