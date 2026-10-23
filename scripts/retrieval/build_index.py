"""Batch-embed finalized content, normalize it, and save a new IndexFlatIP."""
from __future__ import annotations
import argparse
import importlib.metadata
import platform
from datetime import datetime, timezone
from pathlib import Path
import faiss
import numpy as np
from .common import DATA, MODEL, REVISION, MAX_TOKENS, TARGET_TOKENS, OVERLAP_TOKENS, digest, load_model, read_json, require, sha256, write_json
from .validate import validate_dataset, validate_index


def build(data: Path, batch_size: int=32) -> dict:
    require((data/"chunks.json").is_file(),"Missing processed chunks.json")
    validation=validate_dataset(data)
    model=load_model()
    chunks=read_json(data/"chunks.json")
    content=[c["content"] for c in chunks]
    vectors=model.encode(content,batch_size=batch_size,show_progress_bar=True,convert_to_numpy=True,normalize_embeddings=True).astype("float32")
    require(vectors.shape==(len(chunks),768),"Unexpected embedding shape")
    require(np.isfinite(vectors).all(),"Nonfinite embeddings")
    require(np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-5),"Unnormalized embeddings")
    ix=faiss.IndexFlatIP(768); ix.add(vectors)
    index_path=data/"index.faiss"
    temporary=data/"index.faiss.tmp"
    faiss.write_index(ix,str(temporary)); temporary.replace(index_path)
    write_json(data/"validation"/"vector_mapping.json",[{"vector_position":i,"chunk_id":c["chunk_id"],"content_sha256":digest(c["content"])} for i,c in enumerate(chunks)])
    # Verify all saved vectors equal the embeddings of these exact ordered strings.
    reloaded=faiss.read_index(str(index_path))
    require(np.array_equal(reloaded.reconstruct_n(0,len(chunks)),vectors),"Serialization changed vectors")
    manifest={"schema_version":2,"embedding_model":MODEL,"embedding_revision":REVISION,
        "lineage_basis":"sanitized source records; all preserved text covered by indexed plus supplemental chunks",
        "dimension":768,"vector_count":len(chunks),"index_type":"IndexFlatIP","metric":"inner_product",
        "document_normalization":"L2","query_normalization":"L2","model_max_seq_length":model.max_seq_length,
        "max_tokens_including_context_and_specials":MAX_TOKENS,"target_tokens":TARGET_TOKENS,"max_overlap_tokens":OVERLAP_TOKENS,
        "batch_size":batch_size,"device":"cpu","created_utc":datetime.now(timezone.utc).isoformat(),
        "python":platform.python_version(),"platform":platform.platform(),
        "dependencies":{p:importlib.metadata.version(p) for p in ("sentence-transformers","transformers","torch","numpy","faiss-cpu","tokenizers","huggingface-hub")},
        "ordered_content_digest":digest("\n".join(digest(t) for t in content)),
        "all_saved_vectors_equal_generated_embeddings":True,
        "source_archive_sha256":sha256(data/"archive"/"source_records.json"),
        "files":{p.relative_to(data).as_posix():sha256(p) for p in sorted(data.rglob("*")) if p.is_file() and p.name not in {"manifest.json","validation_results.json","retrieval_results.json","preprocessing_summary.json","issues.json"}}}
    write_json(data/"validation"/"manifest.json",manifest)
    validation.update(validate_index(data))
    return validation


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=DATA)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--report",type=Path)
    a=p.parse_args();result=build(a.data,a.batch_size)
    if a.report: write_json(a.report,result)
    print(result)
