"""Small, explicit information-level smoke set, not a general quality benchmark."""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path
import re
import faiss
import numpy as np
from .common import DATA, MODEL, REVISION, digest, load_model, read_json, require, write_json
from backend.app.retrieval.retriever import Retriever

CASES=[
    {"id":"graduate_repeat","question":"How many distinct courses may a graduate student repeat, and which grade counts toward the GPA?",
     "all_patterns":[r"two distinct courses",r"latest recorded|last grade"],"evidence_original_indices":[2795]},
    {"id":"coterminal_shared","question":"How many credit hours can a co-terminal student share between the bachelor's and master's degrees?",
     "all_patterns":[r"nine credit hours|9 credit hours",r"shar"],"evidence_original_indices":[1]},
    {"id":"transcript_portal","question":"How do current students order an official transcript through the Illinois Tech Portal?",
     "all_patterns":[r"Order Official Transcript",r"search bar"],"evidence_original_indices":[3143]},
    {"id":"spring_add_drop","question":"When is the Spring 2026 full-semester last day to add or drop a course?",
     "all_patterns":[r"2026-01-20",r"Add/Drop"],"evidence_original_indices":[2676]},
    {"id":"javad_contact","question":"What are Javad Abbasian's Chemical Engineering roles and email address?",
     "all_patterns":[r"Javad Abbasian",r"abbasian@illinoistech.edu",r"Professor of Chemical Engineering",r"Director of Graduate Affairs"],"evidence_original_indices":[12,15,16,17]},
    {"id":"chemical_graduate_director","question":"Who is the Director of Graduate Affairs in Chemical and Biological Engineering and how can I email them?",
     "all_patterns":[r"Javad Abbasian",r"abbasian@illinoistech.edu",r"Director of Graduate Affairs"],"evidence_original_indices":[12]},
    {"id":"aid_graduate_only","question":"Do graduate-only courses count toward undergraduate federal and state financial aid for co-terminal students?",
     "all_patterns":[r"graduate degree",r"will not count",r"federal/state"],"evidence_original_indices":[2755]},
    {"id":"graduate_tuition","question":"What was the per-credit-hour tuition for Mies Campus graduate students in Fall 2025?",
     "all_patterns":[r"1851",r"Fall 2025",r"credit hour"],"evidence_original_indices":[3033,3216]},
    {"id":"hardship_transcript","question":"Can an employer receive my official transcript if I owe a debt to Illinois Tech?",
     "all_patterns":[r"official transcript",r"employer",r"owes a debt|notwithstanding a financial hold"],"evidence_original_indices":[3251,3145,3139]},
]

NEW_CASES=[
    {"id":"fall_2026_add_drop","question":"When is the Fall 2026 full-semester add/drop deadline without tuition charges?","all_patterns":[r"Fall 2026",r"2026-08-25",r"full.semester"]},
    {"id":"fall_2026_withdraw","question":"When is the Fall 2026 last day to withdraw from full-semester courses?","all_patterns":[r"Fall 2026",r"2026-11-02",r"withdraw"]},
    {"id":"fall_2026_finals","question":"When is final exam week in Fall 2026?","all_patterns":[r"Fall 2026",r"2026-12-07",r"2026-12-12"]},
    {"id":"summer_graduate_limit","question":"What is the summer graduate credit-hour limit and how do I request an overload?","all_patterns":[r"Summer",r"graduate students 6",r"approval",r"Overload"]},
    {"id":"new_continuing_grad_limits","question":"How do fall and spring credit-hour limits differ for new and continuing graduate students?","all_patterns":[r"new graduate.{0,70}15",r"Continuing graduate.{0,60}18"]},
    {"id":"drop_withdraw_effects","question":"How does dropping versus withdrawing affect tuition, maximum credits, the transcript and GPA?","all_patterns":[r"Dropping",r"Withdrawing",r"W does not affect GPA",r"tuition and fees payable",r"counts toward maximum"]},
    {"id":"coursera_summer_withdraw","question":"When was the Coursera Summer 2026 withdrawal deadline?","all_patterns":[r"Coursera",r"Summer 2026",r"2026-07-24"]},
    {"id":"coursera_fall_b_start","question":"When do Coursera Fall 2026 Term B courses begin?","all_patterns":[r"Coursera",r"Fall 2026 Term B",r"2026-10-12"]},
    {"id":"kent_llm_tuition","question":"What is Chicago-Kent full-time LL.M. tuition per semester in 2026–2027?","all_patterns":[r"Chicago.Kent",r"2026.2027",r"22,000 per semester"]},
    {"id":"accounting_contact","question":"What is the Office of Student Accounting phone number and email address?","all_patterns":[r"312-567-3794",r"sa@illinoistech.edu"]},
]


def relevant(text: str, case: dict) -> bool:
    return all(re.search(p,text,re.I|re.S) for p in case["all_patterns"])


def summarize(results: list[dict], case: dict) -> dict:
    hits=[i+1 for i,r in enumerate(results) if relevant(r["content"],case)]
    return {"hit_at_5":bool(hits),"first_relevant_rank":min(hits) if hits else None,
            "results":[{**{k:v for k,v in r.items() if k!="content"},"relevant":relevant(r["content"],case)} for r in results]}


def evaluate(data: Path, baseline_index: Path | None=None, baseline_chunks: Path | None=None, report: Path | None=None, *, baseline_retriever: Path | None=None) -> dict:
    model=load_model(); retriever=Retriever(data,model=model)
    baseline=None; original=None
    if baseline_index or baseline_chunks:
        require(bool(baseline_index and baseline_chunks),"Supply both baseline paths")
        baseline=faiss.read_index(str(baseline_index)); original=read_json(baseline_chunks)
        require(baseline.ntotal==len(original) and baseline.d==768,"Invalid baseline mapping dimensions")
        manifest=read_json(baseline_chunks.parent/"validation"/"manifest.json")
        require(manifest["embedding_model"]==MODEL and manifest["embedding_revision"]==REVISION,"Different baseline embedding model")
        mapping=read_json(baseline_chunks.parent/"validation"/"vector_mapping.json")
        require(mapping==[{"vector_position":i,"chunk_id":c["chunk_id"],"content_sha256":digest(c["content"])} for i,c in enumerate(original)],"Invalid baseline vector mapping")
        # Execute the actual previous directory-ranking code, without invoking
        # its expensive constructor a second time or changing baseline files.
        require(baseline_retriever is not None,"Supply --baseline-retriever for the trusted standalone baseline implementation")
        spec=importlib.util.spec_from_file_location("baseline_retrieval",baseline_retriever)
        legacy=importlib.util.module_from_spec(spec)
        # A trusted standalone baseline may still use its own sibling imports.
        # This path is scoped to offline loading, never application runtime.
        sys.path.insert(0,str(baseline_retriever.resolve().parent))
        try:
            spec.loader.exec_module(legacy)
        finally:
            sys.path.pop(0)
        old_retriever=legacy.Retriever.__new__(legacy.Retriever)
        old_retriever.data=baseline_chunks.parent
        old_retriever.chunks=original; old_retriever.index=baseline; old_retriever.model=model
        old_retriever.people=read_json(baseline_chunks.parent/"directory.json")
    output={"evaluation_type":"nine unchanged smoke questions plus ten gap questions; evidence-hit checks, not answer correctness or a held-out benchmark",
            "top_k":5,"cases":[],"baseline_comparison":baseline is not None}
    for case in CASES+NEW_CASES:
        new=retriever.search_vector(case["question"],5)
        selected=[{k:r[k] for k in ("chunk_id","content","source_url","score","vector_position","original_indices")} for r in new]
        result={**case,"suite":"existing" if case in CASES else "new","improved":summarize(selected,case)}
        assisted=retriever.search(case["question"],5)
        result["integrated"]=summarize([{k:r[k] for k in ("chunk_id","content","source_url","score","vector_position","original_indices","retrieval_method")} for r in assisted],case)
        if baseline is not None:
            v=model.encode([case["question"]],normalize_embeddings=True).astype("float32")
            scores,positions=baseline.search(v,5)
            old=[{"chunk_id":original[int(i)]["chunk_id"],"content":original[int(i)]["content"],"source_url":original[int(i)]["source_url"],"score":float(s)} for s,i in zip(scores[0],positions[0])]
            result["baseline"]=summarize(old,case)
            result["baseline_integrated"]=summarize([{k:r[k] for k in ("chunk_id","content","source_url","retrieval_method")} for r in old_retriever.search(case["question"],5)],case)
        output["cases"].append(result)
    output["improved_hits_at_5"]=sum(c["improved"]["hit_at_5"] for c in output["cases"])
    output["integrated_hits_at_5"]=sum(c["integrated"]["hit_at_5"] for c in output["cases"])
    if baseline is not None:
        output["baseline_hits_at_5"]=sum(c["baseline"]["hit_at_5"] for c in output["cases"])
        # A binary does not identify its embedding model. Compare reproducible
        # samples spread across the original array, including each smoke anchor.
        sample=sorted(set(np.linspace(0,len(original)-1,20,dtype=int).tolist()))
        regenerated=model.encode([original[i]["content"] for i in sample],normalize_embeddings=True,batch_size=16).astype("float32")
        stored=np.stack([baseline.reconstruct(i) for i in sample]);stored/=np.linalg.norm(stored,axis=1,keepdims=True)
        cos=np.sum(stored*regenerated,axis=1)
        output["baseline_vector_sample_check"]={"indices":sample,"cosines":[float(x) for x in cos],"min_cosine":float(cos.min()),"mean_cosine":float(cos.mean()),"count_above_0_9999":int((cos>0.9999).sum()),"note":"Supports sampled vector/text/model consistency only; original model revision is not recoverable from FAISS."}
    # Independent re-embedding of saved-index rows checks correspondence after load.
    sample=sorted(set(np.linspace(0,len(retriever.chunks)-1,16,dtype=int).tolist()))
    vectors=model.encode([retriever.chunks[i]["content"] for i in sample],normalize_embeddings=True).astype("float32")
    saved=np.stack([retriever.index.reconstruct(i) for i in sample])
    error=float(np.abs(vectors-saved).max())
    require(np.allclose(vectors,saved,atol=2e-5,rtol=0),"Saved vectors do not match re-embedded text")
    output["improved_reembedding_check"]={"sample_count":len(sample),"max_absolute_error":error,"passed":True}
    require(bool(retriever.directory_lookup(name="Javad Abbasian")),"Structured directory lookup failed")
    output["structured_lookup_passed"]=True
    for suite in ("existing","new"):
        cases=[c for c in output["cases"] if c["suite"]==suite]
        output[suite+"_summary"]={"questions":len(cases),**{mode:sum(c[mode]["hit_at_5"] for c in cases) for mode in ("baseline","baseline_integrated","improved","integrated") if mode in cases[0]}}
    if report: write_json(report,output)
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=DATA)
    p.add_argument("--baseline-index",type=Path)
    p.add_argument("--baseline-chunks",type=Path)
    p.add_argument("--baseline-retriever",type=Path,help="Trusted standalone baseline Python implementation (only for comparison)")
    p.add_argument("--report",type=Path)
    a=p.parse_args();r=evaluate(a.data,a.baseline_index,a.baseline_chunks,a.report,baseline_retriever=a.baseline_retriever)
    for c in r["cases"]:print(c["id"],"improved",c["improved"]["first_relevant_rank"],"baseline",c.get("baseline",{}).get("first_relevant_rank"))
    print({k:v for k,v in r.items() if k not in {"cases","baseline_vector_sample_check"}})
