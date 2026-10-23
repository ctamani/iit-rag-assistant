"""Validate metadata, character coverage, content hashes, and the saved vector map."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from .common import DATA, MAX_TOKENS, MODEL, REVISION, digest, load_original, read_json, require, sha256, source_url, tokenizer, token_count, write_json
from .preprocess import clean, directory_text, norm
from backend.app.retrieval.artifacts import validate_index


def validate_dataset(data: Path, check_tokens: bool=True) -> dict:
    chunks=read_json(data/"chunks.json")
    indexed_count=len(chunks)
    supplemental=read_json(data/"archive"/"supplemental_chunks.json") if (data/"archive"/"supplemental_chunks.json").exists() else []
    require(all(c.get("retrieval_exclusion") for c in supplemental),"Supplemental record without reason")
    chunks=chunks+supplemental
    units=read_json(data/"archive"/"units.json")
    rows=load_original(data/"archive"/"source_records.json")
    if (data/"audit_original.json").exists():
        require(sha256(data/"archive"/"source_records.json")==read_json(data/"audit_original.json")["original_sha256"]["chunks.json"],"Source archive differs from audited input")
    require(len({c["chunk_id"] for c in chunks})==len(chunks),"Duplicate final IDs")
    groups=defaultdict(list)
    for c in chunks: groups[c["unit_id"]].append(c)
    covered=set()
    for u in units:
        cs=sorted(groups.pop(u["unit_id"],[]),key=lambda c:c["core_start"])
        require(bool(cs),"Unindexed unit")
        require(cs[0]["core_start"]==0 and cs[-1]["core_end"]==len(u["body"]),"Missing unit endpoints")
        require(all(a["core_end"]==b["core_start"] for a,b in zip(cs,cs[1:])),"Missing or overlapping core text")
        require([c["chunk_id"] for c in cs]==u["chunk_ids"],"Incorrect unit mapping")
        for c in cs:
            require(0<=c["overlap_start"]<=c["core_start"]<=c["core_end"]<=len(u["body"]),"Invalid text span")
            require(c["content"]==c["context_prefix"]+u["body"][c["overlap_start"]:c["core_end"]],"Text/span mismatch")
            require(c["content_sha256"]==digest(c["content"]),"Content hash mismatch")
            require(c["original_indices"]==u["rows"],"Lineage mismatch")
            require(c["source_url"]==u["source_url"],"URL changed")
            require(c["source_urls"]==sorted({v for i in u["rows"] for k,v in rows[i]["metadata"].items() if k in ("url","source_url") and v}),"Source or linked URL lost")
            require(c["source_files"]==sorted({rows[i]["metadata"]["source_file"] for i in u["rows"]}),"Source file missing")
        for i in u["rows"]:
            covered.add(i)
            require(source_url(rows[i])==u["source_url"],"Sources conflated")
            txt=clean(rows[i]["text"])[0]
            if u["method"]!="complete_directory_row":
                require(txt in u["body"],f"Original text not represented: {i}")
        for e in u["evidence"]:
            if e["relation"]=="verbatim_directory_row":
                raw=rows[e["original_index"]]["text"][e["start"]:e["end"]]
                require(clean(directory_text(raw))[0]==u["body"],"Directory row changed")
            elif e["relation"]=="confirmed_name_email_or_role":
                raw=rows[e["original_index"]]["text"]
                if raw.startswith("Name: "):
                    import re
                    match=re.fullmatch(r"Name: (.+?)\.(?: Email: (.+?)\.)?",raw)
                    require(match is not None,"Unparseable directory identity")
                    name,email=match.groups()
                    require(u["person"]["name"]==name,"Person name mismatch")
                    require(u["person"]["email"]==(email or ""),"Person email mismatch")
                else: require(norm(raw) in norm(u["person"]["roles_text"]),"Unconfirmed role")
    require(not groups,"Unknown units")
    require(covered==set(range(len(rows))),"Original rows silently discarded")
    # Every directory aggregate line must have explicit evidence, not merely a
    # reference to the original multi-person row.
    directory_lines=defaultdict(list)
    for u in units:
        for e in u["evidence"]:
            if e["relation"]=="verbatim_directory_row":directory_lines[e["original_index"]].append(rows[e["original_index"]]["text"][e["start"]:e["end"]])
    for i,lines in directory_lines.items():
        require(Counter(lines)==Counter(rows[i]["text"].splitlines()),f"Directory aggregate loss: {i}")
    if check_tokens:
        tok=tokenizer()
        counts=[token_count(tok,c["content"]) for c in chunks]
        require(all(n<=MAX_TOKENS for n in counts),"Embedding would truncate")
        require(counts==[c["token_count"] for c in chunks],"Incorrect token counts")
    # Category filtering uses the one canonical array; no duplicate exports.
    return {"chunks":indexed_count,"supplemental_chunks":len(supplemental),"source_rows_covered":len(covered),"units_fully_covered":len(units),"unique_ids":True,"lineage_valid":True,"no_text_loss":True,
            "coverage_basis":"sanitized source archive, including quarantined-record tombstone"}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=DATA)
    p.add_argument("--dataset-only",action="store_true")
    p.add_argument("--report",type=Path,help="Optional review report outside the data directory")
    a=p.parse_args(); result=validate_dataset(a.data)
    if not a.dataset_only:result.update(validate_index(a.data))
    if a.report: write_json(a.report,result)
    print(result)
