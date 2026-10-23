"""Conservative, deterministic preprocessing with lossless source archive and lineage."""
from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from .common import (DATA, MAX_TOKENS, OVERLAP_TOKENS, TARGET_TOKENS, category, digest,
                    load_original, require, source_url, token_count, tokenizer, write_json)

# Reviewed spacing repairs only: no guessing damaged Unicode or policy wording.
REPAIRS = {
    "theOffice of Financial Aidto": "the Office of Financial Aid to",
    "theCode of Conduct": "the Code of Conduct",
    "atregistrar@illinoistech.edu": "at registrar@illinoistech.edu",
    "theIncomplete Grade Request Form": "the Incomplete Grade Request Form",
    "finishedallundergraduate": "finished all undergraduate",
    "theirapprovedco-terminal": "their approved co-terminal",
    "theeForm": "the eForm",
    "theCo-Terminal Planning Worksheet.This": "the Co-Terminal Planning Worksheet. This",
    "Methods page.If": "Methods page. If",
    "Account tab.Each": "Account tab. Each",
    'attendance."You': 'attendance." You',
}


def clean(text: str) -> tuple[str, list[str]]:
    changes = []
    for old,new in REPAIRS.items():
        if old in text:
            text=text.replace(old,new)
            changes.append(f"spacing: {old!r} -> {new!r}")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    if normalized != text:
        changes.append("normalize line endings / nonbreaking spaces")
    return normalized, changes


def norm(text: str) -> str:
    return " ".join(text.split())


def directory_text(raw: str) -> str:
    parts=raw.split(" - ")
    name,role,email=parts[0]," - ".join(parts[1:-1]),parts[-1]
    return f"Name: {name}.\nRoles: {role}."+(f"\nEmail: {email}." if email else "")


def section(row: dict) -> str | None:
    m=row["metadata"]
    return m.get("section_heading") or m.get("heading") or m.get("title") or m.get("page_h1") or m.get("page_title") or None


def scope(row: dict) -> tuple:
    """Never deduplicate equal wording across different supported scopes."""
    m=row["metadata"]
    return tuple(m.get(k) for k in ("academic_year","term","program","campus","semester_scope","calendar_session","source_date"))


def make_units(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    units: list[dict] = []
    issues: list[dict] = []
    structured: list[dict] = []
    handled: set[int] = set()
    rows_to_people = defaultdict(list)
    directory_dedup = {}
    # Complete directory rows are the authority for person association. The first
    # delimiter separates name; the final delimiter separates email; middle text
    # may itself contain delimiters and is preserved in full.
    for i,row in enumerate(rows):
        if category(row)!="directory" or "\n" not in row["text"]:
            continue
        lines=row["text"].splitlines(keepends=True)
        offset=0
        for line in lines:
            raw=line.rstrip("\r\n")
            parts=raw.split(" - ")
            require(len(parts)>=3, f"Unrecognized directory row {i}: {raw}")
            name, role, email=parts[0], " - ".join(parts[1:-1]), parts[-1]
            require(not email or re.fullmatch(r"[^\s@]+@[^\s@]+",email) is not None, f"Ambiguous email at {i}")
            key=(source_url(row),raw,scope(row))
            if key not in directory_dedup:
                u={"body":directory_text(raw),"person":{"name":name,"roles_text":role,"email":email},"rows":[],"evidence":[],"kind":"directory","topic":name,"changes":[],"method":"complete_directory_row"}
                units.append(u)
                directory_dedup[key]=u
                person={"name":name,"roles_text":role or None,"email":email or None,
                        "department":None,"source_url":source_url(row),"unit":u}
                structured.append(person)
            u=directory_dedup[key]
            if i not in u["rows"]: u["rows"].append(i)
            u["evidence"].append({"original_index":i,"start":offset,"end":offset+len(raw),"relation":"verbatim_directory_row"})
            rows_to_people[row["metadata"]["source_file"]].append((name,email,role,u))
            offset+=len(line)
        handled.add(i)
    # A fragment is attached only with matching name/email in the same export
    # group, and a role is attached only when the complete row confirms it.
    current = None
    for i,row in enumerate(rows):
        if category(row)!="directory": continue
        if i in handled:
            current=None
            continue
        txt=row["text"]
        candidates=rows_to_people[row["metadata"]["source_file"]]
        match=re.fullmatch(r"Name: (.+?)\.(?: Email: (.+?)\.)?",txt)
        chosen=None
        if match:
            name,email=match.groups()
            found={id(u):(n,e,r,u) for n,e,r,u in candidates if n==name and (e or None)==email}
            if len(found)==1:
                current=next(iter(found.values()))
                chosen=current[3]
            else:
                current=None
        elif current and norm(txt) in norm(current[2]) and current[3] in [c[3] for c in candidates]:
            chosen=current[3]
        if chosen is not None:
            chosen["rows"].append(i)
            chosen["evidence"].append({"original_index":i,"start":0,"end":len(txt),"relation":"confirmed_name_email_or_role"})
            handled.add(i)
        else:
            issues.append({"original_index":i,"issue":"directory association ambiguous; fragment retained independently"})
    # Reviewed non-adjacent headings: the export places both headings before
    # their bodies. Match exact text + source, never positional proximity alone.
    hardship="https://www.iit.edu/registrar/policies-and-procedures/physical-or-financial-hardship-withdrawal-policy"
    reviewed_groups=[
        ("Official and Unofficial Transcripts",("For official transcripts,", "If a hold has been placed on a student's transcript")),
        ("Physical or Financial Hardship Withdrawal Request",("Hardship withdrawal submissions", "As a general rule, students are not eligible for hardship withdrawals", "Questions regarding the Illinois Tech Hardship Withdrawal Policy")),
    ]
    for heading,starts in reviewed_groups:
        heads=[i for i,r in enumerate(rows) if source_url(r)==hardship and r["text"]==heading]
        bodies=[[i for i,r in enumerate(rows) if source_url(r)==hardship and r["text"].startswith(s)] for s in starts]
        if len(heads)==1 and all(len(x)==1 for x in bodies):
            indices=heads+[x[0] for x in bodies]
            units.append({"body":"\n\n".join(rows[i]["text"] for i in indices),"rows":indices,"evidence":[],"kind":category(rows[indices[0]]),"topic":heading,"changes":[],"method":"reviewed_heading_body_merge"})
            handled.update(indices)
    # Exact same-source content duplicates are represented once with all origins.
    dedup={}
    for i,row in enumerate(rows):
        if i in handled: continue
        key=(source_url(row) or row["metadata"].get("source_file"),row["text"],section(row),scope(row))
        if key not in dedup:
            u={"body":row["text"],"rows":[],"evidence":[],"kind":category(row),"topic":section(row),"changes":[],"method":"original_record"}
            units.append(u); dedup[key]=u
        dedup[key]["rows"].append(i)
    # Redundant headings can safely be linked to the sole same-source body
    # containing the exact heading phrase. Ambiguous headings remain indexed.
    removed=set()
    for u in units:
        txt=u["body"].strip()
        if u["kind"]=="directory" or u["method"]!="original_record" or len(txt)>100 or len(txt.split())<4 or txt.endswith((".","?",":")):
            continue
        source=source_url(rows[u["rows"][0]])
        if not source: continue
        matches=[v for v in units if v is not u and v["kind"]==u["kind"] and len(v["body"])>len(txt)+100
                 and source_url(rows[v["rows"][0]])==source and scope(rows[v["rows"][0]])==scope(rows[u["rows"][0]]) and txt in v["body"]]
        if len(matches)==1:
            v=matches[0]
            v["rows"].extend(u["rows"])
            v["changes"].append("linked redundant heading verbatim contained in body")
            v["evidence"].extend({"original_index":i,"relation":"verbatim_contained_heading","start":0,"end":len(rows[i]["text"])} for i in u["rows"])
            removed.add(id(u))
    units=[u for u in units if id(u) not in removed]
    # Stable identity is content + supported source metadata, not original ID.
    for u in units:
        u["rows"]=sorted(set(u["rows"]))
        u["body"],changes=clean(u["body"])
        u["changes"].extend(changes)
        first=rows[u["rows"][0]]
        u["source_url"]=source_url(first)
        u["source_files"]=sorted({rows[i]["metadata"]["source_file"] for i in u["rows"]})
        identity=(u["source_url"],u["source_files"],u["topic"],u["body"])
        # Preserve legacy IDs where scope is unknown; distinguish otherwise
        # identical records when term/program/campus evidence is available.
        if any(scope(first)): identity=identity+(scope(first),)
        u["unit_id"]="unit_"+digest(str(identity))[:24]
        if '\ufffd' in u["body"]:
            issues.append({"unit_id":u["unit_id"],"issue":"Unicode replacement character; original bytes cannot be recovered from provided export"})
        if not u["source_url"]:
            issues.append({"unit_id":u["unit_id"],"issue":"source URL absent; not inferred"})
        if len(u["body"].split())<15 and u["kind"]!="directory":
            issues.append({"unit_id":u["unit_id"],"issue":"short fragment or link; no reliable body association established"})
    email_names=defaultdict(set)
    for p in structured:
        if p["email"]: email_names[p["email"]].add(p["name"])
    for email,names in email_names.items():
        if len(names)>1:
            issues.append({"email":email,"names":sorted(names),"issue":"same email with different names; records not conflated"})
    for p in structured:
        u=p.pop("unit")
        p.update({"unit_id":u["unit_id"],"original_indices":u["rows"],"source_files":u["source_files"]})
    # Whitespace normalization can make previously different records identical.
    by_id={}
    for u in units:
        if u["unit_id"] in by_id:
            existing=by_id[u["unit_id"]]
            require(existing["body"]==u["body"],"Unit hash collision")
            existing["rows"]=sorted(set(existing["rows"]+u["rows"]))
            existing["evidence"].extend(u["evidence"])
            existing["changes"].extend(u["changes"])
        else: by_id[u["unit_id"]]=u
    units=list(by_id.values())
    require(len({u["unit_id"] for u in units})==len(units),"Unit ID collision")
    return units,structured,issues,{"complete_directory_rows":sum(len(r["text"].splitlines()) for r in rows if category(r)=="directory" and "\n" in r["text"]),"confirmed_directory_fragment_rows":sum(len(u["evidence"])-sum(e["relation"]=="verbatim_directory_row" for e in u["evidence"]) for u in units if u["kind"]=="directory"),"redundant_heading_units_linked":len(removed)}


def split_spans(body: str, prefix: str, tok) -> list[tuple[int,int,int]]:
    """Return (core start, core end, overlap start); core ranges partition body.

    Prefer paragraph boundaries near target, then sentence boundaries, then
    whitespace. Overlap repeats at most 40 tokens within the same source unit.
    Character offsets prevent loss from tokenizer decode/encode normalization.
    """
    if token_count(tok,prefix+body)<=MAX_TOKENS: return [(0,len(body),0)]
    paragraphs=[m.end() for m in re.finditer(r"\n\s*\n",body)]
    sentences=[m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9\u2022])|\n",body)]
    spaces=[m.end() for m in re.finditer(r"\s+",body)]
    result=[]; start=0
    while start<len(body):
        overlap=start
        if result:
            prior=result[-1][0]
            candidates=[x for x in sentences if prior<=x<start]
            candidates += [x for x in spaces if prior<=x<start]
            for pos in sorted(set(candidates)):
                if token_count(tok,body[pos:start])<=OVERLAP_TOKENS:
                    overlap=pos; break
        if token_count(tok,prefix+body[overlap:])<=MAX_TOKENS:
            result.append((start,len(body),overlap)); break
        # Binary search a character bound; then prefer complete paragraphs / sentences.
        lo=start+1; hi=len(body)
        while lo<hi:
            mid=(lo+hi+1)//2
            if token_count(tok,prefix+body[overlap:mid])<=TARGET_TOKENS: lo=mid
            else: hi=mid-1
        limit=lo
        end=None
        for boundaries in (paragraphs,sentences,spaces):
            candidates=[x for x in boundaries if start<x<=limit and token_count(tok,body[start:x])>=80]
            if candidates: end=max(candidates); break
        if end is None: end=limit
        require(end>start,"Splitter failed to advance")
        require(token_count(tok,prefix+body[overlap:end])<=MAX_TOKENS,"Splitter exceeded cap")
        result.append((start,end,overlap)); start=end
    return result


def preprocess(original: Path, out: Path, review_dir: Path | None=None) -> dict:
    require(out.resolve()!=original.parent.resolve(),"Use a separate output directory; never write into the original artifact directory")
    rows=load_original(original)
    tok=tokenizer()
    units,people,issues,stats=make_units(rows)
    chunks=[]
    for u in units:
        row=rows[u["rows"][0]]; m=row["metadata"]
        context=[]
        # Avoid amplifying orphan navigation/heading fragments with page titles.
        # Their source context is still available as metadata and provenance.
        if u["kind"]!="directory" and token_count(tok,u["body"])>=20:
            for label in (u["topic"],m.get("term")):
                if label and label not in context and token_count(tok,str(label))<75: context.append(str(label))
            for key in ("academic_year","program","campus","source_date"):
                value=m.get(key)
                if value and str(value) not in u["body"] and value not in context:
                    context.append(str(value))
        prefix=" | ".join(context)+"\n\n" if context else ""
        spans=split_spans(u["body"],prefix,tok)
        u["chunk_ids"]=[]
        for start,end,overlap in spans:
            content=prefix+u["body"][overlap:end]
            cid="iit_"+digest(u["unit_id"]+str((start,end,overlap))+content)[:24]
            u["chunk_ids"].append(cid)
            original_ids=sorted({rows[i]["metadata"]["chunk_id"] for i in u["rows"] if rows[i]["metadata"].get("chunk_id")})
            # Metadata originals remain intact in source_records.json. All fields
            # here are derived from source evidence or explicitly labeled categories.
            chunks.append({"chunk_id":cid,"content":content,"topic":u["topic"],
                "document_type":u["kind"],"document_name":m.get("page_title") or m.get("title") or None,
                "source_url":u["source_url"],
                "source_urls":sorted({v for i in u["rows"] for k,v in rows[i]["metadata"].items() if k in ("url","source_url") and v}),
                "source_file":u["source_files"][0],"source_files":u["source_files"],
                "original_chunk_ids":original_ids,"original_indices":u["rows"],"unit_id":u["unit_id"],
                "core_start":start,"core_end":end,"overlap_start":overlap,"context_prefix":prefix,
                "token_count":token_count(tok,content),"content_sha256":digest(content),
                **{k:m.get(k) for k in ("academic_year","term","program","campus","semester_scope","calendar_session","source_date","verified_on","verification_status","conflict_group")}})
    for p in people:
        p["chunk_ids"]=next(u["chunk_ids"] for u in units if u["unit_id"]==p["unit_id"])
        p["entity_type"]="person"
    for u in units:
        m=rows[u["rows"][0]]["metadata"]
        if m.get("directory_entity"):
            people.append({**m["directory_entity"],"unit_id":u["unit_id"],"chunk_ids":u["chunk_ids"],
                           "original_indices":u["rows"],"source_files":u["source_files"],"source_url":u["source_url"]})
    # Keep uncertain fragments and their exact character lineage, but do not
    # embed orphan headings/questions. Verified facts and complete identities stay.
    directory_units={p["unit_id"] for p in people}
    indexed=[]; supplemental=[]
    for c in chunks:
        reason=next((rows[i]["metadata"].get("retrieval_exclusion") for i in c["original_indices"] if rows[i]["metadata"].get("retrieval_exclusion")),None)
        if not reason and c["token_count"]<20 and c["unit_id"] not in directory_units and c["verification_status"]!="official_verified":
            text=c["content"].strip()
            dated_event=c["document_type"]=="calendar" and c.get("term") and re.search(r"\b20\d\d-\d{2}-\d{2}\b",text)
            reviewed=any(rows[i]["metadata"].get("short_fragment_review")=="retain" for i in c["original_indices"])
            if not (dated_event or reviewed): reason="Short orphan heading, navigation, unanswered question or insufficient standalone context"
        if reason:
            supplemental.append(dict(c,retrieval_exclusion=reason))
        else: indexed.append(c)
    out.mkdir(parents=True,exist_ok=True)
    (out/"archive").mkdir(exist_ok=True)
    if original.resolve()!=(out/"archive"/"source_records.json").resolve():
        shutil.copyfile(original,out/"archive"/"source_records.json")
    write_json(out/"archive"/"units.json",units)
    if review_dir: write_json(review_dir/"provenance_issues.json",issues)
    write_json(out/"directory.json",people)
    write_json(out/"chunks.json",indexed)
    write_json(out/"archive"/"supplemental_chunks.json",supplemental)
    stats.update({"original_entries":len(rows),"units":len(units),"final_chunks":len(indexed),"supplemental_chunks":len(supplemental),
        "by_category":dict(Counter(c["document_type"] for c in indexed)),"structured_directory_records":len(people),
        "merged_units_multiple_original_rows":sum(len(u["rows"])>1 for u in units),
        "split_units":sum(len(u["chunk_ids"])>1 for u in units),
        "additional_chunks_from_splitting":len(chunks)-len(units),
        "cleaned_units":sum(any(s.startswith(("spacing:","normalize")) for s in u["changes"]) for u in units),
        "units_with_recorded_changes":sum(bool(u["changes"]) for u in units),
        "short_under_20_tokens":sum(c["token_count"]<20 for c in indexed),
        "max_tokens":max(c["token_count"] for c in chunks),"issue_records":len(issues)})
    stats["cleaned_original_rows"]=sum(bool(clean(r["text"])[1]) for r in rows)
    if review_dir: write_json(review_dir/"preprocessing_summary.json",stats)
    return stats


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",required=True,type=Path)
    p.add_argument("--out",default=DATA,type=Path)
    p.add_argument("--review-dir",type=Path,help="Optional diagnostics, separate from application data")
    a=p.parse_args(); print(preprocess(a.original,a.out,a.review_dir))
