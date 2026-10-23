"""Application retrieval; construct once at startup and reuse for all requests."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import faiss
import numpy as np
from .common import DATA, load_model, read_json, require, token_count
from .artifacts import validate_index
from backend.app.shared.student_parser import find_level_mentions

FILTER_FIELDS={"document_type","academic_year","term","program","campus","semester_scope","calendar_session"}
STOP=set("who what is are the of in and how can i you them their a an at to for with get find contact email address department please".split())


class Retriever:
    def __init__(self, data: Path=DATA, model=None):
        self.data=Path(data)
        self.chunks=read_json(self.data/"chunks.json")
        self.index=faiss.read_index(str(self.data/"index.faiss"))
        # Full vector reconstruction belongs to offline validation, not startup.
        validate_index(self.data,chunks=self.chunks,index=self.index,full=False)
        self.model=model if model is not None else load_model()
        self.people=read_json(self.data/"directory.json")
        self.conflicts=read_json(self.data/"policy_conflicts.json")
        self._source_records=None
        self._supplemental=None
        self._prepare_directory()

    def _prepare_directory(self):
        self.by_id={c["chunk_id"]:(i,c) for i,c in enumerate(self.chunks)}
        self.directory_terms=[(p,re.compile(r"(?<!\w)"+re.escape(p["name"].casefold())+r"(?!\w)"),
                               set(re.findall(r"[a-z]+",(p["roles_text"] or "").casefold()))) for p in self.people]

    def _validate_filters(self, filters):
        filters={} if filters is None else filters
        require(isinstance(filters,dict) and set(filters)<=FILTER_FIELDS,"Unknown filter field")
        require(all(isinstance(v,str) and v.strip() for v in filters.values()),"Filters require nonempty strings")
        if "document_type" in filters:
            require(filters["document_type"] in {c["document_type"] for c in self.chunks},"Unknown document_type")
        return filters

    @staticmethod
    def _matches(chunk, filters):
        return all(isinstance(chunk.get(k),str) and chunk[k].casefold()==v.casefold() for k,v in filters.items())

    def _decorate(self, hits, query):
        query_conflict=bool(re.search(r"\brepeat",query,re.I) and re.search(r"graduate|course|GPA",query,re.I))
        for hit in hits:
            group="graduate_repeat" if query_conflict else hit.get("conflict_group")
            if group in self.conflicts:
                # Even top_k=1 carries both official claims and their scope.
                hit["policy_conflict"]=self.conflicts[group]
        return hits

    def search_vector(self, query: str, top_k: int=5, *, filters: dict | None=None) -> list[dict]:
        require(bool(query.strip()),"Query cannot be empty")
        require(top_k>0,"top_k must be positive")
        filters=self._validate_filters(filters)
        require(token_count(self.model.tokenizer,query)<=self.model.max_seq_length,"Query exceeds model token limit; shorten it explicitly")
        vector=self.model.encode([query],normalize_embeddings=True,convert_to_numpy=True).astype(np.float32)
        require(np.isfinite(vector).all(),"Nonfinite query embedding")
        # Searching every row before filtering guarantees the true top-k eligible
        # results, unlike an arbitrary over-fetch window. This is a small corpus.
        count=len(self.chunks) if filters else min(top_k,len(self.chunks))
        scores,positions=self.index.search(vector,count)
        hits=[]
        for score,pos in zip(scores[0],positions[0]):
            if pos>=0 and self._matches(self.chunks[int(pos)],filters):
                hits.append(dict(self.chunks[int(pos)],score=float(score),vector_position=int(pos),retrieval_method="faiss"))
                if len(hits)==top_k: break
        return self._decorate(hits,query)

    def search(self, query: str, top_k: int=5, *, filters: dict | None=None) -> list[dict]:
        """Exact directory identities/roles first, otherwise normalized FAISS.

        Filters apply to both paths; absent scope never means all programs/terms.
        Pass policy_conflict through to the caller without resolving its claims.
        """
        require(bool(query.strip()),"Query cannot be empty")
        require(top_k>0,"top_k must be positive")
        filters=self._validate_filters(filters)
        if re.search(r"\b(who|email|contact|directory|phone|address)\b",query,re.I):
            q=query.casefold()
            terms=set(re.findall(r"[a-z]+",q))-STOP
            emails=set(re.findall(r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9.-]+\.[a-z]{2,}",q))
            hits=[]; seen=set()
            for person,name_pattern,role_words in self.directory_terms:
                named=bool(name_pattern.search(q))
                emailed=bool(person["email"] and person["email"].casefold() in emails)
                role_match=len(terms)>=3 and terms<=role_words
                if not (named or emailed or role_match): continue
                for cid in person["chunk_ids"]:
                    i,c=self.by_id[cid]
                    if cid not in seen and self._matches(c,filters):
                        hits.append(dict(c,score=None,vector_position=i,retrieval_method="directory_exact_identity" if named or emailed else "directory_all_role_terms"))
                        seen.add(cid)
            if hits: return self._decorate(hits[:top_k],query)
        periods=set(re.findall(r"\b(spring|summer|fall)\s+(20\d{2})\b",query.casefold()))
        levels = find_level_mentions(query)
        if (len(periods) == 1 and len(levels) == 1
                and re.search(r"\btuition\b", query, re.I)
                and re.search(r"\bper[\s-]+credit\b", query, re.I)):
            # Similar summer rates and fees can outrank the requested tuition
            # table. Apply explicit level/period scope before taking top-k.
            season, year = next(iter(periods))
            period = re.compile(r"\b" + season + r"\s+" + year + r"\b", re.I)
            level = next(iter(levels))
            hits = self.search_vector(query, len(self.chunks), filters=filters)
            return [hit for hit in hits
                    if hit["document_type"] == "tuition"
                    and period.search(hit["content"])
                    and re.search(r"\bper[\s-]+credit\b", hit["content"], re.I)
                    and level in find_level_mentions(
                        hit["content"] + " " + (hit.get("program") or "")
                        + " " + (hit.get("source_url") or ""))][:top_k]
        calendar_intent=re.search(r"\b(when|date|deadline|calendar)\b",query,re.I) and re.search(r"\b(registration|withdraw|withdrawal|add|drop|exam|exams|grading|grade|grades|begin|begins|start|starts|end|ends|conferral|break)\b",query,re.I)
        if len(periods)==1 and calendar_intent and "term" not in filters and filters.get("document_type","calendar")=="calendar":
            # Embeddings alone often confuse equal events in different years.
            # Respect an explicit period in a calendar question without guessing
            # a student's program for unrelated policy or tuition questions.
            season,year=next(iter(periods))
            period=season+" "+year
            session=re.search(r"\b(?:term|session)\s+([ab12])\b",query,re.I)
            full_semester=bool(re.search(r"full[ -]semester",query,re.I))
            requested_events=[]
            if re.search(r"\b(add|drop)\b",query,re.I): requested_events.append(r"\b(add|drop)\b")
            if re.search(r"\bwithdraw(?:al)?\b",query,re.I): requested_events.append(r"\bwithdraw(?:al)?\b")
            hits=self.search_vector(query,len(self.chunks),filters=filters)
            scoped=[]
            for hit in hits:
                term=(hit.get("term") or "").casefold()
                # Some baseline terms are named "Summer 1"/"Summer 2". Only
                # use the year from an explicit ISO event date in that case.
                dated_year=not re.search(r"\b20\d{2}\b",term) and re.search(r"\b"+year+r"-\d{2}-\d{2}\b",hit['content'])
                period_match=period in term or (dated_year and re.search(r"\b"+season+r"\b",term))
                if hit["document_type"]!="calendar" or not period_match: continue
                if full_semester and term!=period: continue
                if full_semester and len(requested_events)==1:
                    # A single date row may say "Full Semester Withdrawal;
                    # ID B Add/Drop". Never attach one clause's scope to the
                    # other clause's event just because the embedding is close.
                    clauses=re.split(r"[;\n]|(?<=\.)\s+",hit["content"])
                    if not any(re.search(r"full[ -]semester",clause,re.I) and re.search(requested_events[0],clause,re.I) for clause in clauses): continue
                if "coursera" in query.casefold() and hit.get("program")!="Coursera": continue
                if session and not re.search(r"\b"+session[1]+r"\b",term,re.I): continue
                scoped.append(hit)
            return scoped[:top_k]
        return self.search_vector(query,top_k,filters=filters)

    def directory_lookup(self, name: str | None=None, email: str | None=None) -> list[dict]:
        """Return all exact case-insensitive matches; do not guess aliases."""
        require(bool(name or email),"Supply a name or email")
        return [p for p in self.people if (not name or p["name"].casefold()==name.casefold()) and
                (not email or (p["email"] or "").casefold()==email.casefold())]

    def original_records(self, chunk: dict) -> list[dict]:
        """Sanitized provenance. The untouched original remains in the old package."""
        if self._source_records is None:
            self._source_records=read_json(self.data/"archive"/"source_records.json")
        return [{"original_index":i,**self._source_records[i]} for i in chunk["original_indices"]]

    def supplemental_lookup(self, query: str, top_k: int=10) -> list[dict]:
        """Explicit lexical review of preserved fragments, not default evidence.

        All query words must occur. These records have no vectors and must not be
        mistaken for ranked answers. Security quarantine contains only a tombstone.
        """
        require(bool(query.strip()) and top_k>0,"Supply a query and positive top_k")
        if self._supplemental is None:
            self._supplemental=read_json(self.data/"archive"/"supplemental_chunks.json")
        terms=set(re.findall(r"\w+",query.casefold()))
        require(bool(terms),"Query must contain words")
        return [dict(c,score=None,retrieval_method="supplemental_review") for c in self._supplemental
                if terms<=set(re.findall(r"\w+",c["content"].casefold()))][:top_k]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("query")
    p.add_argument("--data",type=Path,default=DATA)
    p.add_argument("--top-k",type=int,default=5)
    p.add_argument("--category")
    a=p.parse_args()
    print(json.dumps(Retriever(a.data).search(a.query,a.top_k,filters={"document_type":a.category} if a.category else None),ensure_ascii=False,indent=2))
