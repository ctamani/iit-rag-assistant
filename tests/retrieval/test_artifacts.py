"""Run with python -m unittest discover -s tests/retrieval -v."""
from __future__ import annotations
import copy
import unittest
from unittest.mock import patch

from backend.app.retrieval.common import DATA, MAX_TOKENS, read_json, tokenizer, token_count
from scripts.retrieval.preprocess import clean, make_units, split_spans
from scripts.retrieval.validate import validate_dataset
from backend.app.retrieval.artifacts import validate_index


class ArtifactTests(unittest.TestCase):
    def test_complete_dataset_coverage(self):
        result=validate_dataset(DATA)
        self.assertTrue(result["no_text_loss"])
        self.assertGreaterEqual(result["source_rows_covered"],3256)

    def test_index_roundtrip_mapping_and_normalization(self):
        self.assertTrue(validate_index(DATA)["mapping_valid"])

    def test_duplicate_ids_rejected(self):
        chunks=copy.deepcopy(read_json(DATA/"chunks.json"))
        chunks[1]["chunk_id"]=chunks[0]["chunk_id"]
        def altered(path):
            return chunks if path.name=="chunks.json" else read_json(path)
        with patch("scripts.retrieval.validate.read_json",side_effect=altered):
            with self.assertRaisesRegex(ValueError,"Duplicate final IDs"):
                validate_dataset(DATA,check_tokens=False)

    def test_mismatched_text_rejected(self):
        chunks=copy.deepcopy(read_json(DATA/"chunks.json"))
        chunks[0]["content"]="incorrect vector text"
        def altered(path):
            return chunks if path.name=="chunks.json" else read_json(path)
        with patch("scripts.retrieval.validate.read_json",side_effect=altered):
            with self.assertRaisesRegex(ValueError,"Text/span mismatch"):
                validate_dataset(DATA,check_tokens=False)

    def test_missing_source_coverage_rejected(self):
        units=copy.deepcopy(read_json(DATA/"archive"/"units.json"))
        units[0]["body"]+="This was never indexed."
        def altered(path):
            return units if path.name=="units.json" else read_json(path)
        with patch("scripts.retrieval.validate.read_json",side_effect=altered):
            with self.assertRaisesRegex(ValueError,"Missing unit endpoints"):
                validate_dataset(DATA,check_tokens=False)

    def test_people_do_not_merge_on_shared_role(self):
        meta={"source_file":"directory.json","source_url":"https://example.invalid/directory","chunk_type":"directory_people"}
        rows=[{"text":t,"metadata":dict(meta)} for t in (
            "Alice One - Professor - a@example.invalid\nBob Two - Professor - b@example.invalid",
            "Name: Alice One. Email: a@example.invalid.","Professor",
            "Name: Bob Two. Email: b@example.invalid.","Professor")]
        units,people,issues,_=make_units(rows)
        self.assertEqual(len(people),2)
        self.assertEqual(units[0]["rows"],[0,1,2])
        self.assertEqual(units[1]["rows"],[0,3,4])
        self.assertFalse(issues)

    def test_ambiguous_person_preserved(self):
        meta={"source_file":"directory.json","source_url":"https://example.invalid/directory","chunk_type":"directory_people"}
        rows=[{"text":t,"metadata":dict(meta)} for t in (
            "Alice One - Professor - a@example.invalid\nAlice One - Dean - a@example.invalid",
            "Name: Alice One. Email: a@example.invalid.","Unconfirmed role")]
        units,people,issues,_=make_units(rows)
        self.assertEqual(len(people),2)
        self.assertEqual(len(units),4)
        self.assertTrue(any("ambiguous" in i["issue"] for i in issues))

    def test_directory_title_not_attached_by_generic_heading_rule(self):
        meta={"source_file":"directory.json","source_url":"https://example.invalid/directory","chunk_type":"directory_people"}
        rows=[{"text":t,"metadata":dict(meta)} for t in (
            "Alice One - Assistant Professor - a@example.invalid\nBob Two - Assistant Professor of Law, Associate Dean for Academic Affairs and Undergraduate Instruction - b@example.invalid",
            "Name: Alice One. Email: a@example.invalid.","Assistant Professor of Law")]
        units,people,issues,_=make_units(rows)
        self.assertTrue(any(u["body"]=="Assistant Professor of Law" and u["rows"]==[2] for u in units))
        self.assertTrue(any("ambiguous" in i["issue"] for i in issues))

    def test_structured_contact_matches_require_evidence(self):
        from backend.app.retrieval.retriever import Retriever
        r=Retriever.__new__(Retriever)
        r.chunks=[{"chunk_id":"a","content":"Alice One, Director of Chemical Engineering"}]
        r.people=[{"name":"Alice One","email":"a@example.invalid","roles_text":"Director of Chemical Engineering","chunk_ids":["a"]}]
        r.conflicts={}
        r._prepare_directory()
        r.search_vector=lambda query,top_k,**kwargs: []
        self.assertEqual(r.search("How can I email Alice One?")[0]["retrieval_method"],"directory_exact_identity")
        self.assertEqual(r.search("Who is the Director of Chemical Engineering?")[0]["retrieval_method"],"directory_all_role_terms")
        self.assertEqual(r.search("Who is the Director of Electrical Engineering?"),[])
        self.assertEqual(r.search("How can I email Alice Ones?"),[])

    def test_cleaning_preserves_numbers_and_uncertain_unicode(self):
        original="Pay $10,250 by 2026-01-20; contact theOffice of Financial Aidto ask. Andr\ufffd"
        corrected,_=clean(original)
        self.assertEqual(corrected,"Pay $10,250 by 2026-01-20; contact the Office of Financial Aid to ask. Andr\ufffd")

    def test_split_preserves_all_characters_and_token_cap(self):
        tok=tokenizer()
        body=("A policy with a $125 fee applies on 2026-01-20. "*160)+("Extraordinarilylongword"*80)
        prefix="Policy\n\n"
        spans=split_spans(body,prefix,tok)
        self.assertEqual("".join(body[a:b] for a,b,_ in spans),body)
        self.assertTrue(all(token_count(tok,prefix+body[o:b])<=MAX_TOKENS for _,b,o in spans))

    def test_structured_directory_is_indexed(self):
        chunks={c["chunk_id"]:c for c in read_json(DATA/"chunks.json")}
        for p in read_json(DATA/"directory.json"):
            self.assertTrue(p["chunk_ids"])
            self.assertTrue(all(cid in chunks for cid in p["chunk_ids"]))
        person=next(p for p in read_json(DATA/"directory.json") if p["name"]=="Javad Abbasian")
        text=" ".join(chunks[cid]["content"] for cid in person["chunk_ids"])
        self.assertIn("abbasian@illinoistech.edu",text)
        self.assertIn("Director of Graduate Affairs",text)


if __name__=="__main__":unittest.main()
