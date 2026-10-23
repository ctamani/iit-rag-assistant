"""Focused regression checks for scope, filters, security, and directory ranking."""
import json
import re
import unittest
from unittest.mock import patch
import numpy as np

from backend.app.retrieval.common import DATA, read_json
from scripts.retrieval.preprocess import make_units
from backend.app.retrieval.retriever import Retriever


class UpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=Retriever()

    def test_scope_preserved_during_deduplication(self):
        meta={"source_file":"calendar.json","source_url":"https://example.invalid/calendar","chunk_type":"calendar"}
        rows=[{"text":"Registration opens November 9.","metadata":dict(meta,term=term)} for term in ("Fall 2026","Fall 2027")]
        units,_,_,_=make_units(rows)
        self.assertEqual(len(units),2)
        self.assertEqual(len({u["unit_id"] for u in units}),2)

    def test_filter_finds_exact_eligible_top_k(self):
        query="When does registration open?"
        filters={"document_type":"calendar","program":"Coursera"}
        results=self.r.search_vector(query,5,filters=filters)
        vector=self.r.model.encode([query],normalize_embeddings=True,convert_to_numpy=True).astype('float32')
        _,positions=self.r.index.search(vector,len(self.r.chunks))
        expected=[self.r.chunks[int(i)]["chunk_id"] for i in positions[0] if self.r._matches(self.r.chunks[int(i)],filters)][:5]
        self.assertEqual([r['chunk_id'] for r in results],expected)
        self.assertEqual(len(results),5)

    def test_unknown_filter_rejected_and_absent_scope_excluded(self):
        with self.assertRaisesRegex(ValueError,"Unknown filter"):
            self.r.search("credit hours",filters={"category":"registrar"})
        with self.assertRaisesRegex(ValueError,"Unknown document_type"):
            self.r.search("credit hours",filters={"document_type":"typo"})
        self.assertEqual(self.r.search("deadline",filters={"term":"Fall 2099"}),[])

    def test_directory_does_not_bypass_category_filter(self):
        hits=self.r.search("How can I email Javad Abbasian?",filters={"document_type":"registrar"})
        self.assertTrue(hits)
        self.assertTrue(all(h['document_type']=='registrar' for h in hits))

    def test_exact_directory_identity_avoids_embedding(self):
        with patch.object(self.r.model,"encode",side_effect=AssertionError("unnecessary embedding")):
            hits=self.r.search("How can I email Javad Abbasian?")
        self.assertIn('abbasian@illinoistech.edu',hits[0]['content'])

    def test_email_substring_is_not_exact_identity(self):
        with patch.object(self.r,"search_vector",return_value=[]) as fallback:
            self.assertEqual(self.r.search("email xabbasian@illinoistech.edu"),[])
            self.assertTrue(fallback.called)

    def test_repeat_conflict_visible_even_top_one(self):
        hit=self.r.search("How many courses can a graduate student repeat?",1)[0]
        conflict=hit['policy_conflict']
        self.assertEqual(conflict['status'],'unresolved')
        self.assertEqual(len(conflict['evidence']),4)
        self.assertTrue(any('Two distinct' in e['claim'] for e in conflict['evidence']))
        self.assertTrue(any('three courses' in e['claim'] for e in conflict['evidence']))
        # A broad source record may also have a credit-load warning; an explicit
        # repeat question must still expose the repeat conflict at top_k=1.
        decorated=self.r._decorate([{'conflict_group':'credit_load'}],'graduate course repeat limit')
        self.assertEqual(decorated[0]['policy_conflict']['status'],'unresolved')

    def test_fall_deadline_program_and_year(self):
        hits=self.r.search("full-semester last day to withdraw",filters={"term":"Fall 2026","program":"Main Registrar calendar"})
        self.assertIn('2026-11-02',hits[0]['content'])
        self.assertTrue(all(h['academic_year']=='2026-2027' for h in hits))

    def test_corrected_summer_graduate_limit(self):
        hits=self.r.search("graduate credit limit",filters={"semester_scope":"Summer"})
        self.assertIn('graduate students 6',hits[0]['content'])

    def test_explicit_graduate_tuition_retrieves_matching_period_before_top_k(self):
        hits = self.r.search("What is the graduate tuition per credit hour for Fall 2025?")
        self.assertTrue(hits)
        self.assertTrue(all(hit["document_type"] == "tuition" for hit in hits))
        self.assertTrue(all("Fall 2025" in hit["content"] for hit in hits))
        self.assertTrue(any("$1851" in hit["content"] for hit in hits))
        self.assertTrue(all("mies-campus-graduate" in hit["source_url"] for hit in hits))
        self.assertFalse(any("undergraduate" in hit["source_url"] for hit in hits))

    def test_tuition_does_not_substitute_another_period_or_ignore_filters(self):
        self.assertEqual(self.r.search(
            "What is the graduate tuition per credit hour for Fall 2099?"), [])
        self.assertEqual(self.r.search(
            "What is the graduate tuition per credit hour for Fall 2025?",
            filters={"document_type": "calendar"}), [])

    def test_coursera_b_scope(self):
        hits=self.r.search("courses begin",filters={"term":"Fall 2026 Term B","program":"Coursera"})
        self.assertIn('2026-10-12',hits[0]['content'])

    def test_explicit_spring_calendar_query_excludes_fall(self):
        for query in ('When is the Spring 2026 full-semester last day to add or drop a course?',
                      'When is the Spring 2026 full-semester add/drop deadline?'):
            hits=self.r.search(query)
            self.assertTrue(hits)
            self.assertIn('2026-01-20',hits[0]['content'])
            self.assertTrue(all(h['term']=='Spring 2026' for h in hits))
            self.assertFalse(any('ID B Add/Drop Deadline' in h['content'] for h in hits))
        self.assertEqual(self.r.search('When is the Fall 2099 full-semester add/drop deadline?'),[])

    def test_summer_session_keeps_baseline_iso_year(self):
        hits=self.r.search('When is the Summer 2026 Session 1 add/drop deadline?')
        self.assertTrue(hits)
        self.assertIn('2026-05-22',hits[0]['content'])
        self.assertTrue(all(h['term']=='Summer 1' for h in hits))

    def test_period_in_policy_question_does_not_force_calendar(self):
        hits=self.r.search('When can a graduate student repeat a course in Fall 2026?')
        self.assertTrue(hits)
        self.assertTrue(any(h.get('policy_conflict') for h in hits))

    def test_supplemental_information_retained(self):
        supplement=read_json(DATA/'archive'/'supplemental_chunks.json')
        indexed={c['chunk_id'] for c in self.r.chunks}
        self.assertTrue(supplement)
        self.assertTrue(all(c['chunk_id'] not in indexed for c in supplement))
        self.assertTrue(self.r.supplemental_lookup('Frequently Asked Questions'))

    def test_security_redactions_apply_to_all_artifacts(self):
        for path in DATA.rglob('*.json'):
            text=path.read_text(encoding='utf8')
            self.assertIsNone(re.search(r'(?i)passcode\s*:\s*(?!\[REDACTED_PENDING_REVIEW\])\S+',text),path.name)
        source=read_json(DATA/'archive/source_records.json')
        self.assertTrue(source[2949]['text'].startswith('[QUARANTINED_PENDING_REVIEW:'))
        self.assertFalse(any(2949 in c['original_indices'] for c in self.r.chunks))

    def test_verified_office_has_no_invented_person_role(self):
        entity=self.r.directory_lookup(name='Office of Student Accounting')[0]
        self.assertEqual(entity['entity_type'],'office')
        self.assertEqual(entity['phone'],'312-567-3794')


if __name__=='__main__': unittest.main()
