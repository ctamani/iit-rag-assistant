# Retrieval tooling

Run from the project root:

```powershell
python -m scripts.retrieval.validate
python -m scripts.retrieval.evaluate --report retrieval-evaluation.json
python -m pytest tests/retrieval/ -q
```

Runtime reads `backend/data/knowledge_base/` (or `IIT_RETRIEVAL_DATA`).
Runtime files are `index.faiss`, `chunks.json`, `directory.json`, and
`policy_conflicts.json`. Integrity records live in `validation/manifest.json`
and `validation/vector_mapping.json`; preserved source records, units, and
supplemental chunks live in `archive/`. Manifest file entries are relative to
the knowledge-base root, not the `validation/` directory.
Data and vectors stay local and Git-ignored. The existing index needs no rebuild.

For an intentional offline rebuild in a separate data directory:

```powershell
python -m scripts.retrieval.preprocess --original <source_records.json> --out <data_directory>
python -m scripts.retrieval.build_index --data <data_directory>
```

Optional baseline evaluation requires `--baseline-index`, `--baseline-chunks`,
and `--baseline-retriever <trusted_standalone_python_file>`; no legacy folder
layout is assumed. The specified baseline implementation is executed.
