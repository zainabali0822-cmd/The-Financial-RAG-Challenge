# Financial RAG Challenge — Treasury Bulletins

**Author:** Yifan Wang
**Years used:** 2018–2025
**Data source:** [Databricks OfficeQA](https://huggingface.co/datasets/databricks/officeqa)

A Baseline vs. Engineered Retrieval-Augmented Generation (RAG) system for answering
factual questions from U.S. Treasury Bulletins, built for the Self-Discovery Lab.

Full code, outputs, and reflection: [`FinancialRAG_Challenge.ipynb`](./FinancialRAG_Challenge.ipynb)

## Architecture

| Component | Baseline | Engineered |
|---|---|---|
| Vector index | FAISS `IndexFlatIP` (cosine) | FAISS `IndexFlatIP` (cosine) |
| Embedding model | `multi-qa-MiniLM-L6-cos-v1` | `multi-qa-MiniLM-L6-cos-v1` |
| Chunk size | 400 chars | 1200 chars |
| Overlap | 0 | 250 chars |
| Metadata | none | `year`, `month`, `quarter` |
| Retrieval strategy | pure semantic search | metadata pre-filter (year/month, parsed from the query) → semantic search within the matching subset |
| Generator | Gemini, with automatic model fallback (`2.5-flash-lite` → `2.0-flash-lite` → `2.0-flash-001` → `flash-lite-latest`) | same |
| Prompt | strict single-line output | table-aware, instructs the model to check row/column intersections |

The notebook is written around three small classes (`Chunker`, `VectorIndex`, `Generator`
helpers) and a `RunConfig` dataclass, so the Baseline and Engineered runs are literally
the same pipeline fed two different configs — not two separate copy-pasted scripts.

## Results

> Fill this in after running the notebook end-to-end (requires a Hugging Face token and
> a Gemini API key set as Colab secrets `HF_TOKEN` / `GEMINI_API_KEY`).

| Metric | Baseline (Simple) | Engineered (Improved) |
|---|---|---|
| Hit Rate@5 | __%__ | __%__ |
| MRR | __0.__ | __0.__ |
| Groundedness | __%__ | __%__ |
| Factual Accuracy | __%__ | __%__ |
| Hallucination Rate | __%__ | __%__ |

## Engineering Reflection

**The Bottleneck.** [Fill in once you have numbers] — compare Hit Rate@5 against
Groundedness/Factual Accuracy on the Baseline run. If Hit Rate@5 is low, the failure
is in the *retriever* ("Librarian" never surfaces the right bulletin); if Hit Rate@5
is reasonable but Factual Accuracy/Groundedness are low, the failure is in the
*generator* ("Student" has the right page but misreads the table).

**The Metadata Fix.** Adding `year`/`month`/`quarter` tags and pre-filtering FAISS
search to the matching subset is expected to move Hit Rate@5 and MRR the most, since
it shrinks the search space before semantic similarity is even computed. Groundedness
and Factual Accuracy should improve too, but only downstream of the retrieval fix.

**Scaling Insight.** Scaling from this 8-year subset (2018–2025) to the full 1939–2025
archive, the first component to break is the flat FAISS index — `IndexFlatIP` scans
every vector per query (O(n)), so both memory footprint and query latency grow linearly
with ~87 years of bulletins. At that scale, an IVF/HNSW-based FAISS index (or a managed
vector DB with sharding) plus coarser metadata partitioning (e.g., per-decade indexes)
would be needed so a single query doesn't have to touch the entire corpus.

## Setup to reproduce

1. Open `FinancialRAG_Challenge.ipynb` in Google Colab.
2. Add two Colab secrets: `HF_TOKEN` (Hugging Face, read access) and `GEMINI_API_KEY`.
3. Run all cells top to bottom — Section 12 runs both the Baseline and Engineered
   pipelines and Section 13 prints the final scorecard.
4. Copy the printed scorecard numbers into the table above.

## What this project demonstrates

- **Chunking is as important as the model.** Splitting a table in half destroys it
  regardless of how capable the downstream LLM is.
- **Retriever vs. generator diagnosis.** Hit Rate/MRR isolate search quality; Groundedness/
  Hallucination Rate isolate answer quality — together they tell you *where* to fix
  the pipeline.
- **Metadata as a search filter, not just a label.** Tagging chunks with year/month/quarter
  and using that to pre-filter FAISS search is what actually moves retrieval metrics,
  not just organizes the data.
- **You need a baseline to claim "better."** Every number here is reported as a pair
  (Baseline, Engineered) so the improvement is measurable, not asserted.
