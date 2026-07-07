# Financial RAG Challenge

**Name:** Zainab Ali

**Recent Years Used:** 2018-2025

**Data Source:** [Databricks OfficeQA](https://huggingface.co/datasets/databricks/officeqa)

**GitHub Link:** https://github.com/zainabali0822-cmd/The-Financial-RAG-Challenge

A Baseline vs. Engineered Retrieval-Augmented Generation (RAG) system for answering
factual questions from U.S. Treasury Bulletins.


**How metadata was used to filter search:** each chunk is tagged with the `year`,
`month`, and `quarter` parsed from its source bulletin's filename. At query time, the
question text is scanned for a 4-digit year and a month name; if found, the Engineered
retriever restricts the FAISS search to only the chunks matching that year/month before
computing similarity, instead of searching the entire corpus.

The notebook is organized around small reusable classes (`Chunker`, `VectorIndex`) and a
`RunConfig` dataclass, so the Baseline and Engineered runs are the same pipeline fed two
different configs, not two separate copy-pasted scripts.

## Part 1: The Scorecard

| Metric | Baseline (Simple) | Engineered (Improved) |
|---|---|---|
| Hit Rate (K=5) | 20.73% | 45.12% |
| MRR | 0.1742 | 0.3182 |
| Groundedness | NA | 29.88% |
| Factual Accuracy | 0% | 0% |
| Hallucination Rate | 0% | 0% |


## Part 2: Engineering Reflection

**The Bottleneck**
My Baseline's Hit Rate@5 was 20.73%, meaning the retriever failed to surface the correct bulletin for roughly 4 out of 5 questions. Factual Accuracy was 0% and Groundedness was N/A (no numeric claims were extracted at all — the model mostly returned INSUFFICIENT_CONTEXT since it rarely had the right chunk to work from). Since Hit Rate was already this low, the primary bottleneck was the Retriever, not the Generator: the "Librarian" wasn't handing the "Student" the right page most of the time, so there was rarely a real answer for the model to get right or wrong in the first place.

**The Metadata Fix**
Adding year/month/quarter metadata and pre-filtering FAISS search to the matching subset more than doubled retrieval performance — Hit Rate@5 rose from 20.73% to 45.12%, and MRR nearly doubled from 0.1742 to 0.3182. This confirms metadata filtering helped the Retrieval metrics directly, by shrinking the search space before similarity was computed. Groundedness moved too (from N/A to 29.88%), but this was a downstream effect: once retrieval started finding the right bulletin more often, the model had actual numeric content to draw from and ground its answers in. Factual Accuracy stayed at 0% in both configs, though — this points to a second, separate problem beyond retrieval: even when the correct chunk was retrieved (up to 45% of the time), the model's answers still didn't match the ground truth closely enough to be scored correct. That's worth digging into separately — likely the strict formatting/tolerance requirements of the scoring function, or the model echoing extra numbers alongside the right one.

**Scaling Insight**
Scaling from this 8-year subset (2018–2025) to the full 1939–2025 archive, the first component likely to break is the flat FAISS index (IndexFlatIP). It scans every vector per query — O(n) — so both memory footprint and query latency would grow roughly 10x with ~87 years of bulletins instead of 8. At that scale, an approximate index (IVF or HNSW) plus coarser metadata partitioning (e.g., per-decade sub-indexes) would be needed so a single query doesn't have to scan the entire 80-year corpus.


