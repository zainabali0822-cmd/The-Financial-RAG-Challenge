# Financial RAG Challenge — Treasury Bulletins

**Name:** Zainab Ali
**Recent Years Used:** 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025
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

**The Bottleneck.** [Fill in once you have numbers.] Compare Hit Rate@5 against
Groundedness/Factual Accuracy on the Baseline run. A low Hit Rate with tiny,
non-overlapping 400-char chunks points to a retriever problem — the search never
surfaces the right bulletin, so no amount of prompt engineering can fix the answer.
A reasonable Hit Rate paired with low Factual Accuracy/Groundedness would instead
point to a generator problem — the right page was found, but the model misread it.

**The Metadata Fix.** [Fill in once you have numbers.] Adding year/month/quarter tags
and pre-filtering FAISS search to the matching subset is expected to move Hit Rate@5
and MRR the most, since it shrinks the search space before similarity is even computed.
Groundedness and Factual Accuracy should improve too, but only as a downstream effect
of the retrieval fix — they don't move on their own.

**Scaling Insight.** Scaling from this 8-year subset (2018–2025) to the full 1939–2025
archive, the first component to break is the flat FAISS index. `IndexFlatIP` scans
every vector per query (O(n)), so both memory footprint and query latency grow linearly
with roughly 87 years of bulletins. At that scale, an IVF/HNSW-based FAISS index (or a
managed vector DB with sharding) plus coarser metadata partitioning (e.g., per-decade
indexes) would be needed so a single query doesn't have to touch the entire corpus.

## Setup to reproduce

1. Open `FinancialRAG_Challenge.ipynb` in Google Colab.
2. Add two Colab secrets: `treasury-rag1` (your Hugging Face token, read access) and
   `GEMINI_API_KEY`.
3. Run all cells top to bottom — the pipeline section runs both the Baseline and
   Engineered configs, and the scorecard section prints the final numbers.
4. Copy the printed scorecard into the table above.
