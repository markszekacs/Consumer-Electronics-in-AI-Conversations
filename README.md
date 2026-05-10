# Scalable NLP Evaluation Framework: Hybrid Retrieval on LLM Conversations
### Consumer Electronics Purchase Intent Detection at Scale

End-to-end evaluation pipeline for detecting domain-specific signals in large-scale conversational AI data. Applied to consumer electronics purchase intent detection across 529,000 real-world LLM conversations from the WildChat dataset.

## Why This Project

Evaluating LLM outputs at scale without ground-truth labels is a core challenge in production AI systems. This framework demonstrates a practical solution: a three-stage hybrid retrieval approach that combines lexical search, semantic similarity, and LLM-as-judge validation to achieve high-precision filtering with explicit precision/recall tradeoffs at each stage.

**Key design decisions:**
- Keyword filtering casts a wide net (high recall, ~37% precision)
- Semantic filtering improves precision without losing relevant signal
- LLM-as-judge provides final verification — scalable ground truth without manual labeling
- Threshold validation module allows explicit precision/recall tuning

## Domain Application: Consumer Electronics in AI Conversations

This framework is applied to a specific domain — detecting consumer electronics purchase intent in real-world LLM conversations — but the architecture is domain-agnostic and directly transferable to any signal detection problem over large conversational datasets.

**Why consumer electronics?**
- High commercial relevance: purchasing decisions increasingly involve LLM-assisted research
- Complex signal: electronics mentions range from casual references to active purchase intent — requires multi-stage filtering to distinguish
- Rich taxonomy: multi-label classification across product categories (smartphones, laptops, audio, etc.) and intent types (comparison, recommendation, troubleshooting, purchase)

**Key findings:**
- Purchase intent conversations peak around product release cycles
- Comparison intent dominates early funnel; recommendation intent signals high purchase readiness
- ~37% of keyword-matched conversations contain only incidental electronics mentions — removed by semantic + LLM validation stages

## Results

- Processed 529,000 conversations → 565 high-precision relevant conversations
- Keyword-only precision: ~37% → Final pipeline precision: ~90%+
- Identified purchase intent patterns, temporal trends, and category-level conversion rates across consumer electronics segments

## Pipeline Architecture

```
529k WildChat conversations
        ↓
01_keyword_filter.py      → lexical matching, high recall
        ↓
02_semantic_filter.py     → embedding similarity, precision boost
        ↓
threshold_validation.py   → LLM-validated threshold calibration
        ↓
03_llm_validation.py      → GPT-4o-mini as judge, final filtering
        ↓
04_classification.py      → multi-label intent classification
        ↓
05_analysis.py            → intent distribution, trends, funnel metrics
```

**Stage-by-stage outputs:**

| Stage | Input | Output | Purpose |
|-------|-------|--------|---------|
| Keyword | 529k conversations | ~15k matched | High-recall lexical filter |
| Semantic | ~15k conversations | ~2k filtered | Embedding-based precision boost |
| Threshold validation | keyword corpus | validation_labels.csv | LLM-calibrated threshold |
| LLM validation | ~2k conversations | ~800 validated | GPT-4o-mini judge |
| Classification | ~800 conversations | classifications.csv | Intent + category labels |
| Analysis | classifications.csv | visualizations | Insights + trends |

## Technical Highlights

- **Hybrid retrieval**: combines BM25-style keyword matching with dense vector search — demonstrates awareness of precision/recall tradeoffs in retrieval systems
- **LLM-as-judge pattern**: uses GPT-4o-mini for scalable evaluation without manual ground truth — directly applicable to production LLM evaluation systems
- **Threshold calibration**: explicit validation module for similarity threshold tuning — reproducible and adjustable
- **Multi-label classification**: conversations can signal multiple intents simultaneously

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
```

Run steps in order:
```
01 → 02 → threshold_validation (optional) → 03 → 04 → 05
```

## Data

`data/classifications.csv` included as pre-computed output.
Full pipeline rerun may produce slightly different results due to random sampling in the non-keyword corpus.

Dataset: [allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat) — 529k real user–LLM conversations, April–November 2023.
