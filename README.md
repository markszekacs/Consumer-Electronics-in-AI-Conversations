# Consumer Electronics in AI Conversations

Analysis of purchase intent patterns in real-world AI conversations, using the WildChat dataset.

## Pipeline

The filtering is three-stage by design - each step serves a different purpose:

**01_keyword_filter.py**
Splits the 529k WildChat conversations into keyword-matched and non-keyword corpora.
Outputs: `data/keyword_matched`, `data/non_keyword`

**02_semantic_filter.py**
Filters both corpora using embedding-based similarity against a set of representative electronics queries. Keyword filtering alone has ~37% precision - semantic filtering cleans this up.
Outputs: `data/semantic_filtered_keyword_matched`, `data/semantic_filtered_non_keyword`, `data/similarities_*.npy`

**threshold_validation.py**
LLM-validates the precision of the 0.15 similarity threshold on the keyword corpus. Run this if you want to verify or adjust the threshold.
Output: `data/validation_labels.csv`

**03_llm_validation.py**
Final relevance check using GPT-4o-mini. Removes conversations where electronics is mentioned incidentally. Combines both pipelines into a single clean dataset.
Output: `data/final_dataset` (~565 conversations)

**04_classification.py**
Classifies each conversation by product category and purchase intent using GPT-4o-mini. Multi-label - a conversation can have multiple intents.
Output: `data/classifications.csv`

**05_analysis.py**
Intent distribution, temporal trends, category breakdown, and funnel conversion rates.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add your OpenAI API key to .env
```

Run steps in order: 01 → 02 → threshold_validation (optional) → 03 → 04 → 05

## Data

`data/classifications.csv` is included as a pre-computed output. Running the full pipeline may produce slightly different results due to random sampling in the non-keyword corpus.

Dataset: [allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat) - 529k real user conversations with LLMs, April-November 2023.
