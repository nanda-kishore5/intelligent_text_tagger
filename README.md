# Intelligent Text Tagger

Simple Python system to tag plain-text/markdown documents using TF-IDF + POS-based keyword extraction,
with a feedback loop that updates tag weights so suggestions improve over time.

## Setup

1. Create a virtualenv and install dependencies:
   python -m venv .venv
   source .venv/bin/activate   # (or .venv\Scripts\activate on Windows)
   pip install -r requirements.txt

2. Download NLTK data (one-time):
   python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

## Basic usage (CLI)
- Suggest tags for all docs in `data/`:
  python -m tagger.cli suggest --data-dir data --top-k 5

- Approve a tag:
  python -m tagger.cli feedback --doc sample1.txt --tag "meeting" --action approve

- Reject a tag:
  python -m tagger.cli feedback --doc sample1.txt --tag "draft" --action reject

- Show current weights:
  python -m tagger.cli show-weights

## Tests
  pytest

## Design decisions
- Uses TF-IDF to find high-signal words and combines that with noun-phrase candidates via NLTK POS tags.
- Keeps a `feedback.json` with:
  - `tag_weights`: global weight per tag (float)
  - `doc_feedback`: per-doc approved/rejected tag lists (for audit)
- When a tag is approved, its weight increases; on rejection, weight decreases (bounded).
- Suggestions combine TF-IDF score and learned `tag_weight` to rank final tags.

## How feedback improves the system
- The raw extraction produces candidates (TF-IDF candidates and noun phrases).
- The system multiplies a candidate score by `(1 + tag_weight[tag])`, so tags that have been
  approved in the past get boosted and appear more often; rejected tags are penalized.
- This is a lightweight, interpretable online learning approach; weights are persisted in `feedback.json`.

## Extensibility
- LLMs: use an LLM for semantic tag generation, then use user feedback to fine-tune a reranker
  (or store embeddings and use vector similarity to match tags).
- Vector search: embed documents and tags; find nearest tag candidates and allow user feedback
  to adjust tag vectors or ranking logic.
