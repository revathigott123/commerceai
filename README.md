# commerceai

Uses large benchmark datasets and semantic matching methods to test where commerce
agents succeed or fail across real-world shopping workflows, built on the
[Amazon ESCI](https://github.com/amazon-science/esci-data) shopping queries dataset.

The project has two stages:

1. **Retrieval evaluation** — how well a BM25 search ranker surfaces relevant
   products for a shopping query, scored against ESCI's human relevance labels
   (NDCG@10, MRR, Precision@5).
2. **Chatbot evaluation** — how well a GPT-powered RAG chatbot *answers*
   shopping questions using those retrieved products, scored two independent
   ways: semantic similarity (SBERT) and an LLM-as-judge rubric score.

## Repository structure

```
commerceai/
├── notebooks/
│   └── CommerceAgent.ipynb   # Colab notebook: retrieval eval + batch chatbot eval
├── scripts/
│   └── interactive_chat.py   # standalone CLI: real-time chatbot testing
├── results/                  # generated eval output (CSV files, gitignored)
├── requirements.txt
└── README.md
```

## What's being evaluated, and how

- **Retrieval quality** — BM25 ranks candidate products for each query; ranking
  is scored against ESCI's Exact/Substitute/Complement/Irrelevant labels.
- **Chatbot answer quality** — for each question, the chatbot's generated
  answer is compared against a reference answer using:
  - **SBERT semantic similarity** (`sentence-transformers`, `all-mpnet-base-v2`)
    — a fast, deterministic cosine-similarity score.
  - **LLM-as-judge** (GPT via the OpenAI API) — a rubric score (correctness,
    groundedness, helpfulness, 1-5) with a written rationale.
  - The correlation between the two is reported, since low agreement is
    itself a useful finding about what each metric actually captures.

Reference answers are built **deterministically from ESCI's human relevance
labels**, not from an LLM, so the evaluation's gold standard stays independent
of the model being judged. See the caveats section in the notebook for more
on LLM-judge self-preference bias and how it's mitigated here.

## Setup

```bash
git clone https://github.com/revathigott123/commerceai.git
cd commerceai
pip install -r requirements.txt
```

You'll also need `git-lfs` installed (used to fetch the ESCI dataset), and an
OpenAI API key. Never commit your key — both entry points prompt for it via
`getpass` if it isn't already set as the `OPENAI_API_KEY` environment
variable.

## Usage

### Notebook: retrieval eval + batch chatbot eval

Open `notebooks/CommerceAgent.ipynb` in Colab (badge at the top of the
notebook) or locally with Jupyter, and run cells top to bottom. It clones the
ESCI dataset, runs the Stage 1 retrieval eval, then runs the Stage 2 batch
chatbot eval over a sampled set of queries and saves `chatbot_eval_results.csv`.

### Script: real-time interactive testing

```bash
python scripts/interactive_chat.py
```

Type shopping questions live and see the chatbot's answer plus SBERT/LLM-judge
scores on the spot. Each scored turn is appended to
`results/live_eval_results.csv`, so repeated runs build up your own real-world
eval set over time.

## Notes on cost and scale

Each evaluated query costs two OpenAI API calls (one to generate the answer,
one to judge it). With `gpt-4o-mini` this is cheap even at a few hundred
queries — see `CHAT_MODEL` / `JUDGE_MODEL` in the notebook and script to swap
models if you want higher-quality (and higher-cost) generation or judging.
`EVAL_SIZE` in the notebook controls how many queries the batch eval samples.
