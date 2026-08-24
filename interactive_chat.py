"""
Real-time interactive commerce chatbot + evaluation.

Standalone terminal companion to CommerceAgent.ipynb: same RAG chatbot
(BM25 retrieval over the ESCI dataset + GPT answer generation) and the
same SBERT + LLM-judge evaluation, but driven by live input() at the
terminal instead of a fixed batch of queries in a notebook cell.

Usage:
    python interactive_chat.py

Requires: pandas, pyarrow, rank_bm25, openai, sentence-transformers, scipy
    pip install pandas pyarrow rank_bm25 openai sentence-transformers scipy

Needs `git` and `git-lfs` on PATH to fetch the ESCI dataset the first time,
and an OPENAI_API_KEY (env var, or you'll be prompted for it).
"""

import csv
import json
import os
import subprocess
from getpass import getpass

import pandas as pd
from rank_bm25 import BM25Okapi

ESCI_DIR = "esci-data"
DATA_BASE = os.path.join(ESCI_DIR, "shopping_queries_dataset")
LIVE_RESULTS_PATH = "live_eval_results.csv"
FIELDNAMES = ["query", "chatbot_answer", "reference_answer", "sbert_similarity",
              "correctness", "groundedness", "helpfulness", "llm_judge_overall", "rationale"]

CHAT_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

JUDGE_PROMPT = """You are grading a shopping assistant's answer against a reference answer.
Score the CANDIDATE answer from 1 (bad) to 5 (excellent) on:
- correctness: does it recommend products consistent with the reference?
- groundedness: does it stick to the candidate products rather than inventing items?
- helpfulness: would a shopper find this answer useful?

Question: {query}
Reference answer: {reference}
Candidate answer: {candidate}

Respond with ONLY a JSON object of the form:
{{"correctness": <1-5>, "groundedness": <1-5>, "helpfulness": <1-5>, "rationale": "<one sentence>"}}
"""


def ensure_esci_data():
    if os.path.exists(ESCI_DIR):
        return
    print("Cloning esci-data (first run only, this may take a minute)...")
    subprocess.run(["git", "lfs", "install", "--skip-repo"], check=True)
    subprocess.run(
        ["git", "clone", "--quiet", "https://github.com/amazon-science/esci-data.git", ESCI_DIR],
        check=True,
    )
    subprocess.run(["git", "lfs", "pull"], cwd=ESCI_DIR, check=True)


def load_data():
    examples = pd.read_parquet(os.path.join(DATA_BASE, "shopping_queries_dataset_examples.parquet"))
    products = pd.read_parquet(os.path.join(DATA_BASE, "shopping_queries_dataset_products.parquet"))

    examples["small_version"] = examples["small_version"].astype(int)
    ex = examples[
        (examples["product_locale"] == "us")
        & (examples["small_version"] == 1)
        & (examples["split"] == "test")
    ].copy()

    prod_us = products[products["product_locale"] == "us"]
    df = ex.merge(prod_us[["product_id", "product_title"]], on="product_id", how="left")
    df = df.dropna(subset=["product_title"])
    return df


def tokenize(text):
    return str(text).lower().split()


def bm25_rank_one_query(group, query_text):
    titles = group["product_title"].tolist()
    tokenized = [tokenize(t) for t in titles]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(tokenize(query_text))
    ranked = group.copy()
    ranked["bm25_score"] = scores
    return ranked.sort_values("bm25_score", ascending=False)


def retrieve_context(df, query, k=5):
    group = df[df["query"] == query]
    if group.empty:
        return []
    ranked = bm25_rank_one_query(group, query)
    return ranked.head(k)["product_title"].tolist()


def chatbot_answer(client, df, query, k=5):
    context = retrieve_context(df, query, k=k)
    context_block = "\n".join(f"- {title}" for title in context) or "(no candidate products found)"

    messages = [
        {"role": "system", "content": (
            "You are a helpful shopping assistant. Answer the customer's question "
            "using ONLY the candidate products listed below. Recommend the most "
            "relevant one(s) and briefly say why. If none of the candidates are "
            "relevant, say so honestly instead of inventing a product."
        )},
        {"role": "user", "content": f"Candidate products:\n{context_block}\n\nCustomer question: {query}"},
    ]

    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
    return resp.choices[0].message.content.strip()


def build_reference_answer(df, query, k=3):
    group = df[df["query"] == query]
    exact = group[group["esci_label"] == "E"]["product_title"].tolist()
    substitute = group[group["esci_label"] == "S"]["product_title"].tolist()

    if exact:
        return f"The best matches for '{query}' are: {', '.join(exact[:k])}."
    elif substitute:
        return f"There is no exact match for '{query}', but reasonable substitutes are: {', '.join(substitute[:k])}."
    return None


def llm_judge(client, query, reference, candidate):
    prompt = JUDGE_PROMPT.format(query=query, reference=reference, candidate=candidate)
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def score_turn(client, sbert_model, util, query, answer, reference):
    sbert_score = util.cos_sim(
        sbert_model.encode(answer, convert_to_tensor=True),
        sbert_model.encode(reference, convert_to_tensor=True),
    ).item()
    verdict = llm_judge(client, query, reference, answer)
    overall = sum(verdict[k] for k in ("correctness", "groundedness", "helpfulness")) / 3
    return sbert_score, verdict, overall


def main():
    ensure_esci_data()
    print("Loading ESCI dataset...")
    df = load_data()
    print(f"Loaded {len(df)} rows, {df['query_id'].nunique()} unique queries.\n")

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

    from openai import OpenAI
    client = OpenAI()

    print("Loading SBERT model (all-mpnet-base-v2)...")
    from sentence_transformers import SentenceTransformer, util
    sbert_model = SentenceTransformer("all-mpnet-base-v2")

    print("\nReady. Type a shopping question (empty line or 'quit' to stop).\n")
    while True:
        query = input("You: ").strip()
        if not query or query.lower() in {"quit", "exit"}:
            print("Session ended.")
            break

        answer = chatbot_answer(client, df, query)
        print(f"\nChatbot: {answer}\n")

        auto_ref = build_reference_answer(df, query) if (df["query"] == query).any() else None
        if auto_ref:
            print(f"[Auto reference from ESCI labels]: {auto_ref}")
            reference = input("Press Enter to accept this reference, or type your own: ").strip() or auto_ref
        else:
            reference = input("Reference answer for scoring (Enter to skip scoring): ").strip()

        record = dict.fromkeys(FIELDNAMES)
        record["query"], record["chatbot_answer"], record["reference_answer"] = query, answer, reference or None

        if reference:
            sbert_score, verdict, overall = score_turn(client, sbert_model, util, query, answer, reference)
            record.update(verdict)
            record["sbert_similarity"] = sbert_score
            record["llm_judge_overall"] = overall
            print(f"\nSBERT similarity:  {sbert_score:.3f}")
            print(f"LLM-judge overall: {overall:.2f}  "
                  f"(correctness={verdict['correctness']}, groundedness={verdict['groundedness']}, helpfulness={verdict['helpfulness']})")
            print(f"Judge rationale: {verdict['rationale']}")
        else:
            print("(skipped scoring -- no reference provided)")

        write_header = not os.path.exists(LIVE_RESULTS_PATH)
        with open(LIVE_RESULTS_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(record)
        print(f"\n(saved to {LIVE_RESULTS_PATH})\n" + "-" * 60)


if __name__ == "__main__":
    main()
