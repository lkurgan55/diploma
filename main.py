"""Run Text-to-SQL on mini_dev (SQLite) with schema + example rows."""

import argparse
import os
import re
import json
import os

import gc, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.schema.table_schema import generate_schema_prompt_sqlite
from src.schema.prompt import build_prompt
from decoding.greedy import GreedyStrategy
from decoding.beam import BeamStrategy
from decoding.top_k import TopKStrategy
from decoding.top_p import TopPStrategy
from time import time

WS = re.compile(r"\s+")

def normalize_sql(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = s.replace("```sql", "```").strip("`")
    s = WS.sub(" ", s.lower()).strip()
    s = s[:-1].strip() if s.endswith(";") else s
    return s

# ---------- Decoding ----------
def gen_with_strategy(model, tokenizer, prompt: str, strategy: str, max_new_tokens: int, db_id: str = '', db_path: str = '') -> str:

    strategy = strategy.lower()
    if strategy == "greedy":
        out = GreedyStrategy(model=model, tokenizer=tokenizer).generate(prompt, max_new_tokens=max_new_tokens)
    elif strategy == "beam":
        out = BeamStrategy(model=model, tokenizer=tokenizer).generate(prompt, max_new_tokens=max_new_tokens)
    elif strategy == "eg_beam":
        out = BeamStrategy(model=model, tokenizer=tokenizer).eg_generate(prompt, max_new_tokens=max_new_tokens, db_path=db_path)
    elif strategy == "egla_beam":
        out = BeamStrategy(model=model, tokenizer=tokenizer).egla_generate(prompt, max_new_tokens=max_new_tokens, db_path=db_path)
    elif strategy == "top_k":
        out = TopKStrategy(model=model, tokenizer=tokenizer).generate(prompt, max_new_tokens=max_new_tokens)
    elif strategy == "top_p":
        out = TopPStrategy(model=model, tokenizer=tokenizer).generate(prompt, max_new_tokens=max_new_tokens)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return out if out else "None"

# ---------- Post-process model output ----------
def extract_sql_from_text(full_text: str) -> str:
    """Extracts the SQL query from the model's output text."""
    sql = full_text.split("assistant", 1)[-1].strip()
    return sql.strip().strip("`")

# ---------- I/O ----------
def load_examples(json_path: str) -> list[dict[str, int | str]]:
    """Loads examples from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Run Text-to-SQL on mini_dev (SQLite) with schema + example rows.")
    ap.add_argument("--model", type=str, default="qwen2.5-3B-Instruct")
    ap.add_argument("--data_json", type=str, default="./datasets/data_minidev/mini_dev_sqlite.json")
    ap.add_argument("--db_root", type=str, default="./datasets/data_minidev/dev_databases")
    ap.add_argument("--strategy", type=str, default="egla_beam",
                    choices=["greedy", "beam", "top_k", "top_p", "eg_beam", 'egla_beam'])
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="0 = всі; >0 = перші N")
    ap.add_argument("--save_csv", type=str, default="./outputs/mini_dev_sqlite_eval.csv")
    ap.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"]) # only cpu 
    ap.add_argument("--schema_rows", type=int, default=3, help="Додати N прикладів рядків у prompt (0=без рядків)")
    args = ap.parse_args()
    
    model_path = f'./models/{args.model}'
    save_file_path = f'outputs/mini_dev_sqlite_{args.strategy}_{args.model}.json'
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)


    print(f"📄 Loading: {args.data_json}")
    ds = load_examples(args.data_json)

    print(f"🧠 Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    dtype = torch.float16 if torch.cuda.is_available() and args.device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )
    device = (torch.device("cuda") if (args.device in ["auto", "cuda"] and torch.cuda.is_available())
              else torch.device("cpu"))
    model.to(device).eval()

    results = []

    if not os.path.exists(save_file_path):
        with open(save_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        with open(save_file_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    existing_ids = [r.get("id") for r in results]
    max_id = existing_ids[-1] + 1 if existing_ids else 0

    n_total = len(ds)
    limit = n_total if args.limit in [0, None] else min(args.limit, n_total)

    print(f"▶️ Running from {max_id} to {limit} samples with strategy='{args.strategy}'")

    idx = max_id
    for record in ds[max_id:max_id+limit]:

        question = record.get("question")
        gold_sql = record.get("SQL")
        db_id = record.get("db_id", "")
        difficulty = record.get("difficulty", "")

        db_path = f"{args.db_root}/{db_id}/{db_id}.sqlite"
        schema_text = generate_schema_prompt_sqlite(db_path, num_rows=args.schema_rows if args.schema_rows > 0 else None)
        prompt = build_prompt(tokenizer, question, schema_text)


        start_time = time()
        generated_txt = gen_with_strategy(model, tokenizer, prompt, args.strategy, args.max_new_tokens, db_id=db_id, db_path=db_path)
        exec_time = round(time() - start_time, 2)
        pred_sql = extract_sql_from_text(generated_txt)

        print(f"\nDB: {db_id}\nQ: {question}\nGold: {normalize_sql(gold_sql)}\nPred: {normalize_sql(pred_sql)}")

        results.append({
            "id": idx,
            "db_id": db_id,
            "difficulty": difficulty,
            "question": question,
            "gold_sql": normalize_sql(gold_sql),
            "pred_sql": normalize_sql(pred_sql),
            "exec_time": exec_time,
        })
        idx += 1

        with open(save_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        torch.cuda.empty_cache()
        gc.collect()

    print(f"💾 Saved: {args.save_csv}")

if __name__ == "__main__":
    main()
