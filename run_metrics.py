"""
Compute Execution Accuracy, String Match Accuracy, and Component Match Accuracy
with optional SQL "fix" (normalization via sqlglot) for each record.
"""

from evaluation.evaluation_ex import execution_equal
from evaluation.string_match import string_match_equal
from evaluation.component_match import component_match_exact

import argparse
import json

db_root = "./datasets/data_minidev/dev_databases"

from sqlglot import parse_one, diff
from sqlglot.optimizer import optimize

def _count_nodes(expr) -> int:
    n = 1
    for _ in expr.walk():
        n += 1
    return n

def ast_similarity_score(pred_sql: str, gold_sql: str, dialect: str = "sqlite") -> float:
    """
    Повертає число у [0..1]: 1 = дуже схоже, 0 = зовсім інше.
    Без жодних деталізованих оцінок/розкладок.
    """
    try:
        gold_ast = optimize(parse_one(gold_sql, read=dialect), dialect=dialect)
        pred_ast = optimize(parse_one(pred_sql, read=dialect), dialect=dialect)
    except Exception:
        return 0.0

    changes = list(diff(pred_ast, gold_ast, delta_only=True))
    gold_size = max(1, _count_nodes(gold_ast))
    # нормована “вартість” = частка змін від розміру gold
    norm_cost = min(1.0, len(changes) / gold_size)
    return 1.0 - norm_cost

def main():
    ap = argparse.ArgumentParser(description="Compute metrics with SQL fix (sqlglot).")
    ap.add_argument(
        "--json_path",
        default="outputs/mini_dev_sqlite_eg_beam_sqlglot_qwen2.5-3B-Instruct.json",
        help="Вхідний JSON із записами (list of dicts).",
    )
    args = ap.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data: list[dict[str]] = json.load(f)

    total = 0
    passed_ex = 0
    passed_sm = 0
    passed_cm = 0
    passed_ast = 0
    total_time = 0.0

    new_records = []
    for record in data[:]:
        rid = record.get("id")
        db_id = record.get("db_id")
        gold_sql = record.get("gold_sql")
        pred_sql = record.get("pred_sql")
        exec_time = record.get("exec_time", 0.0)
        db_path = f"{db_root}/{db_id}/{db_id}.sqlite"

        ex = execution_equal(db_path, pred_sql, gold_sql)
        sm = string_match_equal(pred_sql, gold_sql)
        cm = component_match_exact(pred_sql, gold_sql)
        ast = ast_similarity_score(pred_sql, gold_sql)

        record["ex"] = ex
        record["sm"] = sm
        record["cm"] = cm
        record["ast"] = ast

        total += 1
        passed_ex += ex
        passed_sm += sm
        passed_cm += cm
        passed_ast += ast
        total_time += float(exec_time or 0.0)

        print(f"Record ID: {rid} | DB: {db_id} | EX: {ex} | SM: {sm} | CM: {cm} | AST: {ast}")

        new_records.append(record)

    ex_acc = (passed_ex / total) if total else 0.0
    sm_acc = (passed_sm / total) if total else 0.0
    cm_acc = (passed_cm / total) if total else 0.0
    ast_acc = (passed_ast / total) if total else 0.0
    avg_time = (total_time / total) if total else 0.0

    out_path = args.json_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_records, f, ensure_ascii=False, indent=2)

    print("\n--- SUMMARY ---")
    print(f"Execution Accuracy: {passed_ex}/{total} = {ex_acc:.4f}")
    print(f"String Match Accuracy: {passed_sm}/{total} = {sm_acc:.4f}")
    print(f"Component Match Accuracy: {passed_cm}/{total} = {cm_acc:.4f}")
    print(f"AST Similarity Score: {passed_ast}/{total} = {ast_acc:.4f}")
    print(f"AVG Generation Time: {total_time}/{total} = {avg_time:.4f}")


if __name__ == "__main__":
    main()
