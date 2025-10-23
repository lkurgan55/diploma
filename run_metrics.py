"""
Compute Execution Accuracy, String Match Accuracy, and Component Match Accuracy
with optional SQL "fix" (normalization via sqlglot) for each record.
"""

from evaluation.evaluation_ex import execution_equal
from evaluation.string_match import string_match_equal
from evaluation.component_match import component_match_exact

import argparse
import json
from typing import Dict, Any, Tuple
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.normalize import normalize
from sqlglot.optimizer.simplify import simplify
# --- SQLGlot-based fixer ---
from sqlglot import transpile, parse_one
from sqlglot.optimizer import optimize
from contextlib import closing
import sqlite3
from sqlglot.errors import SchemaError, ParseError, TokenError, OptimizeError

db_root = "./datasets/data_minidev/dev_databases"

LIGHT_RULES = (qualify, normalize, simplify)  # за бажанням: + (canonicalize,)

def _load_sqlite_schema(db_path: str, two_level: bool = False) -> dict:
    """
    Повертає plain-dict схему з lower-case іменами.
    Якщо two_level=True -> {"main": {"table": {"col": "ANY"}}}
    Інакше -> {"table": {"col": "ANY"}}
    """
    import sqlite3
    from contextlib import closing

    mapping: dict[str, dict] = {}
    with closing(sqlite3.connect(db_path)) as conn:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT name, type FROM sqlite_master "
            "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%';"
        ).fetchall()

        for name, _ in rows:
            t = name.lower()
            cols = cur.execute(f"PRAGMA table_info('{name}')").fetchall()
            cols_map = {row[1].lower(): "ANY" for row in cols}

            if two_level:
                mapping.setdefault("main", {})[t] = cols_map
            else:
                mapping[t] = cols_map
    return mapping

def sql_fix(sql: str, *, dialect: str = "sqlite", db_path: str | None = None,
            ):
    try:
        fixed = transpile(sql, read=dialect, write=dialect, pretty=False)[0]
        expr = parse_one(fixed, read=dialect)
    except (ParseError, TokenError):
        return sql

    if db_path:
        try:
            schema = _load_sqlite_schema(db_path)
            expr = qualify(expr, schema=schema, dialect=dialect)
        except (SchemaError, OptimizeError):
            pass

    expr = normalize(expr)
    expr = simplify(expr, dialect=dialect)
    return expr.sql(dialect=dialect)

def main():
    ap = argparse.ArgumentParser(description="Compute metrics with SQL fix (sqlglot).")
    ap.add_argument(
        "--json_path",
        default="outputs/mini_dev_sqlite_eg_beam_qwen2.5-3B-Instruct.json",
        help="Вхідний JSON із записами (list of dicts).",
    )
    args = ap.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    total = 0
    passed_ex = 0
    passed_sm = 0
    passed_cm = 0
    total_time = 0.0

    new_records = []
    for record in data[:]:
        rid = record.get("id")
        db_id = record.get("db_id")
        gold_sql = record.get("gold_sql")
        pred_sql = record.get("pred_sql")
        exec_time = record.get("exec_time", 0.0)

        pred_sql = sql_fix(
            pred_sql, dialect='sqlite', db_path=f"{db_root}/{db_id}/{db_id}.sqlite"
        )
        
        db_path = f"{db_root}/{db_id}/{db_id}.sqlite"

        ex = execution_equal(db_path, pred_sql, gold_sql)
        sm = string_match_equal(pred_sql, gold_sql)
        cm = component_match_exact(pred_sql, gold_sql)

        record["ex"] = ex
        record["sm"] = sm
        record["cm"] = cm
        record["fixed_pred_sql"] = pred_sql.lower()

        total += 1
        passed_ex += ex
        passed_sm += sm
        passed_cm += cm
        total_time += float(exec_time or 0.0)

        print(f"Record ID: {rid} | DB: {db_id} | EX: {ex} | SM: {sm} | CM: {cm}")

        new_records.append(record)

    ex_acc = (passed_ex / total) if total else 0.0
    sm_acc = (passed_sm / total) if total else 0.0
    cm_acc = (passed_cm / total) if total else 0.0
    avg_time = (total_time / total) if total else 0.0

    out_path = args.json_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_records, f, ensure_ascii=False, indent=2)

    print("\n--- SUMMARY ---")
    print(f"Execution Accuracy: {passed_ex}/{total} = {ex_acc:.4f}")
    print(f"String Match Accuracy: {passed_sm}/{total} = {sm_acc:.4f}")
    print(f"Component Match Accuracy: {passed_cm}/{total} = {cm_acc:.4f}")
    print(f"AVG Generation Time: {total_time}/{total} = {avg_time:.4f}")


if __name__ == "__main__":
    main()
