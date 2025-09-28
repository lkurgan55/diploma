"""Compute Execution Accuracy, String Match Accuracy, and Component Match Accuracy for JSON predictions."""

from evaluation.evaluation_ex import execution_equal
from evaluation.string_match import string_match_equal
from evaluation.component_match import component_match_exact

import argparse, json

db_root = './datasets/data_minidev/dev_databases'


def main():
    ap = argparse.ArgumentParser(description="Compute Execution Accuracy for JSON predictions.")
    ap.add_argument("--json_path", default='outputs/mini_dev_sqlite_egla_beam_qwen2.5-3B-Instruct.json', help="Вхідний JSON із записами (list of dicts).")
    ap.add_argument("--save_json", default='test_metrics', help="Куди зберегти оновлений JSON з полями ex/err_*")
    args = ap.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data: list[dict[str, int | str]] = json.load(f)

    total, passed_ex, passed_sm, passed_cm = 0, 0, 0, 0
    new_records = []
    for record in data[:]:
        id = record.get("id")
        db_id    = record.get("db_id")
        gold_sql = record.get("gold_sql")
        pred_sql = record.get("pred_sql")

        db_path = f"{db_root}/{db_id}/{db_id}.sqlite"

        ex = execution_equal(db_path, pred_sql, gold_sql)
        sm = string_match_equal(pred_sql, gold_sql)
        cm = component_match_exact(pred_sql, gold_sql)

        record["ex"] = ex
        record["sm"] = sm
        record["cm"] = cm

        total += 1
        print(f"Record ID: {id} | DB: {db_id} | Execution Equal: {ex} | String Match: {sm} | Component Match: {cm}")
        passed_ex += ex
        passed_sm += sm
        passed_cm += cm

        new_records.append(record)

    ex_acc = (passed_ex / total) if total else 0.0
    sm_acc = (passed_sm / total) if total else 0.0
    cm_acc = (passed_cm / total) if total else 0.0

    with open(args.json_path, "w", encoding="utf-8") as f:
        json.dump(new_records, f, ensure_ascii=False, indent=2)

    print(f"\nExecution Accuracy: {passed_ex}/{total} = {ex_acc:.4f}")
    print(f"String Match Accuracy: {passed_sm}/{total} = {sm_acc:.4f}")
    print(f"Component Match Accuracy: {passed_cm}/{total} = {cm_acc:.4f}")

if __name__ == "__main__":
    main()