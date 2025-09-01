from evaluation.evaluation_ex import execution_equal
from evaluation.string_match import string_match_equal

import argparse, json

db_root = './datasets/data_minidev/dev_databases'
# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Compute Execution Accuracy for JSON predictions.")
    ap.add_argument("--json_path", default='outputs/mini_dev_sqlite_greedy.json', help="Вхідний JSON із записами (list of dicts).")
    ap.add_argument("--save_json", default='test_metrics', help="Куди зберегти оновлений JSON з полями ex/err_*")
    args = ap.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data: list[dict[str, int | str]] = json.load(f)

    total, passed_ex, passed_sm = 0, 0, 0
    for record in data[:]:
        id = record.get("id")
        db_id    = record.get("db_id")
        gold_sql = record.get("gold_sql")
        pred_sql = record.get("pred_sql")

        db_path = f"{db_root}/{db_id}/{db_id}.sqlite"

        ex = execution_equal(db_path, pred_sql, gold_sql)
        sm = string_match_equal(pred_sql, gold_sql)

        record["ex"] = ex
        record["sm"] = sm

        total += 1
        print(f"Record ID: {id} | DB: {db_id} | Execution Equal: {ex} | String Match: {sm}")
        passed_ex += ex
        passed_sm += sm

    ex_acc = (passed_ex / total) if total else 0.0
    sm_acc = (passed_sm / total) if total else 0.0
    print(f"Execution Accuracy: {passed_ex}/{total} = {ex_acc:.4f}")
    print(f"String Match Accuracy: {passed_sm}/{total} = {sm_acc:.4f}")

if __name__ == "__main__":
    main()