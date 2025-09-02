# ---------- Soft-F1 helpers (адаптація логіки BIRD mini_dev) ----------
import sqlite3
from typing import Optional, Iterable

def _normalize_cell(v):
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", "ignore")
        except Exception:
            return str(v)
    return v

def _normalize_rows(rows: Optional[Iterable[tuple]]) -> list[tuple]:
    """Нормалізує значення в клітинках і повертає список кортежів."""
    if not rows:
        return []
    out = []
    for r in rows:
        out.append(tuple(_normalize_cell(x) for x in r))
    return out

def execute_sql_on_db(db_path: str, sql: str) -> Optional[list[tuple]]:
    """Виконує SQL у SQLite і повертає усі рядки (або None у разі помилки)."""
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        con.close()
        return rows
    except sqlite3.Error as e:
        print(f"[SQL ERROR] {e} | Query: {sql}")
        con.close()
        return None

# --- Локальна логіка soft row-match та F1 (узята з mini_dev/evaluation/evaluation_f1.py) ---

def _soft_row_match(pred_row: tuple, gold_row: tuple) -> tuple[float, float, float]:
    """
    Обчислює «м’який» збіг між двома рядками.
    Повертає:
      match_pct (tp частка), pred_only_pct (fp частка), truth_only_pct (fn частка)
    """
    total = len(gold_row)
    matches = 0
    pred_only = 0
    truth_only = 0

    # елементи, що збігаються (без урахування позиції)
    for pv in pred_row:
        if pv in gold_row:
            matches += 1
        else:
            pred_only += 1
    for gv in gold_row:
        if gv not in pred_row:
            truth_only += 1

    match_pct = matches / total if total else 0.0
    pred_only_pct = pred_only / total if total else 0.0
    truth_only_pct = truth_only / total if total else 0.0
    return match_pct, pred_only_pct, truth_only_pct

def _soft_f1_on_rows(pred_rows: list[tuple], gold_rows: list[tuple]) -> float:
    """
    Рахує Soft F1 між списками рядків (після нормалізації).
    Дублікатні рядки прибираємо (set), як у BIRD mini_dev.
    Порядок рядків не важливий — але порівняння йде попарно за індексом
    після приведення до списку (повторює їх реалізацію) :contentReference[oaicite:1]{index=1}.
    """
    # якщо обидва порожні — F1 = 1.0
    if not pred_rows and not gold_rows:
        return 1.0

    pred_set = set(pred_rows) if pred_rows else set()
    gold_set = set(gold_rows)

    pred = list(pred_set)
    gold = list(gold_set)

    match_scores: list[float] = []
    pred_only_scores: list[float] = []
    truth_only_scores: list[float] = []

    # рядки, що існують у gold (порівнюємо з відповідним pred або 0)
    for i, gt_row in enumerate(gold):
        if i >= len(pred):
            match_scores.append(0.0)
            truth_only_scores.append(1.0)
            continue
        pr = pred[i]
        m, po, to = _soft_row_match(pr, gt_row)
        match_scores.append(m)
        pred_only_scores.append(po)
        truth_only_scores.append(to)

    # «зайві» предикт-рядки (яких більше, ніж у gold)
    for _ in range(len(pred) - len(gold)):
        match_scores.append(0.0)
        pred_only_scores.append(1.0)
        truth_only_scores.append(0.0)

    tp = sum(match_scores)
    fp = sum(pred_only_scores)
    fn = sum(truth_only_scores)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

# ---------- Публічна функція «як у тебе»: приймає два SQL і повертає метрику ----------

def soft_f1_equal(db_path: str, pred_sql: str, gold_sql: str) -> float:
    """
    Виконує обидва запити в одній SQLite-БД і повертає Soft F1 у [0..1].
    Повертає 0.0, якщо хоч один запит не виконався.
    """
    rows_p = execute_sql_on_db(db_path, pred_sql)
    rows_g = execute_sql_on_db(db_path, gold_sql)

    if rows_g is None:
        print("[SQL GOLD ERROR]")
        return 0.0
    if rows_p is None:
        return 0.0

    pred_norm = _normalize_rows(rows_p)
    gold_norm = _normalize_rows(rows_g)
    return _soft_f1_on_rows(pred_norm, gold_norm)
