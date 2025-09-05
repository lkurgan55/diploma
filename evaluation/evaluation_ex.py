"""Evaluation based on executing the predicted and gold SQL queries on the same database and comparing results."""
import sqlite3

def _normalize_cell(v):
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", "ignore")
        except Exception:
            return str(v)
    return v

def _rows_to_set(rows: list[tuple]) -> set:
    """Converts a list of rows to a set of normalized tuples for comparison."""
    norm = []
    for r in rows:
        norm.append(tuple(_normalize_cell(x) for x in r))
    return set(norm)

def execute_sql_on_db(db_path: str, sql: str) -> tuple[bool, list[tuple]]:
    """Executes a SQL query on the given SQLite database and returns the results."""
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

def execution_equal(db_path: str, pred_sql: str, gold_sql: str) -> int:
    """Compares the results of two SQL queries executed on the same database."""
    rows_p = execute_sql_on_db(db_path, pred_sql)
    rows_g = execute_sql_on_db(db_path, gold_sql)

    if rows_g is None:
        print(f"[SQL GOLD ERROR]")
        return False

    if rows_p is None:
        print(f"[SQL PRED ERROR]")
        return False

    return _rows_to_set(rows_p) == _rows_to_set(rows_g)
