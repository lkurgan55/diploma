"""Evaluation based on executing the predicted and gold SQL queries on the same database and comparing results."""
import sqlite3, concurrent.futures


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

import sqlite3, time

def execute_sql_on_db(db_path: str, sql: str, timeout_s: float = 60.5, ops_step: int = 1000):
    """Executes a SQL query on a SQLite database with a timeout."""
    con = sqlite3.connect(db_path)
    start = time.monotonic()

    def _progress():
        if time.monotonic() - start > timeout_s:
            return 1
        return 0

    con.set_progress_handler(_progress, ops_step)

    try:
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return rows
    except (sqlite3.OperationalError, sqlite3.ProgrammingError) as e:
        msg = str(e).lower()
        if "interrupted" in msg:
            print(f"[TIMEOUT] >{timeout_s}s: {sql}")
            return None
        print(f"[SQL GOLD ERROR] {e} | Query: {sql}")
        return None
    finally:
        try:
            con.set_progress_handler(None, 0)
        except Exception:
            pass
        con.close()


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

if __name__ == "__main__":

    db_path = f"./datasets/data_minidev/dev_databases/financial/financial.sqlite"
    gold_sql = "select t1.frequency, t2.k_symbol from account as t1 inner join (select account_id, k_symbol, sum(amount) as total_amount from `order` group by account_id, k_symbol) as t2 on t1.account_id = t2.account_id where t1.account_id = 3 and t2.total_amount = 3539"
    pred_sql = "select count(*) from trans where account_id = 3 and type = 'predpis' ;select sum(amount) from trans where account_id = 3 and type = 'odebranie' and amount = 3539"
    print(execution_equal(db_path, pred_sql, gold_sql))