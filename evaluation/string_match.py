import re
import sqlparse

def _strip_sql_comments(sql: str) -> str:
    """Remove SQL comments from the query string."""
    s = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    return s

def _collapse_ws(s: str) -> str:
    """Collapse multiple whitespace characters into a single space and trim."""
    return re.sub(r"\s+", " ", s).strip()

def normalize_sql(sql: str) -> str:
    """Simple SQL normalization for string matching."""
    s = str(sql).strip()
    s = _strip_sql_comments(s)
    s = re.sub(r";+\s*$", "", s)

    try:
        s = sqlparse.format(
            s,
            keyword_case="lower",
            identifier_case=None,
            strip_comments=True,
            reindent=False
        )
    except Exception:
        pass
    return _collapse_ws(s)

def string_match_equal(pred_sql: str, gold_sql: str) -> bool:
    """Compares two SQL queries by string match."""
    return normalize_sql(pred_sql) == normalize_sql(gold_sql)
