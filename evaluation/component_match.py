"""Component-Match metric (exact, AST-based via sqlglot)."""
import sqlglot
from sqlglot import exp

def _norm_ident(s: str) -> str:
    """Normalize SQL identifiers for comparison."""
    return s.strip().lower()

def _col_name(e: exp.Expression) -> str:
    """Get normalized column name from expression."""
    if isinstance(e, exp.Column):
        return _norm_ident(e.alias_or_name)
    return _norm_ident(e.sql(dialect="sqlite"))

def _lit_value(e: exp.Expression) -> str:
    """Get normalized literal value from expression."""
    if isinstance(e, exp.Literal):
        return _norm_ident(e.this)
    return _norm_ident(e.sql(dialect="sqlite"))

def _strip_alias(e: exp.Expression) -> exp.Expression:
    """Remove alias from expression if present."""
    return e.this if isinstance(e, exp.Alias) else e

def extract_tables(tree: exp.Expression) -> set[str]:
    """Extract table names from the SQL AST."""
    out = set()
    for t in tree.find_all(exp.Table):
        out.add(_norm_ident(t.name))
    return out

def extract_select_items(tree: exp.Expression) -> set[str]:
    """Extract selected items (columns, functions) from the SQL AST."""
    items: set[str] = set()
    select = tree.find(exp.Select)
    if not select:
        return items

    for proj in select.expressions:
        proj = _strip_alias(proj)
        if isinstance(proj, exp.Func):
            fn = _norm_ident(proj.name)
            args = [_col_name(a) for a in proj.args.get("expressions", [])] or \
                   ([_col_name(proj.this)] if proj.this is not None else [])
            items.add(f"{fn}(" + ",".join(args) + ")")
        else:
            items.add(_col_name(proj))
    return items

def _flatten_conj(e: exp.Expression) -> list[exp.Expression]:
    """Flatten conjunctions (AND) into a list of expressions."""
    if isinstance(e, exp.And):
        return _flatten_conj(e.left) + _flatten_conj(e.right)
    return [e]

def _normalize_predicate(p: exp.Expression) -> str | None:
    """Normalize a predicate expression into a string representation."""
    cmp_ops = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
    if isinstance(p, cmp_ops):
        L, R = p.left, p.right
        if isinstance(L, exp.Column) and isinstance(R, exp.Column):
            return None

        if isinstance(R, exp.Subquery):
            sub = R.this
            t = list(sub.find_all(exp.Table))
            w = sub.find(exp.Where)
            if t and w:
                tbl = _norm_ident(t[-1].name)
                simple_preds = [_normalize_predicate(x) for x in _flatten_conj(w.this)]
                for sp in simple_preds:
                    if sp and sp.startswith("name="):
                        return f"{tbl}.{sp}"

            return _norm_ident(p.sql(dialect="sqlite"))

        op = _norm_ident(p.key)
        left = _col_name(L)
        right = _lit_value(R)
        return f"{left}{op}{right}"

    if isinstance(p, exp.Between):
        col = _col_name(p.this)
        lo = _lit_value(p.args["low"])
        hi = _lit_value(p.args["high"])
        return f"{col}between{lo}..{hi}"
    if isinstance(p, exp.Like):
        return f"{_col_name(p.this)}like{_lit_value(p.expression)}"
    if isinstance(p, exp.In):
        col = _col_name(p.this)
        vals = ",".join(_lit_value(v) for v in p.expressions)
        return f"{col}in({vals})"
    if isinstance(p, exp.Is):
        return f"{_col_name(p.this)}is{_norm_ident(p.expression.sql(dialect='sqlite'))}"
    return _norm_ident(p.sql(dialect="sqlite"))

def extract_where_filters(tree: exp.Expression) -> set[str]:
    """Extract normalized WHERE clause filters from the SQL AST."""
    fs: set[str] = set()
    where = tree.find(exp.Where)
    if not where:
        return fs
    for p in _flatten_conj(where.this):
        norm = _normalize_predicate(p)
        if norm:
            fs.add(norm)
    return fs

def extract_group_by(tree: exp.Expression) -> set[str]:
    """Extract GROUP BY columns from the SQL AST."""
    out = set()
    grp = tree.find(exp.Group)
    if not grp:
        return out
    for g in grp.expressions:
        out.add(_col_name(g))
    return out

def extract_order_by(tree: exp.Expression) -> set[str]:
    """Extract ORDER BY columns and their directions from the SQL AST."""
    out = set()
    ob = tree.find(exp.Order)
    if not ob:
        return out
    for o in ob.expressions:
        col = _col_name(_strip_alias(o.this))
        direction = "desc" if o.args.get("desc") else "asc"
        out.add(f"{col}:{direction}")
    return out

def extract_limit(tree: exp.Expression) -> set[str]:
    """Extract LIMIT value from the SQL AST."""
    lim = tree.find(exp.Limit)
    if not lim:
        return set()
    return { _norm_ident(lim.expression.sql(dialect="sqlite")) }

def component_sets(sql: str, dialect: str = "sqlite") -> dict[str, set[str]]:
    """Extract SQL components as sets using sqlglot."""
    tree = sqlglot.parse_one(sql, read=dialect)
    return {
        "tables": extract_tables(tree),
        "select": extract_select_items(tree),
        "where":  extract_where_filters(tree),
        "group":  extract_group_by(tree),
        "order":  extract_order_by(tree),
        "limit":  extract_limit(tree),
    }

def component_match_exact(pred_sql: str, gold_sql: str, dialect: str = "sqlite") -> float:
    """Component-Match (exact, AST-based via sqlglot)."""
    try:
        pred = component_sets(pred_sql, dialect)
        gold = component_sets(gold_sql, dialect)
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError):
        return 0.0

    keys = sorted(set(pred.keys()) | set(gold.keys()))
    scores = []
    for k in keys:
        if pred[k] or gold[k]:
            scores.append(1.0 if pred[k] == gold[k] else 0.0)
    return sum(scores) / len(scores) if scores else 1.0  # якщо обидва порожні — умовно 1


if __name__ == "__main__":
    # Example usage
    gold_query = "select cast(sum(t2.home_team_goal) as real) / count(t2.id) from country as t1 inner join match as t2 on t1.id = t2.country_id where t1.name = 'poland' and t2.season = '2010/2011'"
    pred_query = "select avg(home_team_goal) from match where country_id = (select id from country where name = 'poland') and season = '2010/2011'"

    score = component_match_exact(pred_query, gold_query)
    print(f"Component Match Score: {score:.2f}")