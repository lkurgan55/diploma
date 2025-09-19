# simple_sql_validator.py
import re
import sqlite3
from contextlib import closing
from typing import Optional, Set, Tuple

class SQLValidator:
    """
    Мінімальний EG-валідатор для перевірки: чи існують усі таблиці з FROM/JOIN у БД (SQLite).
    Не торкається колонок/типів/операторів — лише існування таблиць.
    """

    def __init__(self, db_path: Optional[str]):
        self.db_path = db_path
        self._schema_loaded = False
        self._tables: dict[str, set[str]] = {}   # NEW: table -> set(columns)
        self._table_names: set[str] = set()      # NEW: швидкий доступ до назв

    # ---------- schema loading ----------
    def _maybe_load_schema(self) -> None:
        if self._schema_loaded or not self.db_path:
            return
        self._schema_loaded = True
        tables: dict[str, set[str]] = {}
        try:
            with closing(sqlite3.connect(self.db_path)) as con, closing(con.cursor()) as cur:
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                for (tname,) in cur.fetchall():
                    t = tname.lower()
                    cur.execute(f"PRAGMA table_info('{tname}')")
                    cols = {row[1].lower() for row in cur.fetchall()}
                    tables[t] = cols
        except Exception:
            tables = {}
        self._tables = tables
        self._table_names = set(tables.keys())


    # ---------- light SQL parsing ----------
    def _extract_cte_names(self, sql: str) -> Set[str]:
        """
        WITH c1 AS (...), c2 AS (...) SELECT ...
        Повертаємо множину імен CTE: {"c1", "c2"} (lowercase).
        Дуже проста евристика — достатня для EG-фільтра.
        """
        s = re.sub(r"\s+", " ", sql).strip()
        if not re.match(r"(?is)^with\b", s):
            return set()
        # шматок між WITH і першим SELECT після нього
        m = re.match(r"(?is)^with\s+(.*)\bselect\b", s)
        tail = m.group(1) if m else s[4:]
        return {nm.lower() for nm in re.findall(r"(?is)\b([a-zA-Z_]\w*)\s+as\s*\(", tail)}

    def _extract_tables_and_derived(self, sql: str, *, closed_only: bool = False) -> Tuple[Set[str], Set[str]]:
        """
        Якщо closed_only=True, вважаємо ім'я таблиці "закритим" лише коли після нього вже є пробіл або ';'.
        Це потрібно для інкрементального декодування (щоб 'transactions' не вважати завершеним,
        коли наступний токен ще має дописати '_1k').
        """
        s = re.sub(r"\s+", " ", sql, flags=re.I)

        phys_tables: Set[str] = set()
        derived_aliases: Set[str] = set()

        # 1) derived: FROM (SELECT ...) alias / JOIN (SELECT ...) alias
        for m in re.finditer(r"\b(from|join)\s*\(\s*select\b.*?\)\s*([a-zA-Z_]\w*)", s, flags=re.I | re.S):
            derived_aliases.add(m.group(2).lower())

        # 2) фізичні імена після FROM/JOIN
        if closed_only:
            # ім'я вважається завершеним лише якщо ДАЛІ вже стоїть пробіл або ';'
            #   приклад: "... FROM transactions <space> ..."  або  "... FROM races; ..."
            pat = r"\b(from|join)\s+([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)?)(?=[\s;])"
        else:
            # стандартно — будь-яка word boundary після імені
            pat = r"\b(from|join)\s+([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)?)\b"

        for m in re.finditer(pat, s, flags=re.I):
            raw = m.group(2)
            base = raw.split(".")[-1].lower()
            phys_tables.add(base)

        return phys_tables, derived_aliases

    def _extract_tables_and_aliases(self, sql: str, *, closed_only: bool = False) -> tuple[set[str], dict, set]:
        s = re.sub(r"\s+", " ", sql, flags=re.I)
        tabs: set[str] = set()
        alias_map: dict[str, str] = {}
        derived_aliases: set[str] = set()
        ctes = self._extract_cte_names(sql)

        # derived: FROM (SELECT ...) alias
        for m in re.finditer(r"\b(from|join)\s*\(\s*select\b.*?\)\s*([a-zA-Z_]\w*)", s, flags=re.I | re.S):
            alias = m.group(2).lower()
            alias_map[alias] = alias
            derived_aliases.add(alias)

        # фізичні: FROM/JOIN <name> [AS] <alias>
        if closed_only:
            pat = r"\b(from|join)\s+([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)?)(?=[\s;])(?:\s+(?:as\s+)?([a-zA-Z_]\w*))?"
        else:
            pat = r"\b(from|join)\s+([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)?)\b(?:\s+(?:as\s+)?([a-zA-Z_]\w*))?"

        for m in re.finditer(pat, s, flags=re.I):
            raw = m.group(2)
            base = raw.split(".")[-1].lower()
            alias = (m.group(3) or "").lower() if m.lastindex and m.group(3) else None

            if base not in ctes:
                tabs.add(base)
            if alias:
                alias_map[alias] = base
            else:
                alias_map.setdefault(base, base)

        # CTE як джерела
        for c in ctes:
            alias_map.setdefault(c, c)

        return tabs, alias_map, derived_aliases


    # ---------- public API ----------
    def tables_exist(self, sql: str, *, closed_only: bool = True) -> bool:
        self._maybe_load_schema()
        if not self._table_names:
            return True

        ctes = self._extract_cte_names(sql)
        phys_tables, _ = self._extract_tables_and_derived(sql, closed_only=closed_only)
        phys_tables = {t for t in phys_tables if t not in ctes}

        for t in phys_tables:
            if t not in self._table_names:
                return False
        return True

    def columns_exist(self, sql: str, *, closed_only: bool = True) -> bool:
        self._maybe_load_schema()
        if not self._tables:
            return True

        ctes = self._extract_cte_names(sql)
        tabs, alias_map, derived_aliases = self._extract_tables_and_aliases(sql, closed_only=closed_only)

           # --- 1) Кваліфіковані alias.col / table.col (деферимо, якщо колона ще не "закрита") ---
        s = re.sub(r"\s+", " ", sql)

        qual_pat = r"([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)"
        qualified_seconds = set()

        for m in re.finditer(qual_pat, s):
            alias = m.group(1).lower()
            col   = m.group(2).lower()
            qualified_seconds.add(col)

            # перевірка "закритості" токена колонки:
            # пропускаємо пробіли після збігу і дивимось на наступний фактичний символ
            j = m.end()
            while j < len(s) and s[j].isspace():
                j += 1

            # якщо дійшли до кінця або далі лише пробіли — колона ще НЕ закрита -> деферимо
            if j >= len(s):
                continue

            # дозволені "закривачі" колонки (після яких можна перевіряти): кома/дужка/крапка з комою,
            # оператори порівняння/арифметики, ключові роздільники
            closers = {",", ")", ";", ".", "=", "<", ">", "+", "-", "*", "/"}
            # також вважаємо закритою, якщо далі починається ключове слово (on, where, group, order, having, from)
            next_is_kw = False
            if s[j].isalpha():
                nxt = s[j:].lower()
                for kw in (" on", " where", " group", " order", " having", " from"):
                    if nxt.startswith(kw):
                        next_is_kw = True
                        break

            if (s[j] not in closers) and (not next_is_kw):
                # між alias.col і наступним токеном ще можуть дописуватись символи колонки -> деферимо
                continue

            # якщо alias ще не оголошено у FROM/JOIN/CTE/derived — деферимо
            alias_known = (alias in alias_map) or (alias in ctes) or (alias in derived_aliases)
            if not alias_known:
                continue

            src = alias_map.get(alias, alias)

            # CTE/derived не звіряємо по фізичній схемі
            if (src in ctes) or (src in derived_aliases):
                continue

            # джерело має бути фізичною таблицею
            if src not in self._tables:
                continue  # деферимо

            # власне перевірка колонки
            if col not in self._tables[src]:
                return False


        # --- некваліфіковані — лише коли рівно одна фізична таблиця ---
        if len(tabs) == 1:
            (only_tab,) = tuple(tabs)
            m = re.search(r"(?is)\bselect\b(.*?)\bfrom\b", s)
            if m:
                chunk = m.group(1)
                # прибираємо літерали, AS alias, імена функцій
                chunk = re.sub(r"'([^']|'')*'", " ", chunk)
                chunk = re.sub(r"\b(as)\s+[a-zA-Z_]\w*", " ", chunk, flags=re.I)
                chunk = re.sub(r"[a-zA-Z_]\w*\s*\(", "(", chunk)
                # прибрати кваліфіковані x.y (залишаємо лише y)
                chunk2 = re.sub(r"([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)", r"\2", chunk)

                toks = re.findall(r"[a-zA-Z_]\w*", chunk2)
                KW = {"select","distinct","case","when","then","else","end",
                    "count","sum","avg","min","max","cast","as","real"}
                for tok in toks:
                    tl = tok.lower()
                    if tl in KW or tl.isnumeric():
                        continue
                    if tl in qualified_seconds:  # дуже важливо: не дублюємо з t2.currency
                        continue
                    if tl not in self._tables.get(only_tab, set()):
                        return False
        return True





validator = SQLValidator(db_path="datasets/data_minidev/dev_databases/debit_card_specializing/debit_card_specializing.sqlite")

sql = """select customerid from yearmonth where date like '2012%' and segment = 'lam' group by customerid order by sum(consumption) asc limit 1"""

print("tables ok:", validator.tables_exist(sql))
print("columns ok:", validator.columns_exist(sql))
