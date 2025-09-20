# simple_sql_validator.py
import re
import sqlite3
from contextlib import closing
from typing import Optional, Set, Tuple

class SQLValidator:
    """
    Мінімальний EG-валідатор...
    """

    def __init__(self, db_path: Optional[str]):
        self.db_path = db_path
        self._schema_loaded = False
        self._tables: dict[str, set[str]] = {}
        self._table_names: set[str] = set()
        
    def syntax_ok(self, sql: str, *, prefix_ok: bool = True) -> bool:
        import re, sqlite3
        from contextlib import closing

        def _normalize_sql(s: str) -> str:
            s = s.strip().lstrip("\ufeff")
            s = s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ")
            s = re.sub(r"[\u00A0\u200B\u200E\u200F]", " ", s)
            s = re.sub(r"\s+", " ", s)
            return s

        def _outside_quotes_scan(s: str):
            """Ітеруємо символи поза лапками, повертаємо (chars, has_forbidden, semi_count, paren_depth, paren_below_zero)"""
            in_s = in_d = False
            semi = 0
            depth = 0
            below_zero = False
            forbidden = False
            for i, ch in enumerate(s):
                if ch == "'" and not in_d:
                    # '' escape
                    if in_s and i + 1 < len(s) and s[i+1] == "'":
                        continue
                    in_s = not in_s
                elif ch == '"' and not in_s:
                    in_d = not in_d
                elif not in_s and not in_d:
                    # рахунок дужок
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                        if depth < 0:
                            below_zero = True
                    # семікрапки
                    if ch == ';':
                        semi += 1
                    # заборонені символи (мін. білий список)
                    if not re.match(r"[A-Za-z0-9_\s\.,\*\(\)=<>!\+\-/%']", ch):
                        forbidden = True
            return forbidden, semi, depth, below_zero

        s = _normalize_sql(sql)
        if not s:
            return True if prefix_ok else False

        # 1) префікс-фільтри «очевидного сміття»
        forbidden, semi, depth, below_zero = _outside_quotes_scan(s)
        if prefix_ok:
            # будь-який ';' у префіксі — не ок
            if semi > 0:
                return False
            # символи на кшталт & | \ ` тощо — не ок
            if forbidden:
                return False
            # більше закривальних, ніж відкривальних дужок — не ок (урізані '(' дозволяємо)
            if below_zero:
                return False

        # 2) урізаний SELECT — дозволяємо у префіксі
        if re.match(r"(?is)^\s*select\s*;?\s*$", s):
            return True if prefix_ok else False

        # 3) у префіксі не перевіряємо завершеність/EXPLAIN, просто пройшли базові санітарні умови
        if prefix_ok:
            # дозволимо також SELECT <expr> без FROM як префікс
            if re.match(r"(?is)^\s*select\s+.+$", s) and not re.search(r"(?i)\bfrom\b", s):
                return True
            # інакше — просто ок (умови 1 вже відсіяли «жесть»)
            return True

        # --- нижче суворий режим для фінального рядка ---
        # одна інструкція
        if semi > 1:
            return False
        # має бути щось після SELECT
        m_sel = re.search(r"(?i)\bselect\b(.*)$", s)
        if not m_sel or not (m_sel.group(1) or "").strip():
            return False
        # завершеність + EXPLAIN
        stmt = s if s.rstrip().endswith(";") else s + ";"
        try:
            if not sqlite3.complete_statement(stmt):
                return False
        except Exception:
            pass
        try:
            with closing(sqlite3.connect(":memory:")) as con, closing(con.cursor()) as cur:
                cur.execute("EXPLAIN " + s)
        except sqlite3.Error as e:
            msg = str(e).lower()
            if ("syntax error" in msg or "unrecognized token" in msg or
                "incomplete input" in msg or "misuse" in msg or
                ("parse" in msg and "error" in msg)):
                return False
            return True
        return True
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

sql = """SELECT SUM_CONSUMPTION AS Difference) AS diff\n(SELECT T2 AS CustomerID, SUM CustomerID AS CustomerID, Consumption) AS Consumption\nFROM yearmonth \n         customers)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))&;"""

print("syntax ok:", validator.syntax_ok(sql))
print("tables ok:", validator.tables_exist(sql))
print("columns ok:", validator.columns_exist(sql))
