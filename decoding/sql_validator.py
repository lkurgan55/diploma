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
        
    def syntax_ok(self, sql: str) -> bool:
        """
        Інкрементальна перевірка:
        True  -> синтаксично можливий префікс (можна продовжувати/залишати гілку).
        False -> безнадійна помилка, гілку слід відкинути.

        Ми вважаємо BAD лише такі випадки:
        - None / порожній рядок із non-whitespace символами відсутній? -> допустимо як префікс (True)
        - негативна глибина дужок (зайва ')')
        - наявність ';' не наприкінці (кілька інструкцій)
        Все інше (відкриті лапки/дужки, «хвости» на ключових словах/операторах) -> True (ще можна дописати).
        """
        if sql is None:
            return False  # зовсім нічого перевіряти

        s = str(sql)
        if not s.strip():
            return True  # пуста/пробільна префікс-рядок: ок для продовження

        # BAD #1: ';' у середині -> кілька інструкцій/розрив
        if ";" in s[:-1]:
            return False

        # Прохід по символах для виявлення ЗАЙВОЇ закриваючої дужки ')'
        depth = 0
        in_single = False  # '
        in_double = False  # "
        in_backt  = False  # `

        i, n = 0, len(s)

        def at(j: int) -> str:
            return s[j] if 0 <= j < n else ""

        while i < n:
            ch = s[i]

            # керуємо лапками (усередині лапок дужки не рахуємо)
            if not (in_double or in_backt):
                if ch == "'":
                    if in_single:
                        if at(i+1) == "'":  # екранована '
                            i += 2
                            continue
                        else:
                            in_single = False
                            i += 1
                            continue
                    else:
                        in_single = True
                        i += 1
                        continue

            if not (in_single or in_backt):
                if ch == '"':
                    if in_double:
                        if at(i+1) == '"':  # екранована "
                            i += 2
                            continue
                        else:
                            in_double = False
                            i += 1
                            continue
                    else:
                        in_double = True
                        i += 1
                        continue

            if not (in_single or in_double):
                if ch == '`':
                    in_backt = not in_backt
                    i += 1
                    continue

            # поза лапками — рахуємо дужки
            if not (in_single or in_double or in_backt):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth < 0:
                        # BAD #2: зайва закриваюча дужка — це не виправити додаванням токенів
                        return False

            i += 1

        # Всі інші стани (відкриті лапки, незакриті дужки, хвости на ключових словах/операторах) — це валідний префікс:
        # їх можна завершити подальшою генерацією, отже True.
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

    def _from_clause_open(self, sql: str) -> bool:
        """
        True, якщо секція FROM ще 'очікує' продовження:
        - закінчується на кому: FROM t1,
        - закінчується на JOIN / JOIN <name> [AS alias] без ON/USING,
        - є ON/USING, але умова ще не завершена (незбалансовані дужки / обірвана справа),
        - після FROM стоїть лише ідентифікатор (таблиця/alias) без роздільника.
        False — коли JOIN-умова завершена або бачимо перехід до WHERE/GROUP/ORDER/HAVING/LIMIT/OFFSET/;.
        """
        s = re.sub(r"\s+", " ", sql, flags=re.I)
        m = re.search(r"(?is)\bfrom\b(.*)$", s)
        if not m:
            return False

        tail = m.group(1)
        # обрізати до кінця секції FROM
        frag = re.split(r"(?is)\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\blimit\b|\boffset\b|;", tail, maxsplit=1)[0]
        frag = frag.rstrip()

        if not frag:
            return True  # "FROM" і кінець рядка

        # 1) кома в кінці → чекаємо наступну таблицю
        if re.search(r",\s*$", frag):
            return True

        # 2) закінчується на JOIN або JOIN <name> [AS alias]
        if re.search(r"(?i)\bjoin\s*$", frag):
            return True
        if re.search(r"(?i)\bjoin\s+[a-z_]\w*(?:\s+as\s+[a-z_]\w*|\s+[a-z_]\w*)?\s*$", frag):
            return True

        # 3) USING ( ... без закриття
        if re.search(r"(?i)\busing\s*\(\s*$", frag):
            return True

        # 4) ON-частина: перевірити незавершеність
        on_pos = [mm.start() for mm in re.finditer(r"(?i)\bon\b", frag)]
        if on_pos:
            on_tail = frag[max(on_pos):]

            # лише "ON" → незавершено
            if re.fullmatch(r"(?is)\bon\b", on_tail.strip()):
                return True

            # незбалансовані дужки в ON (...)
            depth = 0
            in_s = in_d = False
            for ch in on_tail:
                if ch == "'" and not in_d:
                    in_s = not in_s
                elif ch == '"' and not in_s:
                    in_d = not in_d
                elif not in_s and not in_d:
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
            if depth > 0:
                return True

            # закінчується на крапку/оператор/логічне слово — правий операнд ще не дописаний
            if re.search(r"(\.|=|<>|<=|>=|<|>|\band\b|\bor\b)\s*$", on_tail, flags=re.I):
                return True

        # 5) лише ідентифікатор у хвості (таблиця/alias) без роздільника
        if re.search(r"(?i)[a-z_]\w*(?:\s+as\s+[a-z_]\w*|\s+[a-z_]\w*)?\s*$", frag):
            return True

        return False

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
        """
        Повертає:
        - tabs: множина фізичних таблиць (lowercase), що згадані у FROM/JOIN (без CTE/derived)
        - alias_map: alias -> base_source (для фізичних таблиць та для CTE/derived alias'ів)
        - derived_aliases: множина alias'ів для derived-джерел (FROM (SELECT ...) alias / JOIN (SELECT ...) alias)
        """
        s = re.sub(r"\s+", " ", sql, flags=re.I).strip()

        tabs: set[str] = set()
        alias_map: dict[str, str] = {}
        derived_aliases: set[str] = set()
        ctes = self._extract_cte_names(sql)  # {"c1","c2",...}

        # 0) Derived sources: FROM (SELECT ...) alias / JOIN (SELECT ...) alias
        for m in re.finditer(r"\b(from|join)\s*\(\s*select\b.*?\)\s*([a-zA-Z_]\w*)", s, flags=re.I | re.S):
            alias = m.group(2).lower()
            alias_map[alias] = alias  # self-map for derived
            derived_aliases.add(alias)

        # 1) Take the FROM section (till WHERE/GROUP/ORDER/HAVING/LIMIT/OFFSET/; or end)
        from_m = re.search(r"(?is)\bfrom\b(.*)", s)
        if not from_m:
            for c in ctes:
                alias_map.setdefault(c, c)
            return tabs, alias_map, derived_aliases

        tail = from_m.group(1)
        from_sec_full = re.split(r"(?is)\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\blimit\b|\boffset\b|;", tail, maxsplit=1)[0]

        # 2) Patterns compiled with flags (no inline (?ix))
        from_item_pat = re.compile(r"""
            ^\s*                                   # beginning of remaining fragment
            (?P<name>[a-z_]\w*(?:\.[a-z_]\w*)?)    # table (optionally schema.table)
            (?:\s+(?:as\s+)?(?P<alias>[a-z_]\w*))? # optional alias
            (?=                                    # lookahead: how this item ends
                \s*,                               #   comma (another FROM item follows)
            | \s+\bjoin\b                        #   JOIN-chain begins
            | \s*$                               #   end of FROM-section
            )
        """, flags=re.IGNORECASE | re.VERBOSE)

        sep_pat = re.compile(r"^\s*(,|\bjoin\b)", flags=re.IGNORECASE)

        join_pat = re.compile(r"""
            \bjoin\b
            \s+
            (?P<name>[a-z_]\w*(?:\.[a-z_]\w*)?)
            (?:\s+(?:as\s+)?(?P<alias>[a-z_]\w*))?
            (?=                                   # next must be one of
                \s+\bon\b                         #   ON ...
            | \s+\busing\b                      #   USING (...)
            | \s*,                              #   comma
            | \s+\bjoin\b                       #   next JOIN
            | \s*$                              #   end of FROM-section
            )
        """, flags=re.IGNORECASE | re.VERBOSE)

        # 3) Parse items BEFORE the first JOIN (comma-separated list)
        #    We only consume from a working copy; we do NOT modify from_sec_full.
        rest = from_sec_full
        while True:
            m = from_item_pat.match(rest)
            if not m:
                break

            raw = m.group("name")
            base = raw.split(".")[-1].lower()
            alias = (m.group("alias") or "").lower()

            if base not in ctes and base not in derived_aliases:
                tabs.add(base)
            if alias:
                alias_map[alias] = base
            else:
                alias_map.setdefault(base, base)

            # cut what we've matched
            rest = rest[m.end():]

            m_sep = sep_pat.match(rest)
            if not m_sep:
                break

            sep = m_sep.group(1).lower()
            rest = rest[m_sep.end():]
            if sep == "join":
                # Stop here; JOINs will be parsed from from_sec_full below
                break
            # If comma, continue to next FROM item

        # 4) Parse ALL JOINs from the full FROM section
        for jm in join_pat.finditer(from_sec_full):
            raw = jm.group("name")
            base = raw.split(".")[-1].lower()
            alias = (jm.group("alias") or "").lower()

            if base not in ctes and base not in derived_aliases:
                tabs.add(base)
            if alias:
                alias_map[alias] = base
            else:
                alias_map.setdefault(base, base)

        # 5) Add CTE names as sources in alias_map (so alias.col won't be checked against physical schema)
        for c in ctes:
            alias_map.setdefault(c, c)

        # 6) closed_only: keep only tables whose names are "closed" by a delimiter in the FULL FROM section
        if closed_only:
            closed_tabs: set[str] = set()
            for m in from_item_pat.finditer(from_sec_full):
                base = m.group("name").split(".")[-1].lower()
                if base in tabs:
                    closed_tabs.add(base)
            for jm in join_pat.finditer(from_sec_full):
                base = jm.group("name").split(".")[-1].lower()
                if base in tabs:
                    closed_tabs.add(base)
            tabs = closed_tabs

        return tabs, alias_map, derived_aliases

    def _strip_noise_for_cols(self, s: str) -> str:
        # прибираємо літерали, AS alias, імена функцій, кваліфіковані x.y → лишаємо y
        s = re.sub(r"'([^']|'')*'", " ", s)                        # рядкові літерали
        s = re.sub(r'\"([^"]|\"\")*\"', " ", s)                    # подвійні лапки (на всяк)
        s = re.sub(r"\b(as)\s+[a-zA-Z_]\w*", " ", s, flags=re.I)   # AS alias
        s = re.sub(r"[a-zA-Z_]\w*\s*\(", "(", s)                   # func_name( → (
        s = re.sub(r"([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)", r"\2", s)   # x.y → y
        return s

    def _unqualified_tokens(self, chunk: str, *, skip: set[str]) -> set[str]:
        """
        Повертати лише 'закриті' некваліфіковані імена:
        - після токена обов'язково має йти ПРОБІЛ або РОЗДІЛЮВАЧ,
        - токени, що стоять рівно в КІНЦІ фрагмента (як 'WHERE T') — ігноруємо (деферимо).
        """
        # Прибрати шум (літерали, AS alias, імена ф-цій, x.y -> y)
        chunk = self._strip_noise_for_cols(chunk)

        KW = {
            "select","distinct","from","where","on","join","inner","left","right","full","outer","cross","natural",
            "using","group","order","by","having","limit","offset","union","all","intersect","except",
            "and","or","not","in","like","between","is","null","exists","any","some","true","false",
            "case","when","then","else","end",
            "count","sum","avg","min","max","cast","coalesce","substr","printf","round","abs",
            "upper","lower","length","date","datetime","strftime","ifnull","nullif",
            "over","partition","rows","range","preceding","following","current","row",
            "asc","desc","nulls","first","last",
            "as","real","int","integer","text","boolean"
        }
        CLOSERS = {",", ")", ";", ".", "=", "<", ">", "+", "-", "*", "/", "%"}
        out: set[str] = set()

        # Проходимо токени з позиціями, щоб перевірити "закритість"
        for m in re.finditer(r"[A-Za-z_]\w*", chunk):
            tok = m.group(0)
            tl  = tok.lower()
            if tl in KW or tl in skip or tl.isnumeric():
                continue

            j = m.end()  # позиція відразу після токена
            # Знайти наступний НЕ-пробільний символ
            while j < len(chunk) and chunk[j].isspace():
                j += 1

            # 1) Якщо токен стоїть у КІНЦІ фрагмента -> вважаємо НЕЗАКРИТИМ -> пропускаємо (деферимо)
            if j >= len(chunk):
                continue

            # 2) Якщо далі йде роздільник або ключове слово — токен "закритий"
            ch = chunk[j]
            if ch in CLOSERS:
                out.add(tl)
                continue

            # 3) Перевірка на ключове слово після пробілу (наприклад, "WHERE amount DESC")
            if chunk[j].isalpha():
                nxt = chunk[j:].lower()
                # Достатньо короткого переліку найтиповіших, бо ми вже маємо загальний KW
                for kw in (" on", " where", " group", " order", " having",
                            " from", " join", " and", " or", " limit", " offset"):
                    if nxt.startswith(kw):
                        out.add(tl)
                        break
                else:
                    # Інакше після токена йде ще один ідентифікатор/продовження — не вважаємо закритим
                    pass
            else:
                # Не алфавітний символ і не з CLOSERS (рідкі випадки) — не закрито
                pass

        return out


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
        """
        1) Перевіряємо кваліфіковані alias.col / table.col (з деферингом, якщо alias не готовий).
        2) Якщо FROM відкритий → деферимо перевірку некваліфікованих колонок (return True).
        3) Інакше перевіряємо некваліфіковані:
        - якщо токен є рівно в одній фізичній таблиці з FROM → OK,
        - якщо ніде → False,
        - якщо в кількох → деферимо (даємо шанс моделі додати alias).
        """
        self._maybe_load_schema()
        if not self._tables:
            return True

        ctes = self._extract_cte_names(sql)
        tabs, alias_map, derived_aliases = self._extract_tables_and_aliases(sql, closed_only=closed_only)

        # --- 1) Кваліфіковані alias.col / table.col ---
        s = re.sub(r"\s+", " ", sql)
        qual_pat = r"([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)"
        qualified_seconds = set()

        for m in re.finditer(qual_pat, s):
            alias = m.group(1).lower()
            col   = m.group(2).lower()
            qualified_seconds.add(col)

            # чи "закрита" колонка (після неї вже роздільник/оператор/kw)
            j = m.end()
            while j < len(s) and s[j].isspace():
                j += 1
            if j >= len(s):
                continue

            closers = {",", ")", ";", ".", "=", "<", ">", "+", "-", "*", "/"}
            next_is_kw = False
            if s[j].isalpha():
                nxt = s[j:].lower()
                for kw in (" on", " where", " group", " order", " having", " from"):
                    if nxt.startswith(kw):
                        next_is_kw = True
                        break
            if (s[j] not in closers) and (not next_is_kw):
                continue  # ще можуть дописуватись символи колонки

            # alias ще не оголошено -> деферимо
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

            if col not in self._tables[src]:
                return False

        # --- 2) Якщо FROM ще відкритий — не чіпаємо некваліфіковані, даємо моделі дописати JOIN ---
        if self._from_clause_open(sql):
            return True

        # --- 3) Некваліфіковані: універсальний підхід для 1..N таблиць ---
        phys_tabs = {t for t in tabs if t in self._tables}
        skip_tokens = set(qualified_seconds)

        s_norm = re.sub(r"\s+", " ", s)
        sel_m   = re.search(r"(?is)\bselect\b(.*?)\bfrom\b", s_norm)
        where_m = re.search(r"(?is)\bwhere\b(.*?)(\bgroup\b|\border\b|\bhaving\b|\blimit\b|\boffset\b|;|$)", s_norm)
        group_m = re.search(r"(?is)\bgroup\s+by\b(.*?)(\border\b|\bhaving\b|\blimit\b|\boffset\b|;|$)", s_norm)
        order_m = re.search(r"(?is)\border\s+by\b(.*?)(\bhaving\b|\blimit\b|\boffset\b|;|$)", s_norm)
        having_m= re.search(r"(?is)\bhaving\b(.*?)(\blimit\b|\boffset\b|;|$)", s_norm)

        chunks = []
        if sel_m:    chunks.append(sel_m.group(1))
        if where_m:  chunks.append(where_m.group(1))
        if group_m:  chunks.append(group_m.group(1))
        if order_m:  chunks.append(order_m.group(1))
        if having_m: chunks.append(having_m.group(1))



        for chunk in chunks:
            for tok in self._unqualified_tokens(chunk, skip=skip_tokens):
                hit_tabs = sum(1 for t in phys_tabs if tok in self._tables.get(t, set()))
                if hit_tabs == 0:
                    return False
                elif hit_tabs >= 2:
                    # неоднозначно — деферимо (дай моделі шанс додати alias)
                    continue

        return True


if __name__ == "__main__":
    
    validator = SQLValidator(db_path="datasets/data_minidev/dev_databases/debit_card_specializing/debit_card_specializing.sqlite")

    sql = "SELECT SUM(CASE WHEN T2.Segment = 'Discount' THEN 1 ELSE 0 END) - SUM(CASE WHEN T1.Segment = 'Discount' THEN 1 ELSE 0 END) AS Difference\nFROM gasstations AS T1\nJOIN gasstations AS T2 ON T2.Country = 'SK'\nWHERE T"

    print("syntax ok:", validator.syntax_ok(sql))
    print("tables ok:", validator.tables_exist(sql))
    print("columns ok:", validator.columns_exist(sql))
