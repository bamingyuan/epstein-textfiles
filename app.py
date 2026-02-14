from __future__ import annotations

import os
import re
import shlex
import sqlite3
from datetime import date, datetime, timedelta
from functools import lru_cache

from flask import Flask, abort, g, render_template, request, url_for
from markupsafe import Markup, escape


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def db_has_messages_table(db_path: str) -> bool:
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages' LIMIT 1"
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def resolve_default_db_path(base_dir: str = BASE_DIR) -> str:
    env_path = os.environ.get("PDF_MESSAGES_DB")
    if env_path:
        return env_path

    exported_path = os.path.join(base_dir, "exported_data.db")
    legacy_path = os.path.join(base_dir, "pdf_messages.db")
    for candidate in (exported_path, legacy_path):
        if db_has_messages_table(candidate):
            return candidate
    if os.path.exists(exported_path):
        return exported_path
    return legacy_path


DB_PATH = resolve_default_db_path()
PER_PAGE_DEFAULT = 50
PER_PAGE_MAX = 100

ALLOWED_SCOPE = {"current", "all"}
ALLOWED_SORT = {"date_desc", "date_asc"}

SEARCH_INDEX_SQL = [
    'CREATE INDEX IF NOT EXISTS idx_messages_date ON messages("date")',
    'CREATE INDEX IF NOT EXISTS idx_messages_dataset ON messages("dataset")',
    'CREATE INDEX IF NOT EXISTS idx_messages_filename ON messages("filename")',
    'CREATE INDEX IF NOT EXISTS idx_messages_url ON messages("url")',
]

FTS_SQL = [
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        filename,
        dataset,
        date,
        sender,
        recipient,
        doc_type,
        content,
        message,
        url
    )
    """,
    """
    INSERT INTO messages_fts(rowid, filename, dataset, date, sender, recipient, doc_type, content, message, url)
    SELECT
        id,
        filename,
        dataset,
        date,
        "from",
        "to",
        type,
        content,
        message,
        url
    FROM messages
    WHERE NOT EXISTS (SELECT 1 FROM messages_fts LIMIT 1)
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, filename, dataset, date, sender, recipient, doc_type, content, message, url)
        VALUES (new.id, new.filename, new.dataset, new.date, new."from", new."to", new.type, new.content, new.message, new.url);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
        DELETE FROM messages_fts WHERE rowid = old.id;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
        DELETE FROM messages_fts WHERE rowid = old.id;
        INSERT INTO messages_fts(rowid, filename, dataset, date, sender, recipient, doc_type, content, message, url)
        VALUES (new.id, new.filename, new.dataset, new.date, new."from", new."to", new.type, new.content, new.message, new.url);
    END
    """,
]


app = Flask(__name__)


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(_error: BaseException | None) -> None:
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def db_cache_token(db_path: str = DB_PATH) -> tuple[int, int]:
    try:
        stat = os.stat(db_path)
    except OSError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_size)


def open_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def to_int(value: str | None, fallback: int | None = None) -> int | None:
    if value is None or value == "":
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback


def clean_string(value: str | None) -> str:
    return (value or "").strip()


def extract_date_portion(value: str | None) -> str | None:
    text = clean_string(value)
    if not text:
        return None
    if len(text) >= 10 and re.match(r"^\d{4}-\d{2}-\d{2}$", text[:10]):
        return text[:10]
    return text


def extract_time_portion(value: str | None) -> str | None:
    text = clean_string(value)
    if not text:
        return None
    if len(text) >= 19 and re.match(r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}$", text[:19]):
        return text[11:19]
    if len(text) >= 8 and re.match(r"^\d{2}:\d{2}:\d{2}$", text[:8]):
        return text[:8]
    return text


@app.template_filter("display_date")
def display_date_filter(value: str | None) -> str:
    return extract_date_portion(value) or ""


@app.template_filter("display_time")
def display_time_filter(value: str | None) -> str:
    return extract_time_portion(value) or ""


def parse_search_terms(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        terms = shlex.split(raw)
    except ValueError:
        terms = raw.split()
    return [term.strip() for term in terms if term.strip()]


def extract_positive_search_terms(raw: str) -> list[str]:
    terms = parse_search_terms(raw)
    seen: set[str] = set()
    positive: list[str] = []

    for term in terms:
        if term.startswith("-") and len(term) > 1:
            continue
        normalized = term.strip().strip('"').strip()
        if not normalized:
            continue
        if normalized.upper() in {"AND", "OR", "NOT"}:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        positive.append(normalized)
    return positive


STRUCTURE_QUOTE_PREFIX_RE = re.compile(r"^\s*(?:>\s*)*")

STRUCTURE_LINE_RE = re.compile(
    r"^(?:"
    r"/{3,}\s*PAGE\b.*"
    r"|(?:From|To|Subject|Cc|Date)\b\s*:?\s*.*"
    r"|Sent\b(?!\s+from\s+my\b)\s*:?\s*.*"
    r"|On\b.*\bwrote\b\s*:?"
    r")\s*$",
    re.IGNORECASE,
)

ON_WROTE_LINE_RE = re.compile(r"^On\b.*\bwrote\b\s*:?$", re.IGNORECASE)
ON_LINE_START_RE = re.compile(r"^On\b", re.IGNORECASE)


def normalize_structure_line(line: str) -> str:
    return STRUCTURE_QUOTE_PREFIX_RE.sub("", line).strip()


def structure_line_indexes(text: str) -> set[int]:
    lines = text.splitlines()
    normalized_lines = [normalize_structure_line(line) for line in lines]
    structure_indexes: set[int] = set()

    for index, normalized in enumerate(normalized_lines):
        if not normalized:
            continue

        if STRUCTURE_LINE_RE.match(normalized):
            structure_indexes.add(index)
            continue

        if not ON_LINE_START_RE.match(normalized):
            continue

        combined = normalized
        max_join = min(index + 3, len(normalized_lines) - 1)
        for next_index in range(index + 1, max_join + 1):
            next_line = normalized_lines[next_index]
            if not next_line:
                break
            combined = f"{combined} {next_line}"
            if ON_WROTE_LINE_RE.match(combined):
                structure_indexes.update(range(index, next_index + 1))
                break

    return structure_indexes


def search_highlight_pattern(raw_query: str | None) -> re.Pattern[str] | None:
    if not raw_query:
        return None
    terms = extract_positive_search_terms(raw_query)
    if not terms:
        return None
    return re.compile(
        "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True)),
        re.IGNORECASE,
    )


def render_highlighted_text(text: str, pattern: re.Pattern[str] | None) -> Markup:
    if pattern is None:
        return escape(text)
    parts: list[Markup] = []
    cursor = 0

    for match in pattern.finditer(text):
        start, end = match.span()
        if start < cursor:
            continue
        parts.append(escape(text[cursor:start]))
        parts.append(Markup('<mark class="search-highlight">'))
        parts.append(escape(text[start:end]))
        parts.append(Markup("</mark>"))
        cursor = end

    parts.append(escape(text[cursor:]))
    return Markup("").join(parts)


@app.template_filter("highlight_search")
def highlight_search_filter(text: str | None, raw_query: str | None = None) -> Markup:
    if text is None:
        return Markup("")

    pattern = search_highlight_pattern(raw_query)
    structure_indexes = structure_line_indexes(text)
    rendered: list[Markup] = []

    for line_index, line in enumerate(text.splitlines(keepends=True)):
        line_text = line.rstrip("\r\n")
        line_break = line[len(line_text) :]

        line_markup = render_highlighted_text(line_text, pattern)
        if line_index in structure_indexes:
            line_markup = Markup('<strong class="message-structure">') + line_markup + Markup("</strong>")

        rendered.append(line_markup)
        if line_break:
            rendered.append(Markup(line_break))

    if not rendered:
        return render_highlighted_text(text, pattern)
    return Markup("").join(rendered)


def build_fts_query(raw: str) -> str:
    terms = parse_search_terms(raw)
    if not terms:
        return ""
    positive: list[str] = []
    negative: list[str] = []

    for term in terms:
        target = negative if term.startswith("-") and len(term) > 1 else positive
        normalized = term[1:] if target is negative else term
        normalized = normalized.replace('"', '""')
        if " " in normalized:
            target.append(f'"{normalized}"')
        else:
            target.append(f"{normalized}*")

    if not positive:
        return ""

    query = " AND ".join(positive)
    for term in negative:
        query = f"{query} NOT {term}"
    return query


def has_fts(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='messages_fts' LIMIT 1"
    ).fetchone()
    if row is None:
        return False
    sql = (row["sql"] or "").lower()
    # Reject old/broken external-content schema that expects sender/recipient columns in messages table.
    return "content='messages'" not in sql and "content = 'messages'" not in sql


@lru_cache(maxsize=16)
def cached_has_fts(db_path: str, cache_token: tuple[int, int]) -> bool:
    conn = open_db_connection(db_path)
    try:
        return has_fts(conn)
    finally:
        conn.close()


@lru_cache(maxsize=32)
def cached_messages_columns(db_path: str, cache_token: tuple[int, int]) -> tuple[str, ...]:
    conn = open_db_connection(db_path)
    try:
        rows = conn.execute('PRAGMA table_info("messages")').fetchall()
        return tuple(row["name"] for row in rows)
    finally:
        conn.close()


def has_time_column(db_path: str, cache_token: tuple[int, int]) -> bool:
    return "time" in cached_messages_columns(db_path, cache_token)


def date_exists_clause(alias: str = "m") -> str:
    return f'({alias}."date" IS NOT NULL AND TRIM({alias}."date") <> "")'


def undated_clause(alias: str = "m") -> str:
    return f'({alias}."date" IS NULL OR TRIM({alias}."date") = "")'


def date_value_clause(alias: str = "m") -> str:
    return f'SUBSTR({alias}."date", 1, 10)'


def get_latest_available_date(conn: sqlite3.Connection, dataset: str | None = None) -> str | None:
    message_date = date_value_clause("messages")
    where = [date_exists_clause("messages"), f"{message_date} <= date('now')"]
    params: list[str] = []
    if dataset:
        where.append('"messages"."dataset" = ?')
        params.append(dataset)

    query = f"""
        SELECT {message_date} AS "date"
        FROM messages
        WHERE {" AND ".join(where)}
        ORDER BY {message_date} DESC
        LIMIT 1
    """
    row = conn.execute(query, params).fetchone()
    if row and row["date"]:
        return row["date"]

    fallback = conn.execute(
        f"""
        SELECT {message_date} AS "date"
        FROM messages
        WHERE {date_exists_clause("messages")}
        ORDER BY {message_date} DESC
        LIMIT 1
        """
    ).fetchone()
    return fallback["date"] if fallback else None


def normalize_period(
    conn: sqlite3.Connection,
    year: int | None,
    month: int | None,
    day: int | None,
    undated: bool,
    dataset: str | None,
    scope: str,
) -> tuple[int | None, int | None, int | None]:
    if undated:
        return None, None, None

    if year is None and month is not None:
        month = None
        day = None
    if month is None and day is not None:
        day = None

    if year is None:
        if scope != "current":
            return None, None, None
        latest = get_latest_available_date(conn, dataset)
        if latest:
            latest_date = extract_date_portion(latest)
            if latest_date:
                try:
                    latest_dt = datetime.strptime(latest_date, "%Y-%m-%d").date()
                except ValueError:
                    latest_dt = None
                if latest_dt is not None:
                    return latest_dt.year, latest_dt.month, None
        return None, None, None

    if month is not None and not 1 <= month <= 12:
        month = None
        day = None
    if day is not None and not 1 <= day <= 31:
        day = None
    if year is not None and month is not None and day is not None:
        try:
            date(year, month, day)
        except ValueError:
            day = None

    return year, month, day


def date_range(year: int, month: int | None = None) -> tuple[str, str]:
    if month is None:
        start = date(year, 1, 1)
        end = date(year + 1, 1, 1)
        return start.isoformat(), end.isoformat()
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    return start.isoformat(), end.isoformat()


def day_range(year: int, month: int, day: int) -> tuple[str, str]:
    start = date(year, month, day)
    end = start + timedelta(days=1)
    return start.isoformat(), end.isoformat()


def build_browse_filters(
    dataset: str | None,
    scope: str,
    year: int | None,
    month: int | None,
    day: int | None,
    undated: bool,
) -> tuple[list[str], list[str]]:
    where = ["1=1"]
    params: list[str] = []

    if dataset:
        where.append('m."dataset" = ?')
        params.append(dataset)

    if undated:
        where.append(undated_clause("m"))
        return where, params

    # "Dated" browse mode should always exclude undated rows, even for "All dates".
    where.append(date_exists_clause("m"))

    if day is not None and year is not None and month is not None:
        start, end = day_range(year, month, day)
        where.append('m."date" >= ? AND m."date" < ?')
        params.extend([start, end])
    elif month is not None and year is not None:
        start, end = date_range(year, month)
        where.append('m."date" >= ? AND m."date" < ?')
        params.extend([start, end])
    elif year is not None:
        start, end = date_range(year)
        where.append('m."date" >= ? AND m."date" < ?')
        params.extend([start, end])
    return where, params


def apply_like_search(where: list[str], params: list[str], query: str) -> None:
    terms = parse_search_terms(query)
    for term in terms:
        like = f"%{term}%"
        where.append(
            """
            (
                m."filename" LIKE ?
                OR m."content" LIKE ?
                OR m."message" LIKE ?
                OR m."url" LIKE ?
            )
            """
        )
        params.extend([like, like, like, like])


def sort_clause(sort: str, include_time: bool) -> str:
    if include_time:
        # Keep rows without a time value at the end in both sort directions.
        time_missing_last = 'CASE WHEN m."time" IS NULL OR TRIM(m."time") = "" THEN 1 ELSE 0 END ASC'
        ascending = f'm."date" ASC, {time_missing_last}, m."time" ASC, m."id" ASC'
        descending = f'm."date" DESC, {time_missing_last}, m."time" DESC, m."id" DESC'
    else:
        ascending = 'm."date" ASC, m."id" ASC'
        descending = 'm."date" DESC, m."id" DESC'

    if sort == "date_desc":
        return descending
    return ascending


def dataset_counts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT "dataset", COUNT(*) AS count
        FROM messages
        GROUP BY "dataset"
        ORDER BY count DESC, "dataset" ASC
        """
    ).fetchall()


def year_counts(conn: sqlite3.Connection, dataset: str | None) -> list[sqlite3.Row]:
    where = [date_exists_clause("messages")]
    params: list[str] = []
    if dataset:
        where.append('"messages"."dataset" = ?')
        params.append(dataset)
    query = f"""
        SELECT CAST(SUBSTR("date", 1, 4) AS INTEGER) AS year, COUNT(*) AS count
        FROM messages
        WHERE {" AND ".join(where)}
        GROUP BY year
        ORDER BY year DESC
    """
    return conn.execute(query, params).fetchall()


def month_counts(conn: sqlite3.Connection, year: int | None, dataset: str | None) -> list[sqlite3.Row]:
    if year is None:
        return []
    where = [date_exists_clause("messages"), 'SUBSTR("date", 1, 4) = ?']
    params: list[str] = [f"{year:04d}"]
    if dataset:
        where.append('"messages"."dataset" = ?')
        params.append(dataset)
    query = f"""
        SELECT CAST(SUBSTR("date", 6, 2) AS INTEGER) AS month, COUNT(*) AS count
        FROM messages
        WHERE {" AND ".join(where)}
        GROUP BY month
        ORDER BY month ASC
    """
    return conn.execute(query, params).fetchall()


def day_counts(
    conn: sqlite3.Connection, year: int | None, month: int | None, dataset: str | None
) -> list[sqlite3.Row]:
    if year is None or month is None:
        return []
    start, end = date_range(year, month)
    where = [date_exists_clause("messages"), '"messages"."date" >= ? AND "messages"."date" < ?']
    params: list[str] = [start, end]
    if dataset:
        where.append('"messages"."dataset" = ?')
        params.append(dataset)
    query = f"""
        SELECT CAST(SUBSTR("date", 9, 2) AS INTEGER) AS day, COUNT(*) AS count
        FROM messages
        WHERE {" AND ".join(where)}
        GROUP BY day
        ORDER BY day ASC
    """
    return conn.execute(query, params).fetchall()


def rows_to_dict_tuple(rows: list[sqlite3.Row]) -> tuple[dict[str, object], ...]:
    return tuple(dict(row) for row in rows)


@lru_cache(maxsize=64)
def cached_dataset_counts(db_path: str, cache_token: tuple[int, int]) -> tuple[dict[str, object], ...]:
    conn = open_db_connection(db_path)
    try:
        return rows_to_dict_tuple(dataset_counts(conn))
    finally:
        conn.close()


@lru_cache(maxsize=256)
def cached_year_counts(
    db_path: str, cache_token: tuple[int, int], dataset: str | None
) -> tuple[dict[str, object], ...]:
    conn = open_db_connection(db_path)
    try:
        return rows_to_dict_tuple(year_counts(conn, dataset))
    finally:
        conn.close()


@lru_cache(maxsize=512)
def cached_month_counts(
    db_path: str, cache_token: tuple[int, int], year: int | None, dataset: str | None
) -> tuple[dict[str, object], ...]:
    if year is None:
        return ()
    conn = open_db_connection(db_path)
    try:
        return rows_to_dict_tuple(month_counts(conn, year, dataset))
    finally:
        conn.close()


@lru_cache(maxsize=1024)
def cached_day_counts(
    db_path: str,
    cache_token: tuple[int, int],
    year: int | None,
    month: int | None,
    dataset: str | None,
) -> tuple[dict[str, object], ...]:
    if year is None or month is None:
        return ()
    conn = open_db_connection(db_path)
    try:
        return rows_to_dict_tuple(day_counts(conn, year, month, dataset))
    finally:
        conn.close()


@lru_cache(maxsize=512)
def cached_browse_page(
    db_path: str,
    cache_token: tuple[int, int],
    q: str,
    dataset: str | None,
    scope: str,
    sort: str,
    page: int,
    per_page: int,
    undated: bool,
    year: int | None,
    month: int | None,
    day: int | None,
    include_time: bool,
) -> dict[str, object]:
    conn = open_db_connection(db_path)
    try:
        where, params = build_browse_filters(dataset, scope, year, month, day, undated)
        from_sql = "messages m"
        fts_enabled = cached_has_fts(db_path, cache_token)

        if q:
            fts_query = build_fts_query(q)
            if fts_enabled and fts_query:
                from_sql = "messages m JOIN messages_fts ON messages_fts.rowid = m.id"
                where.append("messages_fts MATCH ?")
                params.append(fts_query)
            else:
                apply_like_search(where, params, q)

        where_sql = " AND ".join(where)
        count_sql = f"SELECT COUNT(*) AS count FROM {from_sql} WHERE {where_sql}"
        total = conn.execute(count_sql, params).fetchone()["count"]
        total_pages = max(1, (total + per_page - 1) // per_page)
        current_page = min(page, total_pages)
        offset = (current_page - 1) * per_page

        order_sql = sort_clause(sort, include_time)
        time_select_sql = 'm."time" AS "time"' if include_time else 'NULL AS "time"'
        list_sql = f"""
            SELECT
                m."id",
                m."filename",
                m."dataset",
                m."date",
                {time_select_sql},
                m."url",
                m."content",
                SUBSTR(m."message", 1, 700) AS preview
            FROM {from_sql}
            WHERE {where_sql}
            ORDER BY {order_sql}
            LIMIT ? OFFSET ?
        """
        rows = conn.execute(list_sql, params + [per_page, offset]).fetchall()

        return {
            "total": total,
            "total_pages": total_pages,
            "page": current_page,
            "rows": rows_to_dict_tuple(rows),
            "fts_enabled": fts_enabled,
        }
    finally:
        conn.close()


@app.route("/")
@app.route("/browse")
def browse() -> str:
    conn = get_db()
    q = clean_string(request.args.get("q"))
    dataset = clean_string(request.args.get("dataset")) or None
    scope = clean_string(request.args.get("scope")) or "current"
    if scope not in ALLOWED_SCOPE:
        scope = "current"

    sort = clean_string(request.args.get("sort"))
    if sort not in ALLOWED_SORT:
        sort = "date_asc"

    page = max(1, to_int(request.args.get("page"), 1) or 1)
    per_page = to_int(request.args.get("per_page"), PER_PAGE_DEFAULT) or PER_PAGE_DEFAULT
    per_page = max(10, min(PER_PAGE_MAX, per_page))

    undated = request.args.get("undated") == "1"
    year = to_int(request.args.get("year"))
    month = to_int(request.args.get("month"))
    day = to_int(request.args.get("day"))
    selected_doc_id = to_int(request.args.get("doc"))

    year, month, day = normalize_period(conn, year, month, day, undated, dataset, scope)
    cache_token = db_cache_token(DB_PATH)
    include_time = has_time_column(DB_PATH, cache_token)
    page_data = cached_browse_page(
        DB_PATH,
        cache_token,
        q,
        dataset,
        scope,
        sort,
        page,
        per_page,
        undated,
        year,
        month,
        day,
        include_time,
    )
    total = page_data["total"]
    total_pages = page_data["total_pages"]
    page = page_data["page"]
    rows = page_data["rows"]

    if selected_doc_id is None and rows:
        selected_doc_id = rows[0]["id"]

    selected_doc = None
    if selected_doc_id is not None:
        time_select_sql = '"time" AS "time"' if include_time else 'NULL AS "time"'
        selected_doc = conn.execute(
            f"""
            SELECT
                "id",
                "filename",
                "dataset",
                "date",
                {time_select_sql},
                "from",
                "to",
                "type",
                "content",
                "message",
                "url"
            FROM messages
            WHERE "id" = ?
            """,
            [selected_doc_id],
        ).fetchone()

    years = cached_year_counts(DB_PATH, cache_token, dataset)
    months = cached_month_counts(DB_PATH, cache_token, year, dataset)
    days = cached_day_counts(DB_PATH, cache_token, year, month, dataset)
    datasets = cached_dataset_counts(DB_PATH, cache_token)

    return render_template(
        "browse.html",
        q=q,
        dataset=dataset,
        scope=scope,
        sort=sort,
        page=page,
        per_page=per_page,
        total=total,
        total_pages=total_pages,
        rows=rows,
        selected_doc=selected_doc,
        selected_doc_id=selected_doc_id,
        years=years,
        months=months,
        days=days,
        year=year,
        month=month,
        day=day,
        undated=undated,
        datasets=datasets,
        fts_enabled=page_data["fts_enabled"],
    )


@app.route("/doc/<int:doc_id>")
def doc_detail(doc_id: int) -> str:
    conn = get_db()
    cache_token = db_cache_token(DB_PATH)
    include_time = has_time_column(DB_PATH, cache_token)
    time_select_sql = '"time" AS "time"' if include_time else 'NULL AS "time"'
    doc = conn.execute(
        f"""
        SELECT
            "id",
            "filename",
            "dataset",
            "date",
            {time_select_sql},
            "from",
            "to",
            "type",
            "content",
            "message",
            "url"
        FROM messages
        WHERE "id" = ?
        """,
        [doc_id],
    ).fetchone()

    if doc is None:
        abort(404)

    back = clean_string(request.args.get("back")) or url_for("browse")
    return render_template("doc.html", doc=doc, back=back)


def initialize_search_objects(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        schema_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='messages_fts' LIMIT 1"
        ).fetchone()
        schema_sql = (schema_row[0] or "").lower() if schema_row else ""
        incompatible_external_content = (
            "content='messages'" in schema_sql or "content = 'messages'" in schema_sql
        )
        if incompatible_external_content:
            conn.execute("DROP TRIGGER IF EXISTS messages_ai")
            conn.execute("DROP TRIGGER IF EXISTS messages_ad")
            conn.execute("DROP TRIGGER IF EXISTS messages_au")
            conn.execute("DROP TABLE IF EXISTS messages_fts")

        for statement in SEARCH_INDEX_SQL:
            conn.execute(statement)
        for statement in FTS_SQL:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


@app.cli.command("init-search")
def init_search_command() -> None:
    initialize_search_objects(DB_PATH)
    print("Search indexes and FTS table are ready.")


if __name__ == "__main__":
    app.run(debug=True)
