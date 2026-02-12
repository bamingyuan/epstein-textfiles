from __future__ import annotations

import os
import re
import shlex
import sqlite3
from datetime import date, datetime

from flask import Flask, abort, g, render_template, request, url_for
from markupsafe import Markup, escape


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.environ.get("PDF_MESSAGES_DB", os.path.join(BASE_DIR, "pdf_messages.db"))
PER_PAGE_DEFAULT = 50
PER_PAGE_MAX = 100

ALLOWED_SCOPE = {"current", "all"}
ALLOWED_SORT = {"date_desc", "date_asc", "relevance"}

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


def to_int(value: str | None, fallback: int | None = None) -> int | None:
    if value is None or value == "":
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback


def clean_string(value: str | None) -> str:
    return (value or "").strip()


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


@app.template_filter("highlight_search")
def highlight_search_filter(text: str | None, raw_query: str | None = None) -> Markup:
    if text is None:
        return Markup("")
    if not raw_query:
        return escape(text)

    terms = extract_positive_search_terms(raw_query)
    if not terms:
        return escape(text)

    pattern = re.compile("|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True)), re.IGNORECASE)
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


def date_exists_clause(alias: str = "m") -> str:
    return f'({alias}."date" IS NOT NULL AND TRIM({alias}."date") <> "")'


def undated_clause(alias: str = "m") -> str:
    return f'({alias}."date" IS NULL OR TRIM({alias}."date") = "")'


def get_latest_available_date(conn: sqlite3.Connection, dataset: str | None = None) -> str | None:
    where = [date_exists_clause("messages"), '"messages"."date" <= date("now")']
    params: list[str] = []
    if dataset:
        where.append('"messages"."dataset" = ?')
        params.append(dataset)

    query = f"""
        SELECT "date"
        FROM messages
        WHERE {" AND ".join(where)}
        ORDER BY "date" DESC
        LIMIT 1
    """
    row = conn.execute(query, params).fetchone()
    if row and row["date"]:
        return row["date"]

    fallback = conn.execute(
        """
        SELECT "date"
        FROM messages
        WHERE "date" IS NOT NULL AND TRIM("date") <> ""
        ORDER BY "date" DESC
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
) -> tuple[int | None, int | None, int | None]:
    if undated:
        return None, None, None

    if year is None and month is not None:
        month = None
        day = None
    if month is None and day is not None:
        day = None

    if year is None:
        latest = get_latest_available_date(conn, dataset)
        if latest:
            latest_dt = datetime.strptime(latest, "%Y-%m-%d").date()
            return latest_dt.year, latest_dt.month, None
        return None, None, None

    if month is not None and not 1 <= month <= 12:
        month = None
        day = None
    if day is not None and not 1 <= day <= 31:
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

    if scope == "current":
        where.append(date_exists_clause("m"))
        if day is not None and year is not None and month is not None:
            where.append('m."date" = ?')
            params.append(f"{year:04d}-{month:02d}-{day:02d}")
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


def sort_clause(sort: str, use_relevance: bool) -> str:
    if sort == "date_asc":
        return 'm."date" ASC, m."id" ASC'
    if sort == "relevance" and use_relevance:
        return 'bm25(f), m."date" DESC, m."id" DESC'
    return '(m."date" IS NULL OR TRIM(m."date") = "") ASC, m."date" DESC, m."id" DESC'


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
        sort = "relevance" if q else "date_desc"

    page = max(1, to_int(request.args.get("page"), 1) or 1)
    per_page = to_int(request.args.get("per_page"), PER_PAGE_DEFAULT) or PER_PAGE_DEFAULT
    per_page = max(10, min(PER_PAGE_MAX, per_page))

    undated = request.args.get("undated") == "1"
    year = to_int(request.args.get("year"))
    month = to_int(request.args.get("month"))
    day = to_int(request.args.get("day"))
    selected_doc_id = to_int(request.args.get("doc"))

    year, month, day = normalize_period(conn, year, month, day, undated, dataset)

    where, params = build_browse_filters(dataset, scope, year, month, day, undated)
    from_sql = "messages m"
    use_fts = False

    if q:
        fts_query = build_fts_query(q)
        if has_fts(conn) and fts_query:
            from_sql = "messages m JOIN messages_fts f ON f.rowid = m.id"
            where.append("messages_fts MATCH ?")
            params.append(fts_query)
            use_fts = True
        else:
            apply_like_search(where, params, q)

    where_sql = " AND ".join(where)
    count_sql = f"SELECT COUNT(*) AS count FROM {from_sql} WHERE {where_sql}"
    total = conn.execute(count_sql, params).fetchone()["count"]
    total_pages = max(1, (total + per_page - 1) // per_page)
    if page > total_pages:
        page = total_pages
    offset = (page - 1) * per_page

    order_sql = sort_clause(sort, use_fts and bool(q))
    list_sql = f"""
        SELECT
            m."id",
            m."filename",
            m."dataset",
            m."date",
            m."url",
            m."content",
            SUBSTR(m."message", 1, 700) AS preview
        FROM {from_sql}
        WHERE {where_sql}
        ORDER BY {order_sql}
        LIMIT ? OFFSET ?
    """
    rows = conn.execute(list_sql, params + [per_page, offset]).fetchall()

    if selected_doc_id is None and rows:
        selected_doc_id = rows[0]["id"]

    selected_doc = None
    if selected_doc_id is not None:
        selected_doc = conn.execute(
            """
            SELECT
                "id",
                "filename",
                "dataset",
                "date",
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

    years = year_counts(conn, dataset)
    months = month_counts(conn, year, dataset)
    days = day_counts(conn, year, month, dataset)
    datasets = dataset_counts(conn)

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
        fts_enabled=has_fts(conn),
    )


@app.route("/doc/<int:doc_id>")
def doc_detail(doc_id: int) -> str:
    conn = get_db()
    doc = conn.execute(
        """
        SELECT
            "id",
            "filename",
            "dataset",
            "date",
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
