# Flask Document Browser

Date-first document browser and search UI for `pdf_messages.db`.

## Features

- Three-pane layout (email-style):
  - Left: year/month/day navigation
  - Middle: paginated document list
  - Right: document preview with full text
- Search with scope:
  - Current date view
  - Entire archive
- Sorting:
  - Relevance
  - Date newest first
  - Date oldest first
- Dedicated "Undated" browsing mode
- Clickable original source URL for each document
- Full document permalink page

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional but strongly recommended: initialize indexes + FTS once.

```bash
python scripts/init_search.py --db pdf_messages.db
```

4. Run the app:

```bash
flask --app app run --debug
```

Open `http://127.0.0.1:5000`.

## Notes

- The app defaults to the latest available dated month for browsing.
- `from`, `to`, and `type` are shown in detail views, but are currently mostly empty in this dataset.
- Without FTS initialization, search falls back to `LIKE` and can be slower on large queries.
