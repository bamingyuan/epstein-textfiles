# Epstein Document Browser

Document browser and search UI for newest Epstein files. Simple browsing like in an e-mail client. Included are Dataset 9 (partially), 10, 11.

![Tool screenshot](static/epstein-browser.png?raw=true "Epstein Browser")

## Features

- Three-pane layout (email-style):
  - Left: year/month/day navigation (foldable)
  - Middle: paginated document list
  - Right: document preview with full text
- Search with scope:
  - Current date view
  - Entire archive
- Sorting:
  - Date/time oldest first
- Clickable original source URL for each document
- Use the keyboard arrow keys Up and Down to browse through the documente and space to scroll through longer documents

## Quickstart

1. Download the database `pdf_messages.db.gz` from [mega.nz](https://mega.nz/file/OQcynIDS#zarOf5Yz_NjP9IBCC6Qva9rK0VLkcarxc2AFbQZ27Ng). Extract it into the project root so you end up with `pdf_messages.db`.
2. Optional: Create and activate a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Optional but strongly recommended: initialize indexes + FTS once.

```bash
python scripts/init_search.py --db pdf_messages.db
```

5. Run the app:

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## Notes

- `from`, `to`, and `type` are shown in detail views, but are currently mostly empty in this dataset. This need further AI analysis and at the moment I don't want to pay for the tokens, yet.
- Without FTS initialization, search falls back to `LIKE` and can be slower on large queries.
