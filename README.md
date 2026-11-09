# PDF Document Search Engine

A local Python web app that lets you upload PDFs and search through them using both keyword matching and semantic (meaning-based) search. Everything runs on your machine — no cloud APIs or internet needed after initial setup.

## What it does

- **Upload PDFs** through a browser interface (drag & drop or click)
- **Extract text** page by page using PyMuPDF
- **Keyword search** via SQLite FTS5 with porter stemming
- **Semantic search** using sentence-transformers + FAISS (all local)
- **Hybrid ranking** combines both scores with adjustable weights
- Results always show the **source document name and page number**

## Tech stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| PDF parsing | PyMuPDF (fitz) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | FAISS (CPU) |
| Keyword DB | SQLite FTS5 |
| Frontend | HTML, CSS, vanilla JS |
| Tests | pytest |

## How it works

```
Browser → FastAPI Backend
              ↓
    ┌─────────┴──────────┐
    │                    │
PDF Service        Search Service
    │                    │
  PyMuPDF        ┌──────┴──────┐
    ↓            │             │
  Extract     SQLite FTS5    FAISS
  text         (keyword)   (semantic)
    ↓            │             │
  Chunk       Normalize    Normalize
    ↓            │             │
  Embed         └──────┬──────┘
    ↓                  ↓
  FAISS          Hybrid Ranking
                      ↓
              Doc name + Page #
```

## Project structure

```
├── app/
│   ├── main.py                  # FastAPI app + startup
│   ├── api/routes.py            # API endpoints
│   ├── services/
│   │   ├── pdf_service.py       # PDF validation + extraction
│   │   ├── chunk_service.py     # Text chunking
│   │   ├── embedding_service.py # Sentence transformer wrapper
│   │   ├── keyword_search.py    # SQLite FTS5 search
│   │   ├── semantic_search.py   # FAISS search
│   │   └── hybrid_search.py     # Score merging
│   ├── database/
│   │   ├── sqlite_db.py         # SQLite manager
│   │   └── vector_store.py      # FAISS index manager
│   ├── models/schemas.py        # Pydantic models
│   └── utils/config.py          # Settings
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/                        # created at runtime
│   ├── pdfs/
│   └── index/
├── tests/
├── requirements.txt
├── main.py                      # Application entry point
└── run.py                       # Compatibility entry point
```

## Setup

You need Python 3.10 or newer.

```powershell
# go to project directory
cd "m:\KT2I task\python pdf task"

# create virtual environment
python -m venv .venv
.venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

> The first run downloads the `all-MiniLM-L6-v2` model (~90 MB). After that everything works offline.

## Running

```powershell
python main.py
```

Open http://localhost:8000 in your browser.

## Usage

1. Click "Upload PDF" or drag a file onto the page
2. Wait for indexing to finish
3. Type a question in the search box and hit Enter
4. Results show matched text with the document name and page number
5. Click "View Page" to open the PDF at that page

### Search examples

**Keyword search** — finds exact word matches:
```
Query:  password
Result: "Employees must change their password every 90 days." — Page 12
```

**Semantic search** — understands meaning:
```
Query:  How frequently should I update my credentials?
Result: "Employees must change their password every 90 days." — Page 12
```

Both types run together. You can adjust the keyword/semantic weight balance using the settings gear icon.

## API

Swagger docs at http://localhost:8000/api/docs

| Method | Path | What it does |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/documents/upload` | Upload a PDF |
| GET | `/api/v1/documents` | List documents |
| DELETE | `/api/v1/documents/{id}` | Delete a document |
| GET | `/api/v1/documents/{id}/view` | View PDF |
| POST | `/api/v1/search` | Search |

### Search request

```json
POST /api/v1/search
{
    "query": "How often should employees change their password?",
    "top_k": 5,
    "keyword_weight": 0.4
}
```

### Search response

```json
{
    "query": "How often should employees change their password?",
    "results": [
        {
            "document_name": "employee_handbook.pdf",
            "page_number": 12,
            "score": 0.91,
            "keyword_score": 0.85,
            "semantic_score": 0.95,
            "text": "Employees must change their password every 90 days."
        }
    ],
    "total_results": 1,
    "search_time_ms": 42.3
}
```

## Configuration

You can set these in a `.env` file in the project root:

```
MODEL_NAME=all-MiniLM-L6-v2
CHUNK_SIZE=600
CHUNK_OVERLAP=100
TOP_K=5
KEYWORD_WEIGHT=0.4
SEMANTIC_WEIGHT=0.6
MAX_FILE_SIZE_MB=50
PORT=8000
```

## Running tests

```powershell
python -m pytest tests/ -v
```

## Data storage

Everything is stored locally:

| Path | Contents |
|------|----------|
| `data/pdfs/` | Uploaded PDFs |
| `data/search.db` | SQLite database |
| `data/index/faiss.index` | FAISS vectors |
| `data/index/faiss_metadata.json` | Vector metadata |

## Known limitations

- Scanned PDFs (images without text layer) won't return results
- Password-protected PDFs are rejected
- Large PDFs (200+ pages) can take 30-60s to index
- Upload one file at a time

## Acceptance Criteria

- **PDF Upload**: Users can successfully upload a PDF document through the application interface.
- **Text Extraction**: The application accurately extracts text content from the uploaded PDF document.
- **Indexing**: The extracted text is properly indexed (both keyword and semantic vector indexing) to allow for efficient searching.
- **Fresh State on Startup**: Each time the application is started, the previous data and documents are cleared so the app begins in a fresh state.
- **Relevant Passages**: When a user submits a query, the application returns passages that are highly relevant to the search intent.
- **Correct Attribution**: Search results accurately point to the correct pages within the document where the passages are found.
- **Document Metadata**: Each search result clearly includes both the source document name and the specific page reference.
- **Offline Capability**: After the initial setup (and downloading of any necessary local models), the entire system works offline without internet access.

## License

MIT
#   p y t h o n - p d f 
 
 