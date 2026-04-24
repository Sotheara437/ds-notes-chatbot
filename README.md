# Data Science Notes Chatbot (RAG + Streamlit + Groq)

A Streamlit app that lets you upload PDF notes, index them with embeddings, and ask questions grounded in those documents.

## Features

- Upload one or multiple PDF files and index them.
- Load PDFs from a local `data/` folder.
- Build and persist a Chroma vector index in `chroma_db/`.
- Ask questions with source-aware retrieval.
- Use Groq LLM via `GROQ_API_KEY`.

## Project Structure

```text
Chatbot/
	app.py
	testapi.py
	.env
	src/
		chain.py
		embeddings.py
		pdf_loader.py
		retriever.py
```

## Requirements

- Python 3.12 (recommended: 3.12.x)
- Windows PowerShell (commands below use PowerShell)
- Internet connection for model downloads/API calls

## 1. Create And Activate Virtual Environment

From project root (`D:\Chatbot`):

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

If execution policy blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
. .\.venv\Scripts\Activate.ps1
```

## 2. Install Dependencies

Use this exact command (matches current codebase):

```powershell
python -m pip install --upgrade pip
python -m pip install streamlit python-dotenv pypdf langchain langchain-core langchain-community langchain-classic langchain-groq langchain-text-splitters chromadb sentence-transformers torch torchvision groq
```

Notes:

- `torchvision` is required to avoid `ModuleNotFoundError: No module named 'torchvision'` in this environment.
- Current `requirements.txt` is placeholder text, so use the command above unless you regenerate `requirements.txt`.

## 3. Configure Environment Variables

Create `.env` in project root:

```env
GROQ_API_KEY=your_real_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant
```

`GROQ_MODEL` is optional; default is already set in code.

## 4. Verify API Key (Recommended)

```powershell
python testapi.py
```

Expected success output includes:

- `Groq API is working.`

If you get `401 invalid_api_key`, regenerate key and update `.env`.

## 5. Run The App

Important: run with Streamlit, not `python app.py`.

```powershell
python -m streamlit run app.py
```

Open the URL shown in terminal, usually:

- `http://localhost:8501`

If port is busy:

```powershell
python -m streamlit run app.py --server.port 8505
```

## 6. How To Use

1. Start the app.
2. In sidebar, upload PDF notes.
3. Click `Index Uploaded PDFs`.
4. Ask questions in chat input.
5. Optionally click `Load PDFs from /data` if you keep PDFs in `data/`.

## Common Issues And Fixes

### A) Blank/No UI

Cause: launching with `python app.py` or browser cache issue.

Fix:

```powershell
python -m streamlit run app.py --server.port 8505
```

Then hard refresh browser (`Ctrl+F5`).

### B) `missing ScriptRunContext` warnings

Cause: app started in bare Python mode.

Fix: always run with `streamlit run` as above.

### C) `No module named 'torchvision'`

Fix:

```powershell
python -m pip install torchvision
```

### D) `No PDFs found in /data folder`

Create `data/` in project root and place `.pdf` files inside.

### E) `.env` looks correct but key is still missing

Ensure file is saved on disk and non-empty:

```powershell
(Get-Item .env).Length
Get-Content .env
```

## Helpful Commands

Check syntax:

```powershell
python -m py_compile app.py
```

Deactivate venv:

```powershell
deactivate
```

## Security Note

If API keys were ever exposed in screenshots/chat, rotate them immediately and replace `.env` with new keys.

