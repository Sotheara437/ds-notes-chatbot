<!-- update -->
import streamlit as st
import os
import tempfile

# ── SAFE IMPORTS WITH ERROR HANDLING ─────────────────────────────────────────
try:
    from src.pdf_loader import load_pdf, load_all_pdfs_from_folder
    from src.embeddings import split_text_into_chunks
    from src.retriever import build_vector_store, load_vector_store, get_retriever
    from src.chain import build_rag_chain, ask_question
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.info("Make sure all packages are installed: pip install -r requirements.txt")
    st.stop()

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Science Notes Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp { background-color: #0d1117 !important; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d !important;
}
section[data-testid="stSidebar"] * { color: #f0f6fc !important; }

/* ── ALL BUTTONS BASE ── */
.stButton > button {
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── BUTTON COLORS BY KEY ── */
button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
}

/* Index = purple */
div[data-testid="stSidebar"] div:nth-child(1) .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
}
/* Load /data = amber/orange */
div[data-testid="stSidebar"] div:nth-child(2) .stButton > button {
    background: linear-gradient(135deg, #b45309, #d97706) !important;
}
/* Load Saved = blue */
div[data-testid="stSidebar"] div:nth-child(3) .stButton > button {
    background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
}
/* Clear chat = red */
div[data-testid="stSidebar"] div:nth-child(4) .stButton > button {
    background: linear-gradient(135deg, #b91c1c, #ef4444) !important;
}

/* ── FILE UPLOADER - FULL DARK OVERRIDE ── */
[data-testid="stFileUploader"] {
    background: #1c2128 !important;
    border: 2px dashed #7c3aed !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] section {
    background: #1c2128 !important;
    border: none !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #1c2128 !important;
    border: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    background: #1c2128 !important;
    color: #8b949e !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #8b949e !important;
    fill: #8b949e !important;
}
[data-testid="stFileUploader"] button {
    background: #30363d !important;
    color: #f0f6fc !important;
    border: 1px solid #6e7681 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #7c3aed !important;
    border-color: #7c3aed !important;
}
[data-testid="stFileUploader"] span {
    color: #c9d1d9 !important;
}
[data-testid="stFileUploader"] small {
    color: #8b949e !important;
}

/* File name text after upload */
[data-testid="stFileUploader"] * {
    color: #c9d1d9 !important;
}
[data-testid="stFileUploaderFile"] {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
}
[data-testid="stFileUploaderFileName"] {
    color: #58a6ff !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
[data-testid="stFileUploaderFileData"] {
    color: #8b949e !important;
    font-size: 0.78rem !important;
}

/* Delete X button */
[data-testid="stFileUploader"] button[title="Delete"] {
    background: rgba(248,81,73,0.15) !important;
    border: 1px solid rgba(248,81,73,0.3) !important;
    border-radius: 6px !important;
    color: #f85149 !important;
}
[data-testid="stFileUploader"] button[title="Delete"]:hover {
    background: rgba(248,81,73,0.3) !important;
}

/* Plus add more files button */
[data-testid="stFileUploader"] button[title="Upload"] {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #8b949e !important;
    border-radius: 8px !important;
}

/* Upload icon color */
[data-testid="stFileUploaderDropzoneInstructions"] svg {
    fill: #7c3aed !important;
    color: #7c3aed !important;
}


/* ── CHAT ── */
[data-testid="stChatMessage"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
    margin: 0.5rem 0 !important;
}
[data-testid="stChatInput"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #f0f6fc !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #6e7681 !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: #1c2128 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
}

/* ── PROGRESS ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #7c3aed, #a855f7) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7c3aed; }

/* ── CARDS ── */
.header-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}
.header-title { font-size: 2.8rem; font-weight: 800; margin: 0 0 0.5rem 0; }
.header-title .purple {
    background: linear-gradient(90deg, #7c3aed, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.header-title .white { color: #f0f6fc; }
.header-subtitle { font-size: 1rem; color: #8b949e; margin: 0; line-height: 1.6; }

.feature-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    display: flex;
    align-items: flex-start;
    gap: 14px;
    height: 100%;
}
.feature-card:hover { border-color: #7c3aed; }
.feature-icon {
    font-size: 1.8rem;
    padding: 8px;
    border-radius: 10px;
    min-width: 46px;
    text-align: center;
}
.fi-purple { background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.3); }
.fi-green  { background: rgba(63,185,80,0.12);  border: 1px solid rgba(63,185,80,0.25); }
.fi-blue   { background: rgba(88,166,255,0.12); border: 1px solid rgba(88,166,255,0.25); }
.fi-orange { background: rgba(249,115,22,0.12); border: 1px solid rgba(249,115,22,0.25); }
.feature-title { font-size: 0.95rem; font-weight: 700; color: #f0f6fc; margin: 0 0 4px 0; }
.feature-desc  { font-size: 0.8rem; color: #8b949e; margin: 0; line-height: 1.5; }

.status-card {
    background: #1c2128; border: 1px solid #30363d;
    border-radius: 12px; padding: 12px 14px;
    display: flex; align-items: center; gap: 12px; margin-bottom: 1.2rem;
}
.status-icon {
    font-size: 1.5rem; padding: 8px; border-radius: 10px; min-width: 44px; text-align: center;
}
.status-icon-green { background: rgba(63,185,80,0.1); border: 1px solid rgba(63,185,80,0.2); }
.status-icon-warn  { background: rgba(210,153,34,0.1); border: 1px solid rgba(210,153,34,0.2); }
.status-title { font-size: 0.9rem; font-weight: 600; color: #f0f6fc !important; margin: 0; }
.status-sub   { font-size: 0.78rem; color: #8b949e !important; margin: 0; }

.sidebar-header {
    display: flex; align-items: center; gap: 12px; margin-bottom: 6px;
}
.sidebar-header-icon {
    font-size: 1.4rem; background: rgba(124,58,237,0.15);
    padding: 8px; border-radius: 10px; border: 1px solid rgba(124,58,237,0.3);
}
.sidebar-header-text { font-size: 1.1rem; font-weight: 700; color: #f0f6fc !important; }
.sidebar-subtitle { font-size: 0.82rem; color: #8b949e !important; margin-bottom: 1.2rem; }

.section-header { display: flex; align-items: center; gap: 10px; margin: 1.2rem 0 0.5rem 0; }
.section-icon { font-size: 1.2rem; }
.section-title { font-size: 0.95rem; font-weight: 700; color: #f0f6fc !important; }
.section-desc { font-size: 0.8rem; color: #8b949e !important; margin-bottom: 0.6rem; }

.welcome-box {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 16px; padding: 4rem 2rem; text-align: center; margin: 1rem 0;
}
.welcome-title { font-size: 1.6rem; font-weight: 700; color: #f0f6fc; margin-bottom: 0.5rem; }
.welcome-sub   { font-size: 1rem; color: #8b949e; }

.sidebar-footer {
    font-size: 0.78rem; color: #6e7681 !important; text-align: center;
    padding-top: 1rem; border-top: 1px solid #30363d; margin-top: 1rem;
}
/* ── NUCLEAR FILE UPLOADER DARK FIX ── */
div[data-testid="stFileUploader"] div {
    background-color: #1c2128 !important;
    color: #c9d1d9 !important;
}
div[data-testid="stFileUploader"] div div {
    background-color: #21262d !important;
    color: #58a6ff !important;
}
div[data-testid="stFileUploader"] p {
    color: #c9d1d9 !important;
}
div[data-testid="stFileUploader"] span {
    color: #58a6ff !important;
    font-weight: 600 !important;
}
div[data-testid="stFileUploader"] small {
    color: #8b949e !important;
}
div[data-testid="stFileUploader"] li {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 4px 8px !important;
    color: #58a6ff !important;
    list-style: none !important;
}
div[data-testid="stFileUploader"] li span {
    color: #58a6ff !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
div[data-testid="stFileUploader"] input {
    color: #c9d1d9 !important;
    background: #1c2128 !important;
}

/* Force ALL text inside uploader to be visible */
div[data-testid="stFileUploader"] *:not(button):not(svg) {
    color: #c9d1d9 !important;
    background-color: transparent !important;
}

/* File row specifically */
div[data-testid="stFileUploader"] section > div > div {
    background: #21262d !important;
    border-radius: 8px !important;
    border: 1px solid #30363d !important;
    padding: 4px 8px !important;
}

/* White input box override */
div[data-testid="stFileUploaderDropzone"] {
    background-color: #1c2128 !important;
}
div[data-testid="stFileUploaderDropzone"] * {
    background-color: transparent !important;
    color: #c9d1d9 !important;
}
/* ── CHAT INPUT NUCLEAR FIX ── */
div[data-testid="stChatInput"] {
    background-color: #1c2128 !important;
    border: 1px solid #7c3aed !important;
    border-radius: 14px !important;
}
div[data-testid="stChatInput"] > div {
    background-color: #1c2128 !important;
    border-radius: 14px !important;
}
div[data-testid="stChatInput"] textarea {
    background-color: #1c2128 !important;
    color: #f0f6fc !important;
    caret-color: #7c3aed !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stChatInput"] textarea::placeholder {
    color: #6e7681 !important;
    opacity: 1 !important;
}

/* Submit arrow button */
div[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
}
div[data-testid="stChatInput"] button:hover {
    opacity: 0.85 !important;
}

/* Remove red error border */
div[data-testid="stChatInput"] > div > div {
    border: none !important;
    box-shadow: none !important;
    background-color: #1c2128 !important;
}

/* ALL children inside chat input */
div[data-testid="stChatInput"] * {
    background-color: transparent !important;
}
div[data-testid="stChatInput"] textarea {
    background-color: #1c2128 !important;
    color: #f0f6fc !important;
}
/* ── CHAT MESSAGE TEXT NUCLEAR FIX ── */
div[data-testid="stChatMessage"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 14px !important;
}
div[data-testid="stChatMessage"] * {
    color: #e6edf3 !important;
}
div[data-testid="stChatMessage"] p {
    color: #e6edf3 !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}

/* User message slightly different */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #1c2128 !important;
    border-color: #7c3aed !important;
}

/* Assistant message */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #161b22 !important;
    border-color: #30363d !important;
}

/* Fix markdown inside chat */
div[data-testid="stChatMessage"] .stMarkdown p {
    color: #e6edf3 !important;
}
div[data-testid="stChatMessage"] .stMarkdown li {
    color: #e6edf3 !important;
}
div[data-testid="stChatMessage"] .stMarkdown strong {
    color: #f0f6fc !important;
    font-weight: 700 !important;
}
div[data-testid="stChatMessage"] .stMarkdown code {
    background: #21262d !important;
    color: #79c0ff !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}

/* Expander inside chat */
div[data-testid="stChatMessage"] [data-testid="stExpander"] {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
div[data-testid="stChatMessage"] [data-testid="stExpander"] * {
    color: #8b949e !important;
}

/* Bottom white area fix */
.stApp > div {
    background-color: #0d1117 !important;
}
section.main {
    background-color: #0d1117 !important;
}
section.main > div {
    background-color: #0d1117 !important;
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = 0

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-header-icon">📋</div>
        <div class="sidebar-header-text">Document Manager</div>
    </div>
    <div class="sidebar-subtitle">
        Upload and index your Data Science notes in seconds.
    </div>
    """, unsafe_allow_html=True)

    # Status
    if st.session_state.chain:
        doc_count = st.session_state.docs_loaded
        label = f"{doc_count} document(s) indexed" if doc_count > 0 else "Index loaded from disk"
        st.markdown(f"""
        <div class="status-card">
            <div class="status-icon status-icon-green">✅</div>
            <div>
                <p class="status-title">{label}</p>
                <p class="status-sub">Chatbot is ready to answer</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card">
            <div class="status-icon status-icon-warn">📁</div>
            <div>
                <p class="status-title">No documents indexed</p>
                <p class="status-sub">Upload PDF notes to get started</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Upload
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">☁️</span>
        <span class="section-title">Upload PDF Notes</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drag & drop your PDFs here or click to browse\nMax 200MB per file • PDF only",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="visible"
    )

    if st.button("🗄️  Index Uploaded PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one PDF file.")
        else:
            try:
                all_chunks = []
                progress = st.progress(0, text="Starting...")
                for i, uploaded_file in enumerate(uploaded_files):
                    # Validate file size
                    if uploaded_file.size == 0:
                        st.error(f"❌ {uploaded_file.name} is empty. Skipping.")
                        continue
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        text = load_pdf(tmp_path)
                        if not text.strip():
                            st.warning(
                                f"⚠️ {uploaded_file.name} has no readable text. "
                                "It may be a scanned image PDF."
                            )
                            os.unlink(tmp_path)
                            continue
                        chunks = split_text_into_chunks(
                            text, source_name=uploaded_file.name
                        )
                        all_chunks.extend(chunks)
                        os.unlink(tmp_path)
                        progress.progress(
                            (i + 1) / len(uploaded_files),
                            text=f"✅ Processed: {uploaded_file.name}"
                        )
                    except Exception as e:
                        st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
                        continue

                if not all_chunks:
                    st.error(
                        "❌ No text could be extracted. "
                        "Please upload text-based PDFs."
                    )
                else:
                    with st.spinner("🔨 Building AI knowledge base..."):
                        vector_store = build_vector_store(all_chunks)
                        retriever = get_retriever(vector_store)
                        st.session_state.chain = build_rag_chain(retriever)
                        st.session_state.docs_loaded = len(uploaded_files)
                    st.success(
                        f"✅ Ready! {len(all_chunks)} chunks from "
                        f"{len(uploaded_files)} file(s)."
                    )
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Indexing failed: {e}")
                st.info("Try restarting the app or re-uploading the files.")

    # Load /data
    st.markdown("""
    <div class="section-header" style="margin-top:1.4rem">
        <span class="section-icon">📂</span>
        <span class="section-title">Load from /data Folder</span>
    </div>
    <div class="section-desc">
        Load and index PDF files from the local /data directory.
    </div>
    """, unsafe_allow_html=True)

    if st.button("📁  Load PDFs from /data", use_container_width=True):
        data_path = "./data"
        try:
            if not os.path.exists(data_path):
                st.error("❌ /data folder does not exist. Create it first.")
            elif not any(f.endswith(".pdf") for f in os.listdir(data_path)):
                st.error("❌ No PDF files found in /data folder.")
            else:
                all_chunks = []
                with st.spinner("📂 Loading PDFs from /data..."):
                    pdf_texts = load_all_pdfs_from_folder(data_path)
                    for filename, text in pdf_texts.items():
                        if not text.strip():
                            st.warning(f"⚠️ {filename} has no readable text.")
                            continue
                        chunks = split_text_into_chunks(
                            text, source_name=filename
                        )
                        all_chunks.extend(chunks)
                if not all_chunks:
                    st.error("❌ No text extracted from any PDF.")
                else:
                    vector_store = build_vector_store(all_chunks)
                    retriever = get_retriever(vector_store)
                    st.session_state.chain = build_rag_chain(retriever)
                    st.session_state.docs_loaded = len(pdf_texts)
                    st.success(f"✅ Loaded {len(pdf_texts)} PDF(s)!")
                    st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to load PDFs: {e}")

    # Load saved index
    # Load saved index - always visible
    st.markdown("""
    <div class="section-header" style="margin-top:1rem">
        <span class="section-icon">⚡</span>
        <span class="section-title">Load Previous Session</span>
    </div>
    <div class="section-desc">
        Already indexed before? Load your saved knowledge base instantly.
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚡  Load Saved Index", use_container_width=True):
        try:
            with st.spinner("Loading saved index..."):
                vector_store = load_vector_store()
                if vector_store:
                    retriever = get_retriever(vector_store)
                    st.session_state.chain = build_rag_chain(retriever)
                    st.success("✅ Saved index loaded!")
                    st.rerun()
                else:
                    st.error(
                        "❌ No saved index found. "
                        "Please index your PDFs first."
                    )
        except Exception as e:
            st.error(f"❌ Failed to load index: {e}")

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️  Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("""
    <div class="sidebar-footer">
        🔒 Secure &nbsp;•&nbsp; 🔐 Private &nbsp;•&nbsp; 💻 Local
    </div>
    """, unsafe_allow_html=True)

# ── MAIN AREA ─────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="header-card">
    <div style="font-size:5rem; line-height:1;">🤖</div>
    <div>
        <div class="header-title">
            <span class="purple">Data Science</span>
            <span class="white"> Notes Chatbot</span>
        </div>
        <p class="header-subtitle">
            Premium RAG assistant for PDF notes.<br>
            Upload, index, and ask grounded questions with source tracing.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Feature cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon fi-purple">📄</div>
        <div>
            <p class="feature-title">Upload PDFs</p>
            <p class="feature-desc">Add your Data Science notes in PDF format</p>
        </div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon fi-green">🗄️</div>
        <div>
            <p class="feature-title">Smart Indexing</p>
            <p class="feature-desc">We index and understand your documents</p>
        </div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon fi-blue">💬</div>
        <div>
            <p class="feature-title">Ask Questions</p>
            <p class="feature-desc">Get accurate answers from your notes</p>
        </div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon fi-orange">📖</div>
        <div>
            <p class="feature-title">Source Tracing</p>
            <p class="feature-desc">See sources and references for every answer</p>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

# Welcome or chat
if not st.session_state.chat_history:
    st.markdown("""
    <div class="welcome-box">
        <div style="font-size:4rem; margin-bottom:1rem;">💬</div>
        <div class="welcome-title">Start a conversation</div>
        <div class="welcome-sub">
            Upload your PDF notes and start asking questions!
        </div>
    </div>
    """, unsafe_allow_html=True)

# Chat history
for message in st.session_state.chat_history:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input(
    "Ask a question about your Data Science notes..."
)

if user_input:
    if not st.session_state.chain:
        st.warning(
            "⚠️ No documents indexed yet! "
            "Please upload PDFs and click Index first."
        )
        st.stop()

    # Validate input
    if len(user_input.strip()) < 3:
        st.warning("⚠️ Question too short. Please ask a complete question.")
        st.stop()

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("assistant", avatar="🤖"):
        try:
            with st.spinner("🔍 Searching your notes..."):
                response = ask_question(st.session_state.chain, user_input)

            answer = response["answer"]
            sources = response["sources"]
            st.markdown(answer)

            if sources:
                with st.expander("📎 View Sources"):
                    for source in sources:
                        st.markdown(f"- 📄 `{source}`")

            full_response = answer
            if sources:
                full_response += f"\n\n*📎 Sources: {', '.join(sources)}*"

        except Exception as e:
            answer = "❌ Sorry, something went wrong while answering."
            full_response = answer
            st.error(f"Error: {e}")
            st.info(
                "Try rephrasing your question or re-indexing your documents."
            )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": full_response
    })
