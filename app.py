'''
import os
import time
import tempfile
import subprocess
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# Config
# -----------------------------
load_dotenv()
DEFAULT_MODEL = os.getenv("MODEL_NAME", "llama3.2")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

st.set_page_config(page_title="Ollama Pro Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Ollama 3.2 — Pro Chatbot (LangChain + RAG)")

# -----------------------------
# Helpers
# -----------------------------
def list_ollama_models() -> List[str]:
    """Return installed Ollama models (best-effort)."""
    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
        lines = [ln.strip().split()[0] for ln in out.splitlines()[1:] if ln.strip()]
        return lines or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]

def ensure_ollama_running():
    """Light ping to check if Ollama responds."""
    try:
        _ = subprocess.check_output(["ollama", "--version"], text=True)
    except Exception:
        st.warning("⚠️ Ollama CLI not found on PATH. Install from https://ollama.com/ and restart.")
    # If serve isn’t running, LangChain will still try; we show tip in sidebar.

def chunk_docs(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def build_vectorstore(files) -> FAISS | None:
    """Create a FAISS vector store from uploaded files using Ollama embeddings."""
    if not files:
        return None

    docs = []
    temp_dir = tempfile.mkdtemp()
    for f in files:
        path = os.path.join(temp_dir, f.name)
        with open(path, "wb") as w:
            w.write(f.read())

        if f.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        else:
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())

    chunks = chunk_docs(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # make sure it's pulled
    vs = FAISS.from_documents(chunks, embeddings)
    return vs

def retrieve_context(vs: FAISS, query: str, k: int = 4) -> str:
    if not vs:
        return ""
    docs = vs.similarity_search(query, k=k)
    joined = "\n\n".join([d.page_content for d in docs])
    return joined

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    ensure_ollama_running()

    installed = list_ollama_models()
    model_name = st.selectbox("Model", installed, index=installed.index(DEFAULT_MODEL) if DEFAULT_MODEL in installed else 0)
    temperature = st.slider("Temperature", 0.0, 1.5, DEFAULT_TEMPERATURE, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
    system_prompt = st.text_area("System Prompt", value="You are a helpful, concise assistant.", height=90)

    st.markdown("---")
    st.subheader("📚 RAG (Docs Q&A)")
    use_rag = st.checkbox("Enable RAG with uploaded files", value=True)
    uploaded_files = st.file_uploader("Upload PDF/TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    build_index = st.button("Build/Refresh Index")

    st.markdown("---")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
        st.success("Chat cleared.")

    st.markdown("---")
    st.caption("Tip: In a separate terminal, run `ollama serve`.\nPull models with `ollama pull llama3.2` and `ollama pull nomic-embed-text`.")

# -----------------------------
# Session State
# -----------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vs" not in st.session_state:
    st.session_state.vs = None

# Build/refresh vector store
if build_index:
    with st.spinner("Embedding & indexing documents…"):
        st.session_state.vs = build_vectorstore(uploaded_files)
        if st.session_state.vs:
            st.success("✅ Index ready.")
        else:
            st.info("No files uploaded, RAG disabled.")

# -----------------------------
# LLM + Prompt
# -----------------------------
llm = ChatOllama(
    model=model_name,
    temperature=temperature,
    top_p=top_p,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    (
        "user",
        "Use the provided context only if relevant.\n\n<context>\n{context}\n</context>\n\n"
        "Conversation so far (you may use it for continuity):\n{history}\n\n"
        "User: {question}"
    ),
])

parser = StrOutputParser()

# -----------------------------
# Chat UI
# -----------------------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("Type your message…")

if user_input:
    # show user bubble
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # RAG context
    context = retrieve_context(st.session_state.vs, user_input, k=4) if (use_rag and st.session_state.vs) else ""

    # brief text history for prompt (last ~6 turns)
    history_text = ""
    for m in st.session_state.messages[-12:]:
        who = "User" if m["role"] == "user" else "Assistant"
        history_text += f"{who}: {m['content']}\n"

    # Compose final chain
    chain = prompt | llm | parser

    # Stream-ish render
    placeholder = st.chat_message("assistant")
    out_box = placeholder.empty()

    start = time.time()
    # Use .invoke for simplicity (ChatOllama also supports .stream in LangChain nightly)
    answer = chain.invoke({
        "system_prompt": system_prompt,
        "context": context,
        "history": history_text,
        "question": user_input
    })
    # Render final
    out_box.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Small footer diagnostics
    st.caption(f"⏱️ {time.time() - start:.2f}s | RAG: {'on' if (use_rag and st.session_state.vs) else 'off'}")

######
import os
import io
import time
import json
import tempfile
import subprocess
from typing import List, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv

# LangChain / Ollama / RAG
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI  # for OpenAI-compatible endpoints (optional)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Web search
from duckduckgo_search import DDGS

# -----------------------------
# Setup & Config
# -----------------------------
APP_TITLE = "🤖 Ultra-Pro AI Agent (Ollama + Tools + RAG)"
INDEX_DIR = ".rag_index"  # persistent FAISS path

load_dotenv()
DEFAULT_MODEL = os.getenv("MODEL_NAME", "llama3.2")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # or your llama on provider
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

st.set_page_config(page_title="Ultra-Pro AI Agent", page_icon="🤖", layout="wide")
st.title(APP_TITLE)

# -----------------------------
# Helpers
# -----------------------------
def list_ollama_models() -> List[str]:
    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
        lines = [ln.strip().split()[0] for ln in out.splitlines()[1:] if ln.strip()]
        return lines or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]

def ensure_ollama_hint():
    try:
        _ = subprocess.check_output(["ollama", "--version"], text=True)
    except Exception:
        st.sidebar.warning("⚠️ Ollama CLI not found. Install: https://ollama.com/download")
    st.sidebar.caption("Tip: Run `ollama serve` in a separate terminal. Pull models with `ollama pull llama3.2` & `ollama pull nomic-embed-text`.")

def chunk_docs(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def load_files_to_docs(files) -> List:
    docs = []
    tmpdir = tempfile.mkdtemp()
    for f in files:
        path = os.path.join(tmpdir, f.name)
        with open(path, "wb") as w:
            w.write(f.read())
        if f.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        else:
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
    return docs

def build_or_update_index(files) -> Optional[FAISS]:
    if not files:
        return None
    docs = load_files_to_docs(files)
    chunks = chunk_docs(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(INDEX_DIR)
    return vs

def load_index_if_exists() -> Optional[FAISS]:
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None

def rag_retrieve(vs: Optional[FAISS], query: str, k: int = 4) -> Tuple[str, List[Tuple[str, str]]]:
    """Return joined context & citations [(source, snippet), ...]."""
    if not vs:
        return "", []
    docs = vs.similarity_search(query, k=k)
    context = "\n\n".join([d.page_content for d in docs])
    cites = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "document")
        page = meta.get("page", None)
        label = f"{os.path.basename(src)}" + (f" (p.{page+1})" if page is not None else "")
        snippet = (d.page_content[:300] + "…") if len(d.page_content) > 320 else d.page_content
        cites.append((label, snippet))
    return context, cites

def web_search(query: str, max_results: int = 5) -> Tuple[str, List[Tuple[str, str]]]:
    """DuckDuckGo search: return context text and [(title, url)] for citations."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "result")
                href = r.get("href") or r.get("url") or ""
                snippet = r.get("body", "")
                results.append((title, href, snippet))
    except Exception as e:
        return "", []

    # Build a lightweight context for the LLM
    lines = []
    cites = []
    for (title, url, snippet) in results:
        lines.append(f"{title}\nURL: {url}\n{snippet}")
        cites.append((title, url))
    return "\n\n".join(lines), cites

def safe_run_python(code: str, timeout_s: int = 4) -> Tuple[bool, str]:
    """Run tiny Python snippets safely in a subprocess; return (ok, output_or_error)."""
    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8")
    tmp.write(code)
    tmp.close()
    try:
        proc = subprocess.run(
            ["python", tmp.name],
            capture_output=True, text=True, timeout=timeout_s
        )
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        if proc.returncode == 0:
            return True, out or "(no output)"
        else:
            return False, err or "Execution failed."
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout_s}s."
    except Exception as e:
        return False, f"Error: {e}"

def stream_like(text: str, placeholder, delay: float = 0.0):
    """Naive streaming effect."""
    buf = ""
    for ch in text:
        buf += ch
        placeholder.markdown(buf)
        if delay:
            time.sleep(delay)

# -----------------------------
# Sidebar: Controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    ensure_ollama_hint()

    mode = st.radio("Backend", ["Ollama (local)", "OpenAI-compatible (remote)"], index=0)
    if mode == "Ollama (local)":
        models = list_ollama_models()
        model_name = st.selectbox("Model", models, index=models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0)
    else:
        model_name = st.text_input("Remote model", value=OPENAI_MODEL)

    temperature = st.slider("Temperature", 0.0, 1.5, DEFAULT_TEMPERATURE, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)

    st.markdown("---")
    st.subheader("🎭 Role Preset")
    role = st.selectbox("Select a role", ["General", "Teacher", "Coder", "Data Analyst", "Friendly Buddy"])
    role_prompts = {
        "General": "You are a helpful, concise assistant.",
        "Teacher": "You are a patient teacher. Explain step by step and include simple examples.",
        "Coder": "You are a senior Python engineer. Provide clean code with comments and edge cases.",
        "Data Analyst": "You analyze data clearly. Use bullets, formulas, and crisp insights.",
        "Friendly Buddy": "You are friendly, casual, and supportive. Keep it light but helpful."
    }
    system_prompt = st.text_area("System Prompt", value=role_prompts[role], height=110)

    st.markdown("---")
    st.subheader("🧰 Tools")
    use_rag = st.checkbox("RAG: Use uploaded files", value=True)
    use_search = st.checkbox("Web Search when useful", value=False)
    use_python = st.checkbox("Python Code Runner (dangerous code will fail)", value=False)

    uploaded_files = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)
    cols = st.columns(2)
    with cols[0]:
        if st.button("Build / Refresh Index"):
            with st.spinner("Indexing documents..."):
                vs = build_or_update_index(uploaded_files)
                if vs:
                    st.success("✅ Index ready & saved.")
                else:
                    st.info("No files uploaded; RAG disabled.")
    with cols[1]:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.memory = ConversationBufferMemory(return_messages=True)
            st.success("Chat cleared.")

    st.markdown("---")
    if st.button("Export Chat"):
        path = "chat_export.txt"
        with open(path, "w", encoding="utf-8") as f:
            for m in st.session_state.get("messages", []):
                f.write(f"{m['role'].upper()}: {m['content']}\n\n")
        st.success(f"💾 Saved: {path}")

# -----------------------------
# Session State
# -----------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vs" not in st.session_state:
    st.session_state.vs = load_index_if_exists()  # load persistent index if exists

# -----------------------------
# LLM & Prompt
# -----------------------------
def get_llm():
    if mode == "Ollama (local)":
        return ChatOllama(model=model_name, temperature=temperature, top_p=top_p)
    # OpenAI-compatible
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE
    )

llm = get_llm()
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("user",
     "Use the PROVIDED CONTEXT only if relevant.\n\n"
     "<context>\n{context}\n</context>\n\n"
     "Conversation so far (for continuity):\n{history}\n\n"
     "User: {question}")
])

# -----------------------------
# Chat display (history first)
# -----------------------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Ask anything…")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Build history text (last N turns)
    history_text = ""
    for m in st.session_state.messages[-12:]:
        who = "User" if m["role"] == "user" else "Assistant"
        history_text += f"{who}: {m['content']}\n"

    # Tools: RAG, Search, Python
    context_blocks = []
    citations = []

    if use_rag and st.session_state.vs:
        ctx, cites = rag_retrieve(st.session_state.vs, user_input, k=4)
        if ctx:
            context_blocks.append("### Documents\n" + ctx)
            citations.extend([("doc", c[0], c[1]) for c in cites])  # (type, label, snippet)

    if use_search:
        web_ctx, web_cites = web_search(user_input, max_results=5)
        if web_ctx:
            context_blocks.append("### Web\n" + web_ctx)
            citations.extend([("web", title, url) for (title, url) in web_cites])

    # Optional: detect python code fence and run (very simple heuristic)
    python_output = ""
    if use_python and ("```python" in user_input and "```" in user_input.split("```python", 1)[1]):
        try:
            code_block = user_input.split("```python", 1)[1].split("```", 1)[0]
            ok, out = safe_run_python(code_block)
            python_output = f"Python run {'OK' if ok else 'FAILED'}:\n{out}"
            context_blocks.append("### Python\n" + python_output)
        except Exception as e:
            context_blocks.append(f"### Python\nError extracting code: {e}")

    final_context = "\n\n".join(context_blocks)

    chain = prompt | llm | parser
    placeholder = st.chat_message("assistant").empty()

    start = time.time()
    answer = chain.invoke({
        "system_prompt": system_prompt,
        "context": final_context,
        "history": history_text,
        "question": user_input
    })

    # Stream-like render
    stream_like(answer, placeholder, delay=0.0)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Citations panel
    with st.expander("📎 Sources & Tools context", expanded=False):
        if not citations and not final_context:
            st.write("_No external context used._")
        else:
            if python_output:
                st.markdown("**Python Tool Output**")
                st.code(python_output)
            if citations:
                st.markdown("**References**")
                for kind, a, b in citations:
                    if kind == "doc":
                        st.write(f"• {a}")
                        st.caption(b)
                    else:
                        st.write(f"• [{a}]({b})")

    st.caption(f"⏱️ {time.time()-start:.2f}s | Backend: {mode} | Model: {model_name} | RAG: {'on' if (use_rag and st.session_state.vs) else 'off'} | Search: {'on' if use_search else 'off'}")

'''
import os
import io
import json
import time
import tempfile
import subprocess
from datetime import datetime
from typing import List, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv

# LLM & RAG
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Web search
from duckduckgo_search import DDGS

# Voice & Sentiment
import pyttsx3
import speech_recognition as sr
from textblob import TextBlob

# -----------------------------
# Setup & Config
# -----------------------------
APP_TITLE = "🤖 Ultra-Pro AI Agent (Ollama, Tools, RAG & Voice)"
INDEX_DIR = ".rag_index"

load_dotenv()
DEFAULT_MODEL = os.getenv("MODEL_NAME", "llama3.2")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
st.title(APP_TITLE)

# -----------------------------
# Helpers
# -----------------------------
def list_ollama_models() -> List[str]:
    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
        lines = [ln.strip().split()[0] for ln in out.splitlines()[1:] if ln.strip()]
        return lines or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]

def ensure_ollama_hint():
    try:
        _ = subprocess.check_output(["ollama", "--version"], text=True)
    except Exception:
        st.sidebar.warning("⚠️ Ollama CLI not found. Install: https://ollama.com/download")
    st.sidebar.caption("Tip: Run `ollama serve` in a separate terminal. Pull models with `ollama pull llama3.2` & `ollama pull nomic-embed-text`.")

def chunk_docs(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def load_files_to_docs(files) -> List:
    docs = []
    tmpdir = tempfile.mkdtemp()
    for f in files:
        path = os.path.join(tmpdir, f.name)
        with open(path, "wb") as w:
            w.write(f.read())
        if f.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        else:
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
    return docs

def build_or_update_index(files) -> Optional[FAISS]:
    if not files:
        return None
    docs = load_files_to_docs(files)
    chunks = chunk_docs(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(INDEX_DIR)
    return vs

def load_index_if_exists() -> Optional[FAISS]:
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None

def rag_retrieve(vs: Optional[FAISS], query: str, k: int = 4) -> Tuple[str, List[Tuple[str, str]]]:
    if not vs:
        return "", []
    docs = vs.similarity_search(query, k=k)
    context = "\n\n".join([d.page_content for d in docs])
    cites = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "document")
        page = meta.get("page", None)
        label = f"{os.path.basename(src)}" + (f" (p.{page+1})" if page is not None else "")
        snippet = (d.page_content[:300] + "…") if len(d.page_content) > 320 else d.page_content
        cites.append((label, snippet))
    return context, cites

def web_search(query: str, max_results: int = 5) -> Tuple[str, List[Tuple[str, str]]]:
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "result")
                href = r.get("href") or r.get("url") or ""
                snippet = r.get("body", "")
                results.append((title, href, snippet))
    except:
        return "", []

    lines = []
    cites = []
    for (title, url, snippet) in results:
        lines.append(f"{title}\nURL: {url}\n{snippet}")
        cites.append((title, url))
    return "\n\n".join(lines), cites

def safe_run_python(code: str, timeout_s: int = 4) -> Tuple[bool, str]:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8")
    tmp.write(code)
    tmp.close()
    try:
        proc = subprocess.run(
            ["python", tmp.name],
            capture_output=True, text=True, timeout=timeout_s
        )
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        if proc.returncode == 0:
            return True, out or "(no output)"
        else:
            return False, err or "Execution failed."
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout_s}s."
    except Exception as e:
        return False, f"Error: {e}"

def stream_like(text: str, placeholder, delay: float = 0.0):
    buf = ""
    for ch in text:
        buf += ch
        placeholder.markdown(buf)
        if delay:
            time.sleep(delay)

def summarize_docs(files) -> str:
    """Simple concatenation + summary (could enhance with LLM later)."""
    if not files:
        return "No files uploaded."
    docs = load_files_to_docs(files)
    full_text = "\n\n".join([d.page_content for d in docs])
    if len(full_text) > 2000:
        return full_text[:2000] + "\n…(truncated)"
    return full_text

def analyze_sentiment(text: str) -> str:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "Positive 🙂"
    elif polarity < -0.2:
        return "Negative 😟"
    else:
        return "Neutral 😐"

def speak_text(text: str):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen_to_user(timeout=5) -> str:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info(f"Listening for {timeout} seconds...")
        try:
            audio = r.listen(source, timeout=timeout)
            return r.recognize_google(audio)
        except Exception as e:
            return f"(Voice input failed: {e})"

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings & Tools")
    ensure_ollama_hint()

    models = list_ollama_models()
    model_name = st.selectbox("Model", models, index=models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0)
    temperature = st.slider("Temperature", 0.0, 1.5, DEFAULT_TEMPERATURE, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)

    st.subheader("🎭 Role Preset")
    role = st.selectbox("Select a role", ["General", "Teacher", "Coder", "Data Analyst", "Friendly Buddy"])
    role_prompts = {
        "General": "You are a helpful, concise assistant.",
        "Teacher": "You are a patient teacher. Explain step by step and include simple examples.",
        "Coder": "You are a senior Python engineer. Provide clean code with comments and edge cases.",
        "Data Analyst": "You analyze data clearly. Use bullets, formulas, and crisp insights.",
        "Friendly Buddy": "You are friendly, casual, and supportive. Keep it light but helpful."
    }
    system_prompt = st.text_area("System Prompt", value=role_prompts[role], height=110)

    st.markdown("---")
    st.subheader("🧰 Tools")
    use_rag = st.checkbox("RAG: Use uploaded files", value=True)
    use_search = st.checkbox("Web Search when useful", value=False)
    use_python = st.checkbox("Python Code Runner", value=False)
    use_voice = st.checkbox("Voice Assistant", value=False)

    uploaded_files = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Build / Refresh Index"):
        with st.spinner("Indexing documents..."):
            vs = build_or_update_index(uploaded_files)
            if vs:
                st.success("✅ Index ready & saved.")
            else:
                st.info("No files uploaded; RAG disabled.")
    if st.button("Summarize Uploaded Files"):
        summary = summarize_docs(uploaded_files)
        st.info(summary)

# -----------------------------
# State
# -----------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vs" not in st.session_state:
    st.session_state.vs = load_index_if_exists()

# -----------------------------
# LLM
# -----------------------------
def get_llm():
    return ChatOllama(model=model_name, temperature=temperature, top_p=top_p)

llm = get_llm()
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("user",
     "Use the PROVIDED CONTEXT only if relevant.\n\n"
     "<context>\n{context}\n</context>\n\n"
     "Conversation so far (for continuity):\n{history}\n\n"
     "User: {question}")
])

# -----------------------------
# Chat display
# -----------------------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# -----------------------------
# Chat input
# -----------------------------
user_input = ""
if use_voice and st.button("🎤 Speak"):
    user_input = listen_to_user()
else:
    user_input = st.chat_input("Ask anything…")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    history_text = ""
    for m in st.session_state.messages[-12:]:
        who = "User" if m["role"] == "user" else "Assistant"
        history_text += f"{who}: {m['content']}\n"

    context_blocks = []
    citations = []

    if use_rag and st.session_state.vs:
        ctx, cites = rag_retrieve(st.session_state.vs, user_input, k=4)
        if ctx:
            context_blocks.append("### Documents\n" + ctx)
            citations.extend([("doc", c[0], c[1]) for c in cites])

    if use_search:
        web_ctx, web_cites = web_search(user_input, max_results=5)
        if web_ctx:
            context_blocks.append("### Web\n" + web_ctx)
            citations.extend([("web", title, url) for (title, url) in web_cites])

    python_output = ""
    if use_python and ("```python" in user_input and "```" in user_input.split("```python", 1)[1]):
        try:
            code_block = user_input.split("```python", 1)[1].split("```", 1)[0]
            ok, out = safe_run_python(code_block)
            python_output = f"Python run {'OK' if ok else 'FAILED'}:\n{out}"
            context_blocks.append("### Python\n" + python_output)
        except Exception as e:
            context_blocks.append(f"### Python\nError extracting code: {e}")

    final_context = "\n\n".join(context_blocks)

    chain = prompt | llm | parser
    placeholder = st.chat_message("assistant").empty()

    start = time.time()
    answer = chain.invoke({
        "system_prompt": system_prompt,
        "context": final_context,
        "history": history_text,
        "question": user_input
    })

    stream_like(answer, placeholder, delay=0.0)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Sentiment
    sentiment = analyze_sentiment(answer)
    st.sidebar.metric("Conversation Sentiment", sentiment)

    # Voice
    if use_voice:
        speak_text(answer)

    # Sources & context
    with st.expander("📎 Sources & Tools context", expanded=False):
        if not citations and not final_context:
            st.write("_No external context used._")
        else:
            if python_output:
                st.markdown("**Python Tool Output**")
                st.code(python_output)
            if citations:
                st.markdown("**References**")
                for kind, a, b in citations:
                    if kind == "doc":
                        st.write(f"• {a}")
                        st.caption(b)
                    else:
                        st.write(f"• [{a}]({b})")

    st.caption(f"⏱️ {time.time()-start:.2f}s | Model: {model_name} | RAG: {'on' if (use_rag and st.session_state.vs) else 'off'} | Search: {'on' if use_search else 'off'}")
