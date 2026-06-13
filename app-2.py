import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langsmith import traceable

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro HR Help Desk",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 0.95rem; }
    .source-badge {
        background: #e8f4f8;
        border-left: 3px solid #2d6a9f;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #333;
        margin: 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── API Keys ────────────────────────────────────────────────────────────────
def get_secret(key):
    val = os.environ.get(key, "")
    if not val:
        try:
            val = st.secrets.get(key, "")
        except Exception:
            pass
    return val

GROQ_API_KEY      = get_secret("GROQ_API_KEY")
LANGCHAIN_API_KEY = get_secret("LANGCHAIN_API_KEY")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]    = "zyro-rag-challenge"

CORPUS_PATH = "./hr_docs/"

# ─── Prompts ─────────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert HR assistant for Zyro Dynamics Pvt. Ltd.
Answer the employee question accurately using ONLY the information from the
retrieved HR policy documents below.

STRICT RULES:
- Answer ONLY using the context provided. Do not use any outside knowledge.
- Include specific numbers, days, percentages, and policy names when available.
- Do NOT make up information. If context is insufficient, say so clearly.
- Keep your answer concise, factual, and easy to understand.

CONTEXT FROM POLICY DOCUMENTS:
{context}

EMPLOYEE QUESTION: {question}

ANSWER:""")

OOS_PROMPT = ChatPromptTemplate.from_template("""
You are a classifier. Determine if the following question is related to
HR policies, employment, workplace rules, leave, compensation, benefits,
performance, conduct, or other HR topics at a company.

Respond with ONLY one word: "IN_SCOPE" or "OUT_OF_SCOPE".

IN_SCOPE: leave, salary, WFH, performance reviews, benefits, onboarding,
separation, travel expenses, POSH, code of conduct, IT security, payroll.

OUT_OF_SCOPE: cooking, sports, entertainment, general knowledge, weather,
programming tutorials, medical diagnoses, news, geography.

QUESTION: {question}

CLASSIFICATION:""")

REFUSAL_MESSAGE = (
    "I'm sorry, but I can only answer HR-related questions from "
    "Zyro Dynamics policy documents. Your question is outside the scope "
    "of HR policies. Please ask about leave, compensation, WFH, performance "
    "reviews, benefits, onboarding, separation, travel expenses, code of "
    "conduct, POSH, or IT security policies."
)

# ─── Cached RAG Pipeline ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading HR policy documents...")
def build_rag_pipeline():
    # Load PDFs
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    docs   = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # Embeddings — use HuggingFaceEmbeddings with explicit import
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
    )

    # LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512
    )

    return retriever, llm, len(docs), len(chunks)


def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown").split("/")[-1]
        page   = doc.metadata.get("page", "?")
        parts.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


@traceable(name="zyro-ask-bot", project_name="zyro-rag-challenge")
def ask_bot(question, retriever, llm):
    # Classify
    cls_val  = OOS_PROMPT.invoke({"question": question})
    cls_resp = llm.invoke(cls_val)
    cls      = StrOutputParser().invoke(cls_resp).strip().upper()

    if "OUT_OF_SCOPE" in cls:
        return {"answer": REFUSAL_MESSAGE, "sources": [], "is_out_of_scope": True}

    # RAG
    retrieved = retriever.invoke(question)
    context   = format_docs(retrieved)
    prompt    = RAG_PROMPT.invoke({"context": context, "question": question})
    response  = llm.invoke(prompt)
    answer    = StrOutputParser().invoke(response)

    sources = []
    for doc in retrieved:
        src = doc.metadata.get("source", "Unknown").split("/")[-1]
        pg  = doc.metadata.get("page", "?")
        cit = f"{src} (p.{pg})"
        if cit not in sources:
            sources.append(cit)

    return {"answer": answer, "sources": sources, "is_out_of_scope": False}


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 Zyro Dynamics")
    st.markdown("### HR Help Desk")
    st.divider()
    st.markdown("""
**I can answer questions about:**

📅 Leave & Attendance  
💰 Salary & Benefits  
🏠 Work From Home  
📊 Performance Reviews  
🤝 Onboarding & Separation  
✈️ Travel & Expenses  
🛡️ POSH & Code of Conduct  
💻 IT & Data Security
    """)
    st.divider()
    if not GROQ_API_KEY:
        st.warning("⚠️ GROQ_API_KEY not set in Streamlit Secrets.")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Powered by: LangChain · FAISS · Groq LLaMA 3.3 · LangSmith")

# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏢 Zyro Dynamics HR Help Desk</h1>
    <p>AI-powered assistant for HR policy questions. Answers grounded in official policy documents.</p>
</div>
""", unsafe_allow_html=True)

if not GROQ_API_KEY:
    st.error("Please set GROQ_API_KEY in Streamlit Secrets (Settings → Secrets).")
    st.stop()

# Load pipeline
retriever, llm, doc_count, chunk_count = build_rag_pipeline()

col1, col2, col3 = st.columns(3)
col1.metric("📄 Policy Documents", "11")
col2.metric("🔍 Knowledge Chunks", chunk_count)
col3.metric("🎯 Retrieval", "MMR k=6")
st.divider()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 **Hello! I'm the Zyro Dynamics HR Help Desk assistant.**\n\nAsk me anything about company HR policies!",
        "sources": [],
        "is_out_of_scope": False
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f'<div class="source-badge">📄 {src}</div>', unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Ask an HR policy question..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching policy documents..."):
            result = ask_bot(prompt, retriever, llm)
        st.markdown(result["answer"])
        if result.get("sources"):
            with st.expander("📄 Sources", expanded=False):
                for src in result["sources"]:
                    st.markdown(f'<div class="source-badge">📄 {src}</div>', unsafe_allow_html=True)
        if result.get("is_out_of_scope"):
            st.info("ℹ️ Out-of-scope question — please ask about HR policies.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "is_out_of_scope": result.get("is_out_of_scope", False)
    })
