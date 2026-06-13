import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langsmith import traceable

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro HR Help Desk",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Environment / API Keys ─────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", st.secrets.get("LANGCHAIN_API_KEY", ""))

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

CORPUS_PATH = "./hr_docs/"   # Folder containing the 11 HR PDFs

# ─── Constants ───────────────────────────────────────────────────────────────
REFUSAL_MESSAGE = (
    "I'm sorry, but I can only answer HR-related questions based on "
    "Zyro Dynamics' internal policy documents. Your question appears to be "
    "outside the scope of HR policies. Please ask about topics like leave, "
    "compensation, WFH, performance reviews, benefits, onboarding, separation, "
    "travel expenses, code of conduct, POSH, or IT security policies."
)

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

IN_SCOPE examples: leave policy, salary, WFH, performance reviews, benefits,
onboarding, separation, travel expenses, POSH, code of conduct, IT security.

QUESTION: {question}

CLASSIFICATION:""")

# ─── Cached RAG Pipeline ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading HR policy documents and building knowledge base...")
def build_rag_pipeline():
    """Load docs, build FAISS index, and return retriever + LLM."""
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512
    )

    return retriever, llm

def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown").split("/")[-1]
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

@traceable(name="zyro-ask-bot", project_name="zyro-rag-challenge")
def ask_bot(question: str, retriever, llm) -> dict:
    # Classify
    cls_prompt = OOS_PROMPT.invoke({"question": question})
    cls_resp = llm.invoke(cls_prompt)
    classification = StrOutputParser().invoke(cls_resp).strip().upper()

    if "OUT_OF_SCOPE" in classification:
        return {"answer": REFUSAL_MESSAGE, "sources": [], "is_out_of_scope": True}

    # RAG
    retrieved_docs = retriever.invoke(question)
    context = format_docs(retrieved_docs)
    prompt_val = RAG_PROMPT.invoke({"context": context, "question": question})
    response = llm.invoke(prompt_val)
    answer = StrOutputParser().invoke(response)

    sources = []
    for doc in retrieved_docs:
        src = doc.metadata.get("source", "Unknown").split("/")[-1]
        page = doc.metadata.get("page", "?")
        cit = f"{src} (p.{page})"
        if cit not in sources:
            sources.append(cit)

    return {"answer": answer, "sources": sources, "is_out_of_scope": False}

# ─── UI ──────────────────────────────────────────────────────────────────────
# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1e3a5f/white?text=Zyro+Dynamics", width=200)
    st.title("HR Help Desk")
    st.markdown("""
    **Welcome!** I can answer questions about:
    - 📅 Leave & Attendance
    - 💰 Salary & Benefits
    - 🏠 Work From Home
    - 📊 Performance Reviews
    - 🤝 Onboarding & Separation
    - ✈️ Travel & Expenses
    - 🛡️ POSH & Code of Conduct
    - 💻 IT Security
    """)
    st.divider()
    st.caption("Powered by Zyro Dynamics HR Policy Corpus")
    st.caption("RAG · FAISS · Groq LLaMA 3.3")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main
st.title("🏢 Zyro Dynamics HR Help Desk")
st.markdown("Ask me anything about Zyro Dynamics HR policies and I'll answer based on official documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Hello! I'm the Zyro Dynamics HR Help Desk assistant. Ask me anything about company policies — leave, benefits, WFH, performance reviews, and more!",
        "sources": []
    }]

# Load pipeline
retriever, llm = build_rag_pipeline()

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources", expanded=False):
                for src in msg["sources"]:
                    st.caption(f"• {src}")

# Chat input
if prompt := st.chat_input("Ask an HR question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            result = ask_bot(prompt, retriever, llm)

        st.markdown(result["answer"])

        if result.get("sources"):
            with st.expander("📄 Sources", expanded=False):
                for src in result["sources"]:
                    st.caption(f"• {src}")

        if result.get("is_out_of_scope"):
            st.info("ℹ️ This question is outside HR policy scope.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", [])
    })