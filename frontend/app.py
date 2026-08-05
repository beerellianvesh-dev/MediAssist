"""Streamlit chat interface for MedAssist RAG."""

import sys
import time
from pathlib import Path

import requests
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

st.set_page_config(page_title="MedAssist RAG", page_icon="💊", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []


def call_health() -> dict | None:
    try:
        response = requests.get(f"{settings.api_base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def call_stats() -> dict | None:
    try:
        response = requests.get(f"{settings.api_base_url}/stats", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def call_query(question: str, top_k: int, provider: str) -> dict:
    response = requests.post(
        f"{settings.api_base_url}/query",
        json={"question": question, "top_k": top_k, "provider": provider},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.header("💊 MedAssist RAG")
    st.caption("Medical Document Q&A powered by RAG over openFDA drug labels.")

    st.subheader("System Status")
    health = call_health()
    if health:
        status_icon = "🟢" if health["status"] == "ok" else "🟠"
        st.write(f"{status_icon} API: {health['status']}")
        st.write(f"📚 Vector DB documents: {health['vector_db_count']}")
        model_icon = "🟢" if health["model_status"] == "available" else "🔴"
        st.write(f"{model_icon} Ollama: {health['model_status']}")
    else:
        st.error("⚠️ Cannot reach API. Is the backend running?")

    st.divider()
    st.subheader("Settings")
    provider = st.selectbox("LLM Provider", ["ollama", "groq", "openai", "anthropic"], index=0)
    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=15, value=5)

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("MedAssist RAG — Medical Document Q&A")
st.caption(
    "Answers are generated only from ingested FDA drug label data. "
    "This is not a substitute for professional medical advice."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📎 Sources ({len(message['sources'])}) · "
                              f"{message.get('response_time_ms', 0):.0f}ms · "
                              f"{message.get('model_used', '')} · "
                              f"confidence {message.get('confidence', 0):.2f}"):
                for src in message["sources"]:
                    st.markdown(f"- **{src['drug_name']}** — {src['section_name']}")
                    if src.get("source_url"):
                        st.markdown(f"  [{src['source_url']}]({src['source_url']})")

if question := st.chat_input("Ask a question about an ingested drug label..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = call_query(question, top_k=top_k, provider=provider)
                st.markdown(result["answer"])
                if result.get("sources"):
                    with st.expander(
                        f"📎 Sources ({len(result['sources'])}) · "
                        f"{result['response_time_ms']:.0f}ms · "
                        f"{result['model_used']} · "
                        f"confidence {result['confidence']:.2f}"
                    ):
                        for src in result["sources"]:
                            st.markdown(f"- **{src['drug_name']}** — {src['section_name']}")
                            if src.get("source_url"):
                                st.markdown(f"  [{src['source_url']}]({src['source_url']})")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "response_time_ms": result.get("response_time_ms", 0),
                        "model_used": result.get("model_used", ""),
                        "confidence": result.get("confidence", 0),
                    }
                )
            except requests.HTTPError as exc:
                detail = exc.response.json().get("detail", str(exc)) if exc.response is not None else str(exc)
                st.error(f"Error: {detail}")
            except requests.RequestException as exc:
                st.error(f"Could not reach the API at {settings.api_base_url}: {exc}")
