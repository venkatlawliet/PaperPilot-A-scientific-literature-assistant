#The face of the application
from __future__ import annotations
import os
import json
import streamlit as st
from groq import Groq
from claude_mcp_client import ClaudeMCPClient
from s2_client import search_papers
from content_resolver import resolve_pdf_url_from_s2_item
from d2_utils import llm_generate_d2, render_d2_to_svg
from llm_bridge import answer_with_llama, answer_with_claude
from mcp_integration import WebSearchClient
from supabase_client import (
    create_user,
    authenticate_user,
    create_paper,
    list_papers_for_user,
    append_chat_turn,
    get_chat_history,
)
from hybrid_partition_ingest import (
    ingest_paper_for_user,
    ingest_paper_from_file,
    build_llm_context,
    build_bm25_from_chunks,
)
st.set_page_config(page_title="ResearchMCP-The new", page_icon="🪐", layout="wide")
st.markdown(
    """
<style>
[data-testid="stAppViewContainer"], html, body {
    background-color: #000 !important;
    color: #fff !important;
}
[data-testid="stHeader"] {background: #000 !important;}
h1,h2,h3,h4,h5,h6,label,p,span,div {color: #fff !important;}
textarea, input[type="text"] {
    background: #111 !important;
    color: #fff !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
}
.stButton>button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.6rem 1rem !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.stButton>button:hover {
    background-color: #1e40af !important;
}
.block-container {padding-top: 4rem !important;}
.login-block {margin-top: 1.5rem;}
</style>
""",
    unsafe_allow_html=True,
)
def exit_paper_mode():
    st.session_state.update({
        "mode": "general",
        "paper_ingested": False,
        "paper_id": None,
        "paper_title": None,
        "bm25": None,
        "memory": st.session_state.get("general_memory", []).copy(),
    })
def save_to_memory(question: str, answer: str, source_type: str, d2_code: str = None):
    st.session_state["memory"].append({
        "question": question,
        "answer": answer,
        "d2_code": d2_code,
        "source_type": source_type,
    })
def ensure_bm25():
    if "bm25" not in st.session_state or st.session_state["bm25"] is None:
        st.session_state["bm25"] = build_bm25_from_chunks(
            st.session_state["user_id"],
            st.session_state["paper_id"]
        )
    return st.session_state["bm25"]
def new_groq_rewrite(user_query: str) -> dict:
    system_prompt = """
You are a query rewriting expert in a hybrid search + D2 diagram generation pipeline.
PURPOSE OF YOUR ROLE:
Some user queries (such as "generate a diagram of the architecture") cannot be
directly used for vector database retrieval, because they do not appear as literal
text in the paper. If sent as-is, the vector search would return irrelevant or noisy
chunks. Therefore, your job is to rewrite these diagram-generation queries into
factual, information-seeking forms (e.g., "Describe the architecture used in the paper")
so the system can retrieve the correct chunks before generating diagrams.
Your responsibilities:
1. Determine whether the user's query requires rewriting to enable accurate chunk
   retrieval from the vector database for diagram/architecture generation.
2. Rewrite ONLY when the user requests:
      - a diagram
      - an architecture drawing
      - a pipeline visualization
      - a flowchart
      - any structural or image-like representation
3. When rewriting, preserve the user's intent while turning the query into a form
   suitable for semantic + keyword retrieval.
4. If rewriting is not needed, return the original query EXACTLY unchanged.

CRITICAL RULES:
- DO NOT rewrite normal factual questions (e.g., values, datasets, tables, methods,
  equations, metrics, terminology).
- DO NOT rewrite conversational or general questions.
- DO NOT assume or guess missing details.
- DO NOT merge with past conversation history.
- DO NOT change pronouns or meaning.
- DO NOT generate diagrams yourself. Your task is ONLY to rewrite queries for retrieval.

Rewriting Logic:
- If the user asks to "draw", "generate", "create", "sketch", "visualize", "illustrate",
  or produce any kind of diagram/architecture → set `needs_rewriting = true` and rewrite
  the query into a descriptive, retrieval-friendly information request.
- Otherwise → `needs_rewriting = false` and the query remains unchanged.

OUTPUT FORMAT (STRICT):
Return ONLY this JSON (nothing before or after):

{
  "needs_rewriting": true/false,
  "rewritten_query": "..."
}
"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"needs_rewriting": False, "rewritten_query": user_query}
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            model="llama-3.1-8b-instant",
            max_tokens=256,
            temperature=0,
        )
        raw = resp.choices[0].message.content
        return json.loads(raw)
    except Exception:
        return {"needs_rewriting": False, "rewritten_query": user_query}
def _extract_claude_text(response: dict) -> str:
    if not isinstance(response, dict):
        return "No clear answer found."
    content = response.get("content", []) or []
    texts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    joined = "\n".join(t.strip() for t in texts if t.strip())
    return joined or "No clear answer found."
DEFAULT_KEYS = [
    "user_id",
    "username",
    "namespace",
    "memory",
    "general_memory", 
    "paper_id",
    "paper_title",
    "pdf_url",
    "paper_ingested",
    "s2_results",
    "show_research_form",
]
for k in DEFAULT_KEYS:
    st.session_state.setdefault(k, None)
if st.session_state["memory"] is None:
    st.session_state["memory"] = []
if st.session_state["general_memory"] is None:
    st.session_state["general_memory"] = []
if "mode" not in st.session_state:
    st.session_state["mode"] = "general"
if st.session_state["show_research_form"] is None:
    st.session_state["show_research_form"] = False
if st.session_state["user_id"] is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Login to ResearchQuesta")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        c1, c2 = st.columns(2)
        if c1.button("Login"):
            if not u or not p:
                st.error("Missing credentials")
            else:
                res = authenticate_user(u, p)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.session_state.update(
                        {
                            "user_id": res["id"],
                            "username": res["username"],
                            "namespace": res["namespace"],
                        }
                    )
                    st.rerun()
        if c2.button("Register"):
            if not u or not p:
                st.error("Missing credentials")
            else:
                res = create_user(u, p)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.session_state.update(
                        {
                            "user_id": res["id"],
                            "username": res["username"],
                            "namespace": res["namespace"],
                        }
                    )
                    st.success("Account created.")
                    st.rerun()
    st.stop()
top1, top2 = st.columns([3, 1])
with top1:
    st.title("ResearchMCP")
with top2:
    st.markdown(f"Logged in as **{st.session_state['username']}**")
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()
st.markdown("---")
with st.expander("Your Ingested Papers", expanded=False):
    user_papers = list_papers_for_user(st.session_state["user_id"])
    if not user_papers:
        st.info("No papers ingested yet.")
    else:
        options = {f"{p['title']} ({str(p['created_at'])[:10]})": p for p in user_papers}
        selected = st.selectbox("Select a paper:", list(options.keys()))
        paper = options[selected]
        if st.button("Load Paper"):
            st.session_state["paper_id"] = paper["id"]
            st.session_state["paper_title"] = paper["title"]
            st.session_state["pdf_url"] = paper["pdf_url"]
            st.session_state["paper_ingested"] = True
            try:
                bm25 = build_bm25_from_chunks(
                    st.session_state["user_id"],
                    st.session_state["paper_id"]
                )
                st.session_state["bm25"] = bm25
            except Exception as e:
                st.error(f"BM25 rebuild failed: {e}")
                st.session_state["bm25"] = None

            st.session_state["memory"] = get_chat_history(
                user_id=st.session_state["user_id"],
                paper_id=paper["id"],
                limit=50,
            )
            st.session_state["mode"] = "research"
            st.success(f"Loaded: {paper['title']}")
            st.rerun()
if st.session_state["mode"] == "research" and st.session_state.get("paper_ingested"):
    st.info(f"**Active Paper:** {st.session_state['paper_title']}")
    if st.button("Exit Paper Mode"):
        exit_paper_mode()
        st.rerun()
else:
    st.info("**General Chat Mode** — Ask anything or search for a research paper")
MainClaude = ClaudeMCPClient(request_timeout=60)
st.markdown("---")
st.subheader("Chat")
chat_container = st.container()
with chat_container:
    for turn in st.session_state["memory"]:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            source_type = turn.get("source_type", "knowledge")
            if source_type == "paper":
                st.caption("From paper")
            elif source_type == "web":
                st.caption("From web search")
            elif source_type == "knowledge":
                st.caption("From knowledge")
            elif source_type == "system":
                st.caption("System")
            if turn.get("d2_code"):
                try:
                    svg_path = render_d2_to_svg(turn["d2_code"])
                    st.image(svg_path, width=800)
                except Exception:
                    st.write(turn.get("answer", "Diagram generated."))
            else:
                st.write(turn["answer"])
if st.session_state.get("show_research_form"):
    st.markdown("---")
    st.subheader("Research Paper Lookup")
    tab_search, tab_upload = st.tabs(["Search Paper", "Upload Paper"])
    with tab_search:
        with st.form("research_lookup_form"):
            paper_title_input = st.text_input("Paper title or keywords:")
            #paper_question_input = st.text_area("Your question about the paper:", height=100)
            submitted = st.form_submit_button("Search Papers")
        if submitted and paper_title_input:
            with st.spinner("Searching Semantic Scholar..."):
                results = search_papers(paper_title_input)
                st.session_state["paper_ingested"] = False
                st.session_state["paper_id"] = None
                st.session_state["s2_results"] = results
                #st.session_state["paper_question"] = paper_question_input
                st.session_state["show_research_form"] = False
            st.rerun()
    with tab_upload:
        with st.form("upload_pdf_form"):
            uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
            default_title = ""
            if uploaded_file:
                default_title = os.path.splitext(uploaded_file.name)[0]
            upload_title = st.text_input("Paper Title:", value=default_title)
            upload_submitted = st.form_submit_button("Upload & Ingest")
        if upload_submitted:
            if not uploaded_file:
                st.error("Please select a PDF file.")
            elif not upload_title.strip():
                st.error("Please enter a title.")
            else:
                with st.spinner(f"Ingesting '{upload_title}'..."):
                    try:
                        file_bytes = uploaded_file.read()
                        paper_row = create_paper(
                            user_id=st.session_state["user_id"],
                            title=upload_title.strip(),
                            pdf_url=f"uploaded://{uploaded_file.name}",
                        )
                        paper_id = paper_row["id"]
                        result = ingest_paper_from_file(
                            file_bytes=file_bytes,
                            user_id=st.session_state["user_id"],
                            namespace=st.session_state["namespace"],
                            paper_id=paper_id,
                            paper_title=upload_title.strip(),
                        )
                        if st.session_state["mode"] == "general":
                            st.session_state["general_memory"] = st.session_state["memory"].copy()
                        st.session_state.update({
                            "paper_id": paper_id,
                            "paper_title": upload_title.strip(),
                            "pdf_url": f"uploaded://{uploaded_file.name}",
                            "paper_ingested": True,
                            "bm25": result["bm25"],
                            "s2_results": None,
                            "show_research_form": False,
                            "mode": "research",
                            "memory": [],
                        })
                        st.success(f"Paper ingested: {upload_title}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to ingest: {e}") 
if st.session_state.get("s2_results"):
    results = st.session_state["s2_results"]
    st.markdown("---")
    st.subheader("Search Results from Semantic Scholar")
    if not results:
        st.error("No papers found. Try different keywords.")
        if st.button("Try Again"):
            st.session_state["s2_results"] = None
            st.session_state["show_research_form"] = True
            st.rerun()
    else:
        titles = [r["title"] for r in results]
        idx = st.selectbox(
            "Select a paper to ingest:",
            range(len(titles)),
            format_func=lambda i: titles[i]
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Ingest This Paper"):
                chosen = results[idx]
                pdf_url = resolve_pdf_url_from_s2_item(chosen)
                if not pdf_url:
                    st.error("Could not find PDF URL for this paper.")
                else:
                    with st.spinner(f"Ingesting '{chosen['title']}'..."):
                        paper_row = create_paper(
                            user_id=st.session_state["user_id"],
                            title=chosen["title"],
                            pdf_url=pdf_url,
                        )
                        paper_id = paper_row["id"]
                        result = ingest_paper_for_user(
                            pdf_url=pdf_url,
                            user_id=st.session_state["user_id"],
                            namespace=st.session_state["namespace"],
                            paper_id=paper_id,
                            paper_title=chosen["title"],
                        )
                        if st.session_state["mode"] == "general":
                            st.session_state["general_memory"] = st.session_state["memory"].copy()
                        st.session_state.update({
                            "paper_id": paper_id,
                            "paper_title": chosen["title"],
                            "pdf_url": pdf_url,
                            "paper_ingested": True,
                            "bm25": result["bm25"],
                            "s2_results": None,
                            "chosen_idx": None,
                            "paper_question": None,
                            "mode": "research",
                            "memory": [],  
                        })
                    
                    st.success(f"Paper ingested: {chosen['title']}")
                    st.rerun()
        with col2:
            if st.button("Cancel"):
                st.session_state["s2_results"] = None
                st.rerun()
if st.session_state["mode"] == "research" and st.session_state.get("paper_ingested"):
    placeholder = f"Ask about '{st.session_state['paper_title']}' or anything else..."
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Exit Paper", key="exit_bottom"):
            exit_paper_mode()
            st.rerun()
else:
    placeholder = "Ask anything..."
user_input = st.chat_input(placeholder)
if user_input:
    user_input = user_input.strip()
    if not user_input:
        st.stop()
    if (st.session_state["mode"] == "research" 
        and st.session_state.get("paper_ingested") 
        and st.session_state.get("paper_id")):
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Searching paper..."):
                    try:
                        bm25 = ensure_bm25()
                        rewrite = new_groq_rewrite(user_input)
                        if rewrite["needs_rewriting"]:
                            retrieval_q = rewrite["rewritten_query"] or user_input
                            context = build_llm_context(
                                st.session_state["user_id"],
                                st.session_state["paper_id"],
                                retrieval_q,
                                bm25,
                            )
                            d2_result = llm_generate_d2(context, user_input)
                            d2_code = d2_result.get("d2_code", "").strip()
                            if not d2_code:
                                answer = "Diagram generation failed — no valid D2 code returned."
                                st.error(answer)
                                d2_code = None
                            else:
                                svg_path = render_d2_to_svg(d2_code)
                                st.image(svg_path, width=800)
                                answer = "Diagram generated from paper context."
                        else:
                            context = build_llm_context(
                                st.session_state["user_id"],
                                st.session_state["paper_id"],
                                user_input,
                                bm25,
                            )
                            answer = answer_with_claude(
                                context_text=context,
                                question=user_input,
                                model="claude-3-haiku-20240307",
                                max_tokens=1024,
                            )
                            st.write(answer)
                            d2_code = None
                        save_to_memory(user_input, answer, "paper", d2_code)
                        append_chat_turn(
                            user_id=st.session_state["user_id"],
                            paper_id=st.session_state["paper_id"],
                            question=user_input,
                            answer=answer,
                            d2_code=d2_code,
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
        st.rerun()
    else:
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        claude_input = {
                            "original_user_query": user_input,
                            "retrieval_query": user_input,
                        }
                        recent_history = []
                        for turn in st.session_state["memory"][-5:]:
                            recent_history.append({"role": "user", "content": turn["question"]})
                            recent_history.append({"role": "assistant", "content": turn["answer"]})
                        resp = MainClaude.send_message(
                            json.dumps(claude_input),
                            conversation_history=recent_history
                        )
                        tool = resp.get("__tool_name")
                        payload = resp.get("__tool_payload")
                        if os.getenv("DEBUG_MODE") == "true":
                            with st.expander("Debug Info", expanded=False):
                                st.json(resp)
                        if not tool or tool == "direct_answer":
                            answer = _extract_claude_text(resp)
                            st.write(answer)
                            save_to_memory(user_input, answer, "knowledge")
                        elif tool == "web_search":
                            query = (payload or {}).get("query", "").strip()
                            if not query:
                                query = user_input
                            search_client = WebSearchClient()
                            results = search_client.search(query, count=5)
                            if not results:
                                answer = "No web results found for your query."
                                st.warning(answer)
                            else:
                                context_chunks = []
                                for r in results:
                                    context_chunks.append(
                                        f"Title: {r.title}\nURL: {r.url}\nSummary: {r.description}"
                                    )
                                web_context = "\n\n".join(context_chunks)
                                answer = answer_with_llama(
                                    context_text=web_context,
                                    question=f"Using the web search context above, answer: {user_input}",
                                    model="llama-3.1-8b-instant",
                                    max_tokens=512,
                                )
                                st.write(answer)
                                with st.expander("Sources", expanded=False):
                                    for r in results:
                                        st.markdown(f"**{r.title}**")
                                        if r.url:
                                            st.caption(r.url)
                                        if r.description:
                                            st.write(r.description)
                                        st.markdown("---")
                            save_to_memory(user_input, answer, "web")
                        elif tool == "research_lookup":
                            answer = "Research mode activated! Please provide paper details in the form below."
                            st.info(answer)
                            st.session_state["show_research_form"] = True
                            st.session_state["mode"] = "research"
                            save_to_memory(user_input, answer, "system")
                        else:
                            answer = f"Received unknown tool response: {tool}"
                            st.warning(answer)
                            save_to_memory(user_input, answer, "system")
                    except Exception as e:
                        error_msg = f"Error: {e}"
                        st.error(error_msg)
                        save_to_memory(user_input, error_msg, "system")
        st.rerun()
st.markdown("---")
col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
if "confirm_clear" not in st.session_state:
    st.session_state["confirm_clear"] = False
with col1:
    if not st.session_state["confirm_clear"]:
        if st.button("Clear Chat"):
            st.session_state["confirm_clear"] = True
            st.rerun()
    else:
        if st.button("Yes, Clear"):
            st.session_state["memory"] = []
            st.session_state["last_response"] = None
            st.session_state["bm25"] = None
            st.session_state["show_research_form"] = False
            st.session_state["confirm_clear"] = False
            st.rerun()
with col2:
    if st.session_state["confirm_clear"]:
        if st.button("Cancel"):
            st.session_state["confirm_clear"] = False
            st.rerun()
with col3:
    if st.button("Reset All"):
        user_id = st.session_state["user_id"]
        username = st.session_state["username"]
        namespace = st.session_state["namespace"]
        st.session_state.clear()
        st.session_state["user_id"] = user_id
        st.session_state["username"] = username
        st.session_state["namespace"] = namespace
        st.session_state["memory"] = []
        st.session_state["general_memory"] = []
        st.session_state["mode"] = "general"
        st.session_state["show_research_form"] = False
        st.session_state["confirm_clear"] = False
        st.rerun()
