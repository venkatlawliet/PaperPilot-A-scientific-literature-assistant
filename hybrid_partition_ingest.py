from __future__ import annotations
import os, re, uuid, logging, json, time, tempfile
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from supabase_client import (
    save_paper_chunks,
    get_chunks_for_paper,
    get_user_by_id,
    save_bm25_state,
    load_bm25_state,
)
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
from landingai_ade import LandingAIADE
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "researchmcp")
PINECONE_DIMENSION = 768
PINECONE_METRIC = "dotproduct"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
client = LandingAIADE(apikey=os.environ.get("VISION_AGENT_API_KEY"))
text_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
@dataclass
class Part:
    text: str
    page: int
    type: str
    caption: Optional[str] = None
    extra: Optional[Dict] = None
REF_HEADERS_RE = re.compile(
    r"^\s*(references?|bibliography|works\s+cited|literature\s+cited)\s*:?\s*$",
    re.I,
)
INLINE_NUM_CIT_RE = re.compile(
    r"\[\s*(?:\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+)*)\s*\]"
)
_pc_client: Optional[Pinecone] = None
_pinecone_index = None
def bm25_to_state(texts: List[str]) -> Dict[str, Any]:
    return {"texts": texts}
def bm25_from_state(state: Dict[str, Any]) -> BM25Encoder:#fit with bm25 state(it is also text anyway) loaded from supabase
    texts = state.get("texts") or []
    if not texts:
        raise RuntimeError("bm25_state has no texts.")
    bm25 = BM25Encoder()
    bm25.fit(texts)
    return bm25
def build_bm25_from_chunks(user_id: str, paper_id: int) -> BM25Encoder: #reffitng bm25 when user loads for history papers(fit with directly stored text in supabase)
    chunks = get_chunks_for_paper(user_id, paper_id)
    texts: List[str] = []
    for c in chunks:
        t = (c.get("text") or "").strip()
        if t:
            texts.append(t)
    if texts:
        bm25 = BM25Encoder()
        bm25.fit(texts)
        return bm25
    state = load_bm25_state(user_id, paper_id)
    if state:
        return bm25_from_state(state)
    raise RuntimeError(
        "No text chunks or BM25 state found for this paper. Re-ingest required."
    )
def _get_pc() -> Pinecone:
    global _pc_client
    if _pc_client is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not set.")
        _pc_client = Pinecone(api_key=api_key)
    return _pc_client
def get_or_create_index():
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index
    pc = _get_pc()
    names = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in names:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return _pinecone_index
def clean_text(text: str) -> str:
    text = INLINE_NUM_CIT_RE.sub("", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "", text)
    return text.strip()
def extract_parts_from_url(pdf_url: str) -> List[Part]:
    try:
        retries = 3
        response = None
        for attempt in range(retries):
            try:
                response = client.parse(document_url=pdf_url, model="dpt-2-latest") #We pass the research paper's URL to ADE for extraction
                break
            except Exception as e:
                if attempt == retries - 1:
                    raise
                logger.warning(f"Retrying ADE ({attempt+1}/{retries}): {e}")
                time.sleep(5)
        ade_json = json.loads(response.model_dump_json())
        content_list = ade_json.get("chunks") or ade_json.get("content") or []
        if not content_list:
            raise ValueError("ADE returned 0 chunks (PDF may be scanned or invalid).")
        parts = []
        for item in content_list:
            text = item.get("markdown") or item.get("text", "")
            if not text.strip():
                continue
            parts.append(
                Part(
                    text=clean_text(text),
                    page=item.get("grounding", {}).get("page", 1),
                    type=item.get("type", "text"),
                    caption=None,
                    extra=item.get("grounding", {}),
                )
            )
        logger.info(f"ADE extracted {len(parts)} chunks.")
        return parts
    except Exception as e:
        logger.error(f"ADE extraction failed: {e}")
        raise
def extract_parts_from_file(file_bytes: bytes) -> List[Part]:
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        retries = 3
        response = None
        for attempt in range(retries):
            try:
                response = client.parse(document=file_bytes, model="dpt-2-latest")
                break
            except Exception as e:
                if attempt == retries - 1:
                    raise
                logger.warning(f"Retrying ADE ({attempt+1}/{retries}): {e}")
                time.sleep(5)
        os.unlink(tmp_path)
        ade_json = json.loads(response.model_dump_json())
        content_list = ade_json.get("chunks") or ade_json.get("content") or []
        if not content_list:
            raise ValueError("ADE returned 0 chunks (PDF may be scanned or invalid).")
        parts = []
        for item in content_list:
            text = item.get("markdown") or item.get("text", "")
            if not text.strip():
                continue
            parts.append(
                Part(
                    text=clean_text(text),
                    page=item.get("grounding", {}).get("page", 1),
                    type=item.get("type", "text"),
                    caption=None,
                    extra=item.get("grounding", {}),
                )
            )
        logger.info(f"ADE extracted {len(parts)} chunks from uploaded file.")
        return parts
    except Exception as e:
        logger.error(f"ADE extraction from file failed: {e}")
        raise
def ingest_paper_from_file(
    file_bytes: bytes,
    user_id: str,
    namespace: str,
    paper_id: int,
    paper_title: str,
) -> Dict[str, Any]:
    index = get_or_create_index()
    parts = extract_parts_from_file(file_bytes)
    parts = [p for p in parts if p.text.strip()]
    if not parts:
        raise RuntimeError("ADE could not extract any text from the uploaded PDF.")
    texts = [p.text for p in parts]
    vecs = text_model.encode(texts, batch_size=32, show_progress_bar=True)
    metas = create_meta(parts, user_id, paper_id, paper_title)
    ids = [f"{paper_id}-{uuid.uuid4()}" for _ in range(len(vecs))]
    save_paper_chunks(user_id, paper_id, metas)
    docs = [Document(page_content=p.text, metadata={"page": p.page}) for p in parts]
    bm25, sparse_vectors = sparse_fit_and_encode(docs)
    state = bm25_to_state(texts)
    save_bm25_state(user_id, paper_id, state)
    hybrid_vectors = []
    for i in range(len(vecs)):
        s = sparse_vectors[i]
        s = {
            "indices": s["indices"].tolist() if hasattr(s["indices"], "tolist") else s["indices"],
            "values": s["values"].tolist() if hasattr(s["values"], "tolist") else s["values"]
        }
        hybrid_vectors.append({
            "id": ids[i],
            "values": vecs[i].tolist(),
            "sparse_values": s,
            "metadata": metas[i],
        })
    index.upsert(vectors=hybrid_vectors, namespace=namespace)
    logger.info(f"Ingested uploaded '{paper_title}' — {len(hybrid_vectors)} vectors.")
    return {"parts": parts, "bm25": bm25, "num_vectors": len(hybrid_vectors)}
def create_meta(parts: List[Part], user_id: str, paper_id: int, paper_title: str):
    out = []
    for p in parts:
        grounding_page = None
        if isinstance(p.extra, dict):
            grounding_page = p.extra.get("page")
        out.append({
            "user_id": user_id,
            "paper_id": float(paper_id),
            "paper_title": paper_title,
            "page": p.page,
            "type": p.type,
            "caption": p.caption or "",
            "text": p.text,
            "grounding": grounding_page if grounding_page else p.page,
        })
    return out
def sparse_fit_and_encode(chunks: List[Document]):
    texts = [c.page_content for c in chunks]
    bm25 = BM25Encoder()
    bm25.fit(texts)
    sparse_vectors = bm25.encode_documents(texts)
    return bm25, sparse_vectors
def weight_by_alpha(sparse, dense, alpha: float):
    if not sparse or "indices" not in sparse or "values" not in sparse:
        sparse = {"indices": [], "values": []}
    return (
        {
            "indices": sparse["indices"],
            "values": [v * (1 - alpha) for v in sparse["values"]],
        },
        [v * alpha for v in dense],
    )
def query_ade_index(
    query: str,
    bm25: BM25Encoder,
    dense_model,
    namespace: str,
    paper_id: int,
    top_k: int = 5,
    alpha: float = 0.6,
):
    index = get_or_create_index()
    try:
        stats = index.describe_index_stats()
    except Exception as e:
        print("Error getting stats:", e)
    if bm25 is None:
        raise RuntimeError("No BM25 loaded for this paper. Missing bm25_state?")
    q_dense = dense_model.encode([query])[0].tolist()
    q_sparse = bm25.encode_queries([query])[0]
    sq, dq = weight_by_alpha(q_sparse, q_dense, alpha)
    if "indices" not in sq or "values" not in sq:
        sq = {"indices": [], "values": []}
    return index.query(
        vector=dq,
        sparse_vector=sq,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
        filter={"paper_id": {"$eq": float(paper_id)}},
    )
PRIMER = (
    "You are a Q&A bot. Answer ONLY from the text below. "
    "If the answer is not present, say \"I don't know.\" "
    "Cite the page number if possible.\n\n"
)
def build_llm_context(
    user_id: str,
    paper_id: int,
    question: str,
    bm25: BM25Encoder,
    top_k: int = 5,
    alpha: float = 0.6,
) -> str:
    user = get_user_by_id(user_id)
    if isinstance(user, dict):
        namespace = user.get("namespace") #each user has a unique namespace and in each namespace we store multiple papers of the user with paper_id
    else:
        namespace = getattr(user, "namespace", None)
    if not namespace:
        raise RuntimeError("User namespace not found for build_llm_context.")
    results = query_ade_index(
        query=question,
        bm25=bm25,
        dense_model=text_model,
        namespace=namespace,
        paper_id=paper_id,
        top_k=top_k,
        alpha=alpha,
    )
    if hasattr(results, "to_dict"):
        data = results.to_dict()
    elif isinstance(results, dict):
        data = results
    else:
        data = {"matches": getattr(results, "matches", [])}
    lines = []
    for m in data.get("matches", []):
        if isinstance(m, dict):
            meta = m.get("metadata", {}) or {}
            score = m.get("score", 0.0)
        else:
            meta = getattr(m, "metadata", {}) or {}
            score = getattr(m, "score", 0.0)
        snippet = (meta.get("text") or "").strip()
        if not snippet:
            continue
        lines.append(
            f"[Page {meta.get('page')} | {meta.get('type')}]\n"
            f"{snippet}\n"
            f"(Score: {score:.3f})\n"
        )
    if not lines:
        return PRIMER + "No relevant context found for this question."
    return PRIMER + "\n".join(lines)
def ingest_paper_for_user(
    pdf_url: str,
    user_id: str,
    namespace: str,
    paper_id: int,
    paper_title: str,
) -> Dict[str, Any]:
    index = get_or_create_index()
    parts = extract_parts_from_url(pdf_url)
    parts = [p for p in parts if p.text.strip()]
    if not parts:
        raise RuntimeError(
            "ADE is not able to extract any text from the provided PDF."
        )
    texts = [p.text for p in parts]
    vecs = text_model.encode(texts, batch_size=32, show_progress_bar=True)
    metas = create_meta(parts, user_id, paper_id, paper_title)
    ids = [f"{paper_id}-{uuid.uuid4()}" for _ in range(len(vecs))]
    save_paper_chunks(user_id, paper_id, metas)
    docs = [Document(page_content=p.text, metadata={"page": p.page}) for p in parts]
    bm25, sparse_vectors = sparse_fit_and_encode(docs)
    state = bm25_to_state(texts)
    save_bm25_state(user_id, paper_id, state)
    hybrid_vectors = []
    for i in range(len(vecs)):
        s = sparse_vectors[i]
        s = {
            "indices": s["indices"].tolist() if hasattr(s["indices"], "tolist") else s["indices"],
            "values": s["values"].tolist() if hasattr(s["values"], "tolist") else s["values"]
        }
        hybrid_vectors.append(
            {
                "id": ids[i],
                "values": vecs[i].tolist(),
                "sparse_values": s,
                "metadata": metas[i],
            }
        )
    index.upsert(vectors=hybrid_vectors, namespace=namespace)
    logger.info(
        f"Ingested '{paper_title}' — {len(hybrid_vectors)} vectors stored "
        f"under namespace '{namespace}'."
    )
    return {
        "parts": parts,
        "bm25": bm25,
        "num_vectors": len(hybrid_vectors),
    }