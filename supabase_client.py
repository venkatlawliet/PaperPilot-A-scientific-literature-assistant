#here we manage everything related to supabase and pinecone
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import os
import json
import bcrypt
from typing import Optional, List, Dict, Any
from supabase import create_client, Client
from pinecone import Pinecone
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "researchmcp")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_ANON_KEY missing.")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
_pc = None
_index = None
def _pinecone():
    global _pc, _index
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY missing.")
    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    if _index is None:
        existing = _pc.list_indexes().names()
        if PINECONE_INDEX_NAME not in existing:
            raise RuntimeError(
                f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist. "
                f"Create it first in hybrid_partition_ingest.py."
            )
        _index = _pc.Index(PINECONE_INDEX_NAME)
    return _index
def _safe_namespace(username: str) -> str:
    safe = username.lower().strip().replace(" ", "_")
    safe = "".join(c for c in safe if (c.isalnum() or c == "_"))
    return f"user_{safe}"
def create_user(username: str, password: str) -> Dict[str, Any]:
    if len(username) < 3:
        return {"error": "Username must be at least 3 characters."}
    if len(password) < 4:
        return {"error": "Password must be at least 4 characters."}
    existing = supabase.table("users").select("id").eq("username", username).execute()
    if existing.data:
        return {"error": "Username already exists."}
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode("utf-8", "ignore")
    namespace = _safe_namespace(username)
    result = supabase.table("users").insert({
        "username": username,
        "hashed_password": hashed,
        "namespace": namespace,
    }).execute()
    if not result.data:
        return {"error": "Failed to create user."}
    user = result.data[0]
    return {
        "id": user["id"],
        "username": user["username"],
        "namespace": user["namespace"],
        "created_at": user.get("created_at"),
    }
def authenticate_user(username: str, password: str) -> Dict[str, Any]:
    result = (
        supabase.table("users")
        .select("*")
        .eq("username", username)
        .execute()
    )
    if not result.data:
        return {"error": "User not found."}
    user = result.data[0]
    if not bcrypt.checkpw(password.encode(), user["hashed_password"].encode()):
        return {"error": "Invalid password."}
    return {
        "id": user["id"],
        "username": user["username"],
        "namespace": user["namespace"],
    }
def list_papers_for_user(user_id: str) -> List[Dict[str, Any]]:
    res = (
        supabase.table("papers")
        .select("id, title, pdf_url, created_at, status")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []
def get_chat_history(
    user_id: str,
    paper_id: int,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    res = (
        supabase.table("paper_chats")
        .select("question, answer, d2_code, svg_path,source_type")
        .eq("user_id", user_id)
        .eq("paper_id", paper_id)
        .order("created_at")
        .limit(limit)
        .execute()
    )
    return res.data or []
def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    res = (
        supabase.table("users")
        .select("id, username, namespace, created_at")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None
def save_bm25_state(user_id: str, paper_id: int, state: Dict[str, Any]):
    supabase.table("papers").update({
        "bm25_state": json.dumps(state)
    }).eq("id", paper_id).eq("user_id", user_id).execute()
def load_bm25_state(user_id: str, paper_id: int) -> Optional[Dict[str, Any]]:
    res = (
        supabase.table("papers")
        .select("bm25_state")
        .eq("user_id", user_id)
        .eq("id", paper_id)
        .single()
        .execute()
    )
    if not res.data:
        return None
    raw = res.data.get("bm25_state")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except:
            return None
    return raw
def create_paper(user_id: str, title: str, pdf_url: str):
    # if not pdf_url.startswith("http"):
    #     raise RuntimeError("Invalid PDF URL provided.")
    res = (
        supabase.table("papers")
        .insert({
            "user_id": user_id,
            "title": title,
            "pdf_url": pdf_url,
            "status": "ingested",
        })
        .execute()
    )
    if not res.data:
        raise RuntimeError("Failed to insert paper")
    return res.data[0]
def save_paper_chunks(user_id: str, paper_id: int, chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        return
    rows = [
        {
            "user_id": user_id,
            "paper_id": paper_id,
            "page": c.get("page"),
            "type": c.get("type"),
            "caption": c.get("caption") or "",
            "text": c.get("text") or "",
            "grounding": c.get("grounding"),
        }
        for c in chunks
    ]
    batch_size = 500
    for i in range(0, len(rows), batch_size):
        supabase.table("paper_chunks").insert(rows[i: i + batch_size]).execute()
def get_chunks_for_paper(user_id: str, paper_id: int) -> List[Dict[str, Any]]:
    res = (
        supabase.table("paper_chunks")
        .select("page, type, caption, text, grounding")
        .eq("user_id", user_id)
        .eq("paper_id", paper_id)
        .order("id")
        .execute()
    )
    return res.data or []
def append_chat_turn(
    user_id: str,
    paper_id: int,
    question: str,
    answer: str,
    d2_code: Optional[str] = None,
    svg_path: Optional[str] = None,
    source_type: str = "paper", 
) -> None:
    supabase.table("paper_chats").insert(
        {
            "user_id": user_id,
            "paper_id": paper_id,
            "question": question,
            "answer": answer,
            "d2_code": d2_code,
            "svg_path": svg_path,
            "source_type": source_type, 
        }
    ).execute()