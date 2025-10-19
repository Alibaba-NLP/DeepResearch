# tool_memory.py
# A lightweight conversation + page memory tool with SQLite FTS5 retrieval.
# Usable both as a callable tool and as a programmatic logger from the agent.

import os
import json
import sqlite3
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

try:
    # Align with your existing tool base class
    from qwen_agent.tools import BaseTool
except Exception:
    # Fallback if BaseTool shape changes; we only need a .name and .call
    class BaseTool:
        name: str = "memory"
        description: str = ""
        def call(self, params: Dict[str, Any], **kwargs) -> str:
            raise NotImplementedError

DB_PATH = os.getenv("AGENT_MEMORY_DB", "agent_memory.db")

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS conv (
  id INTEGER PRIMARY KEY,
  session_id TEXT,
  question_id TEXT,
  role TEXT,                -- system | user | assistant | tool
  content TEXT,
  created_at TEXT           -- ISO8601
);

CREATE VIRTUAL TABLE IF NOT EXISTS conv_fts USING fts5(
  content, role, content='conv', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS conv_ai AFTER INSERT ON conv BEGIN
  INSERT INTO conv_fts(rowid, content, role) VALUES (new.id, new.content, new.role);
END;
CREATE TRIGGER IF NOT EXISTS conv_au AFTER UPDATE ON conv BEGIN
  INSERT INTO conv_fts(conv_fts, rowid, content, role) VALUES('delete', old.id, old.content, old.role);
  INSERT INTO conv_fts(rowid, content, role) VALUES (new.id, new.content, new.role);
END;
CREATE TRIGGER IF NOT EXISTS conv_ad AFTER DELETE ON conv BEGIN
  INSERT INTO conv_fts(conv_fts, rowid, content, role) VALUES('delete', old.id, old.content, old.role);
END;

CREATE TABLE IF NOT EXISTS pages (
  id INTEGER PRIMARY KEY,
  session_id TEXT,
  question_id TEXT,
  url TEXT,
  title TEXT,
  content TEXT,
  snippet TEXT,
  fetched_at TEXT,          -- ISO8601
  meta_json TEXT,
  content_hash TEXT UNIQUE  -- url+content hash for idempotent upsert
);

CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
  title, content, snippet, url, content='pages', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
  INSERT INTO pages_fts(rowid, title, content, snippet, url)
  VALUES (new.id, new.title, new.content, new.snippet, new.url);
END;
CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE ON pages BEGIN
  INSERT INTO pages_fts(pages_fts, rowid, title, content, snippet, url) VALUES('delete', old.id, old.title, old.content, old.snippet, old.url);
  INSERT INTO pages_fts(rowid, title, content, snippet, url)
  VALUES (new.id, new.title, new.content, new.snippet, new.url);
END;
CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
  INSERT INTO pages_fts(pages_fts, rowid, title, content, snippet, url)
  VALUES('delete', old.id, old.title, old.content, old.snippet, old.url);
END;
"""

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _hash_url_content(url: Optional[str], content: str) -> str:
    h = hashlib.sha1()
    h.update((url or "").encode("utf-8"))
    h.update(b"||")
    h.update(content.encode("utf-8"))
    return h.hexdigest()

def _as_str(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

class MemoryTool(BaseTool):
    """
    Tool name: "memory"

    Actions:
      - save_message: {session_id, question_id, role, content}
      - save_page:    {session_id, question_id, url?, title?, content, snippet?, meta?}
      - retrieve:     {query, limit? (default 8)}
      - list_recent:  {session_id?, question_id?, limit?}
      - clear:        {session_id?, question_id?}  # dangerous; use sparingly

    Returns JSON strings for tool friendliness.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.name = "memory"
        self.description = "Persist and retrieve prior conversation and accessed pages."
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._ensure()

    # ---------- Public tool API ----------

    def call(self, params: Dict[str, Any], **kwargs) -> str:
        action = (params or {}).get("action", "").lower()
        try:
            if action == "save_message":
                rec = self.log_message(
                    params.get("session_id"), params.get("question_id"),
                    params.get("role"), params.get("content")
                )
                return json.dumps({"status": "ok", "action": action, "id": rec}, ensure_ascii=False)

            if action == "save_page":
                rec = self.log_page(
                    params.get("session_id"), params.get("question_id"),
                    params.get("url"), params.get("title"),
                    params.get("content"), params.get("snippet"), params.get("meta")
                )
                return json.dumps({"status": "ok", "action": action, "id": rec}, ensure_ascii=False)

            if action == "retrieve":
                query = params.get("query", "").strip()
                limit = int(params.get("limit", 8))
                items = self.retrieve(query, limit=limit)
                return json.dumps({"status": "ok", "action": action, "items": items}, ensure_ascii=False)

            if action == "list_recent":
                items = self.list_recent(
                    session_id=params.get("session_id"),
                    question_id=params.get("question_id"),
                    limit=int(params.get("limit", 12)),
                )
                return json.dumps({"status": "ok", "action": action, "items": items}, ensure_ascii=False)

            if action == "clear":
                self.clear(
                    session_id=params.get("session_id"),
                    question_id=params.get("question_id"),
                )
                return json.dumps({"status": "ok", "action": action}, ensure_ascii=False)

            return json.dumps({"status": "error", "error": f"unknown action: {action}"}, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

    # ---------- Programmatic helpers for the agent ----------

    def log_message(self, session_id: str, question_id: str, role: str, content: str) -> int:
        if not content:
            return -1
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO conv(session_id, question_id, role, content, created_at) VALUES(?,?,?,?,?)",
            (session_id, question_id, role, content, _now_iso()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def log_page(
        self, session_id: str, question_id: str,
        url: Optional[str], title: Optional[str],
        content: str, snippet: Optional[str] = None, meta: Optional[Dict[str, Any]] = None
    ) -> int:
        if not content:
            return -1
        content_hash = _hash_url_content(url, content)
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        snip = snippet or (content[:280] + ("…" if len(content) > 280 else ""))
        cur = self._conn.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO pages(session_id, question_id, url, title, content, snippet, fetched_at, meta_json, content_hash)
            VALUES(?,?,?,?,?,?,?,?,?)
        """, (session_id, question_id, url, title, content, snip, _now_iso(), meta_json, content_hash))
        self._conn.commit()
        cur.execute("SELECT id FROM pages WHERE content_hash=?", (content_hash,))
        row = cur.fetchone()
        return int(row[0]) if row else -1

    def retrieve(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        if not query:
            return []
        cur = self._conn.cursor()
        # Search both corpora and merge by BM25 (lower is better)
        cur.execute("""
            SELECT 'conv' AS type, c.id, c.role, NULL AS title, NULL AS url, c.content, c.created_at AS ts, bm25(conv_fts) AS score
            FROM conv_fts f JOIN conv c ON c.id=f.rowid
            WHERE conv_fts MATCH ?
            UNION ALL
            SELECT 'page' AS type, p.id, NULL AS role, p.title, p.url, p.content, p.fetched_at AS ts, bm25(pages_fts) AS score
            FROM pages_fts f2 JOIN pages p ON p.id=f2.rowid
            WHERE pages_fts MATCH ?
            ORDER BY score ASC LIMIT ?
        """, (query, query, limit))
        rows = cur.fetchall()
        items = []
        for r in rows:
            items.append({
                "type": r[0],
                "id": r[1],
                "role": r[2],
                "title": r[3],
                "url": r[4],
                "snippet": (r[5] or "")[:320],
                "timestamp": r[6],
                "score": r[7],
            })
        return items

    def list_recent(self, session_id: Optional[str], question_id: Optional[str], limit: int = 12) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        items: List[Dict[str, Any]] = []
        if session_id or question_id:
            where = []
            vals = []
            if session_id:
                where.append("session_id=?")
                vals.append(session_id)
            if question_id:
                where.append("question_id=?")
                vals.append(question_id)
            w = " AND ".join(where) if where else "1=1"

            cur.execute(f"SELECT id, role, content, created_at FROM conv WHERE {w} ORDER BY id DESC LIMIT ?", (*vals, limit))
            items += [{"type": "conv", "id": r[0], "role": r[1], "snippet": (r[2] or "")[:320], "timestamp": r[3]} for r in cur.fetchall()]
            cur.execute(f"SELECT id, title, url, snippet, fetched_at FROM pages WHERE {w} ORDER BY id DESC LIMIT ?", (*vals, limit))
            items += [{"type": "page", "id": r[0], "title": r[1], "url": r[2], "snippet": r[3], "timestamp": r[4]} for r in cur.fetchall()]
        else:
            cur.execute("SELECT id, role, content, created_at FROM conv ORDER BY id DESC LIMIT ?", (limit,))
            items += [{"type": "conv", "id": r[0], "role": r[1], "snippet": (r[2] or "")[:320], "timestamp": r[3]} for r in cur.fetchall()]
            cur.execute("SELECT id, title, url, snippet, fetched_at FROM pages ORDER BY id DESC LIMIT ?", (limit,))
            items += [{"type": "page", "id": r[0], "title": r[1], "url": r[2], "snippet": r[3], "timestamp": r[4]} for r in cur.fetchall()]
        # Return most recent first, capped
        items = sorted(items, key=lambda x: x["timestamp"] or "", reverse=True)[:limit]
        return items

    def clear(self, session_id: Optional[str], question_id: Optional[str]) -> None:
        cur = self._conn.cursor()
        if session_id or question_id:
            where = []
            vals = []
            if session_id:
                where.append("session_id=?")
                vals.append(session_id)
            if question_id:
                where.append("question_id=?")
                vals.append(question_id)
            w = " AND ".join(where)
            cur.execute(f"DELETE FROM conv WHERE {w}", tuple(vals))
            cur.execute(f"DELETE FROM pages WHERE {w}", tuple(vals))
        else:
            cur.execute("DELETE FROM conv")
            cur.execute("DELETE FROM pages")
        self._conn.commit()

    # ---------- Internal ----------

    def _ensure(self):
        with self._conn:
            self._conn.executescript(SCHEMA)
