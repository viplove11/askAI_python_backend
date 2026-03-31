import os
import json
import faiss
import re
import numpy as np
from datetime import datetime, date
from decimal import Decimal
from uuid import UUID

from sqlalchemy import create_engine, text
from groq import Groq
from dotenv import load_dotenv
from app.core.embeddings import get_embedding_service

load_dotenv()

# ─────────────────────────────────────────────
# Lazy singletons — initialized on first use,
# NOT at import time (prevents startup timeout)
# ─────────────────────────────────────────────
_groq_client = None
_db_engine = None


def _log_model_event(purpose: str, model: str, status: str, detail: str = ""):
    suffix = f" | {detail}" if detail else ""
    print(f"[MODEL][{purpose}][{status}] {model}{suffix}")


def _sql_llm_models() -> list:
    """Ordered SQL-generation model chain."""
    defaults = [
        os.getenv("SQL_LLM_MODEL", "qwen/qwen3-32b").strip(),
        os.getenv("SQL_LLM_FALLBACK_MODEL", "llama-3.3-70b-versatile").strip(),
        os.getenv("SQL_LLM_SECOND_FALLBACK_MODEL", "llama-3.1-8b-instant").strip(),
    ]
    seen = set()
    ordered = []
    for model_name in defaults:
        if model_name and model_name not in seen:
            seen.add(model_name)
            ordered.append(model_name)
    return ordered


def _format_llm_models() -> list:
    """Ordered response-formatting model chain."""
    defaults = [
        os.getenv("FORMAT_LLM_MODEL", "llama-3.3-70b-versatile").strip(),
        os.getenv("FORMAT_LLM_FALLBACK_MODEL", "llama-3.1-8b-instant").strip(),
    ]
    seen = set()
    ordered = []
    for model_name in defaults:
        if model_name and model_name not in seen:
            seen.add(model_name)
            ordered.append(model_name)
    return ordered

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def get_db_engine():
    global _db_engine
    if _db_engine is None:
        _db_engine = create_engine(os.getenv("DATABASE_URL"))
    return _db_engine


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────
def universal_serializer(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, UUID):
        return str(obj)
    return str(obj)


def _humanize_field_name(field_name: str) -> str:
    """Convert DB-style keys into clean, presentation-friendly labels."""
    if not field_name:
        return "Field"
    cleaned = re.sub(r"[_\-]+", " ", str(field_name)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.title()


def _format_lookup_value(value):
    """Render values in a readable and consistent way for lookup cards."""
    if value is None:
        return "N/A"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def format_lookup_response(records: list, title: str) -> str:
    """Build a professional markdown lookup response without table syntax."""
    sections = [f"### {title}", ""]

    for idx, record in enumerate(records, start=1):
        sections.append(f"#### Record {idx}")
        for key, value in record.items():
            label = _humanize_field_name(key)
            display_val = _format_lookup_value(value)
            sections.append(f"- **{label}:** {display_val}")
        if idx < len(records):
            sections.append("")
            sections.append("---")
            sections.append("")

    return "\n".join(sections).strip()


# ─────────────────────────────────────────────
# Intent classifier — decides response format
# ─────────────────────────────────────────────
def detect_intent(user_query: str) -> str:
    """
    Returns one of:
      - 'count'   → single number answer expected  (kitne, how many, count, total)
      - 'lookup'  → single record / detail lookup  (who is, details of, info about)
      - 'list'    → multiple records as a table    (list, show all, sabhi, dikhao)
    """
    q = user_query.lower()

    count_keywords = ["kitne", "how many", "count", "total", "kul", "sankhya", "ginti"]
    lookup_keywords = [
        "who is", "detail", "info", "pata", "address", "contact",
        "phone", "mobile", "naam", "name of", "kiska", "kaun",
    ]
    list_keywords = [
        "list", "show", "all", "sabhi", "dikhao", "suchi", "record",
        "members", "sadasya", "give me", "fetch", "report",
    ]

    if any(k in q for k in count_keywords):
        return "count"
    if any(k in q for k in lookup_keywords):
        return "lookup"
    if any(k in q for k in list_keywords):
        return "list"
    # default — treat as list
    return "list"


# ─────────────────────────────────────────────
# Main chatbot class
# ─────────────────────────────────────────────
class DatabaseChatbot:
    def __init__(self, project_path=None):
        if project_path is None:
            project_path = os.getenv("STORAGE_PATH", "./store/current_db")
        self.index = faiss.read_index(os.path.join(project_path, "vector.faiss"))
        with open(os.path.join(project_path, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        with open(os.path.join(project_path, "id_map.json"), "r") as f:
            self.id_map = json.load(f)
        self.embedder = get_embedding_service()
        _log_model_event(
            "embeddings",
            f"{self.embedder.provider}/{self.embedder.model_name}",
            "active",
        )
        self._validate_embedding_compatibility(project_path)

    def _validate_embedding_compatibility(self, project_path: str):
        info_path = os.path.join(project_path, "embedding_info.json")
        if not os.path.exists(info_path):
            print("[WARN] embedding_info.json not found. Skipping embedding compatibility check.")
            return

        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        expected_provider = info.get("provider")
        expected_model = info.get("model")
        expected_dim = info.get("vector_dim")

        if (
            expected_provider != self.embedder.provider
            or expected_model != self.embedder.model_name
        ):
            raise RuntimeError(
                "Embedding provider/model mismatch. "
                f"Index built with {expected_provider}/{expected_model}, "
                f"current config is {self.embedder.provider}/{self.embedder.model_name}. "
                "Regenerate index via /admin/generate-brain."
            )

        if expected_dim and int(expected_dim) != int(self.index.d):
            raise RuntimeError(
                "Embedding dimension mismatch between FAISS index and embedding_info.json. "
                "Regenerate index via /admin/generate-brain."
            )

    # ─────────────────────────────────────────
    # Vector search → relevant tables + schema
    # ─────────────────────────────────────────
    def get_context(self, user_query: str):
        query_vec = self.embedder.embed_query(user_query)
        _, indices = self.index.search(np.array(query_vec).astype("float32"), k=3)

        context_str = "STRICT SCHEMA REFERENCE (use ONLY these tables and columns):\n"
        relevant_tables = []

        for idx in indices[0]:
            if idx < 0 or idx >= len(self.id_map):
                continue
            table_name = self.id_map[int(idx)]
            relevant_tables.append(table_name)
            info = self.metadata[table_name]
            cols = [f"{c['name']} ({c['type']})" for c in info["columns"]]
            sample_vals = info.get("sample_values", {})
            context_str += f"\nTable: `{table_name}`\n  Columns: {', '.join(cols)}\n"
            if sample_vals:
                context_str += (
                    f"  Sample values: {json.dumps(sample_vals, ensure_ascii=False)}\n"
                )

        return context_str, relevant_tables

    # ─────────────────────────────────────────
    # Dynamic header from matched tables
    # ─────────────────────────────────────────
    def _resolve_header(self, tables: list) -> str:
        joined = " ".join(tables).lower()
        if any(k in joined for k in ["bjp", "sadasyata", "membership"]):
            return "BJP Membership List"
        if "mandal" in joined:
            return "Mandal Karyakarni List"
        if "ward" in joined or "panchayat" in joined:
            return "Ward/Panchayat Details"
        return "Data Report"

    # ─────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────
    def ask(self, user_query: str):
        context, tables = self.get_context(user_query)
        intent = detect_intent(user_query)
        header_title = self._resolve_header(tables)
        groq = get_groq_client()

        # ── 1. SQL GENERATION ────────────────────────────────────────────────────
        sql_prompt = f"""
You are a MySQL expert working with a database that has MIXED-LANGUAGE content.
Column names and stored values may be in English, Hindi (Devanagari), or transliterated Hindi.

{context}

User question: {user_query}

STRICT RULES — violating any rule makes the query wrong:
1. USE ONLY the tables listed in the schema above. Do NOT invent table names.
2. MINIMISE joins: if the required data exists in ONE table, use only that table.
   Only JOIN a second table when a foreign key lookup is genuinely needed for the answer.
3. For string comparisons on Hindi/mixed columns use COLLATE utf8mb4_unicode_ci.
4. For partial name matches use LIKE '%value%' with COLLATE utf8mb4_unicode_ci.
5. LIMIT results to 26 rows maximum.
6. Return ONLY the raw SQL query — no explanation, no markdown fences, no backticks.
7. If the question asks for a count/total, use SELECT COUNT(*) — do NOT return individual rows.
8. Never use DELETE, UPDATE, INSERT, DROP, or ALTER.
"""

        sql_query = None
        last_sql_error = None
        for model_name in _sql_llm_models():
            _log_model_event("sql_generation", model_name, "attempt")
            try:
                res = groq.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": sql_prompt}],
                    temperature=0,
                )
                sql_query = (
                    res.choices[0].message.content.strip()
                    .replace("```sql", "")
                    .replace("```", "")
                    .strip()
                )
                _log_model_event("sql_generation", model_name, "success")
                break
            except Exception as e:
                last_sql_error = e
                _log_model_event("sql_generation", model_name, "failed", str(e))
                print(f"[WARN] SQL generation model failed ({model_name}): {e}")

        if not sql_query:
            return {
                "answer": "### Error\nSQL generation failed. Please try again shortly.",
                "error": str(last_sql_error) if last_sql_error else "Unknown SQL generation error.",
                "sql": "N/A",
                "data": [],
            }

        # ── 2. SECURITY CHECK ────────────────────────────────────────────────────
        forbidden = [r"\bDELETE\b", r"\bUPDATE\b", r"\bINSERT\b", r"\bDROP\b", r"\bALTER\b"]
        if any(re.search(word, sql_query.upper()) for word in forbidden):
            return {"answer": "### Unauthorized\nThis action is not permitted.", "data": []}

        # ── 3. EXECUTE ───────────────────────────────────────────────────────────
        try:
            with get_db_engine().connect() as conn:
                result = conn.execute(text(sql_query))
                raw_rows = [dict(row._mapping) for row in result]

            has_more = len(raw_rows) > 25
            display_data = raw_rows[:25]
            clean_data = json.loads(
                json.dumps(display_data, default=universal_serializer)
            )

            # ── 4. NO DATA ───────────────────────────────────────────────────────
            if not clean_data:
                return {
                    "answer": (
                        f"### {header_title}\n\n"
                        f"No matching records found for: **{user_query}**"
                    ),
                    "sql": sql_query,
                    "data": [],
                    "intent": intent,
                }

            # ── 5. FORMAT RESPONSE BY INTENT ────────────────────────────────────

            # COUNT — natural conversational sentence
            if intent == "count":
                count_val = list(clean_data[0].values())[0]
                return {
                    "answer": f"{header_title} mein total **{count_val}** members hain.",
                    "sql": sql_query,
                    "data": clean_data,
                    "intent": intent,
                }

            # LOOKUP — short detail card, NOT a full table
            if intent == "lookup":
                final_answer = format_lookup_response(
                    clean_data, f"{header_title} - Lookup"
                )
                return {
                    "answer": final_answer,
                    "sql": sql_query,
                    "data": clean_data,
                    "intent": intent,
                }

            # LIST — markdown table (default behaviour)
            summary_prompt = f"""
The user asked: "{user_query}"
Format the data below as a clean Markdown table.

RULES:
- Output the markdown table immediately — no introductory text.
- Column headers should be human-readable (translate/expand abbreviations if obvious).
- Do NOT add any text after the table except the "more records" note if needed.

Data: {json.dumps(clean_data, ensure_ascii=False)}
"""
            final_answer = None
            last_fmt_error = None
            for model_name in _format_llm_models():
                _log_model_event("list_formatting", model_name, "attempt")
                try:
                    fmt_res = groq.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a markdown table formatter. Output only the header and table.",
                            },
                            {"role": "user", "content": summary_prompt},
                        ],
                    )
                    final_answer = fmt_res.choices[0].message.content.strip()
                    _log_model_event("list_formatting", model_name, "success")
                    break
                except Exception as e:
                    last_fmt_error = e
                    _log_model_event("list_formatting", model_name, "failed", str(e))

            if final_answer is None:
                return {
                    "answer": "### Error\nResult formatting failed. Please try again.",
                    "error": str(last_fmt_error) if last_fmt_error else "Unknown formatting error.",
                    "sql": sql_query,
                    "data": clean_data,
                }
            if has_more:
                final_answer += "\n\n*Showing first 25 of more records.*"

            return {
                "answer": final_answer,
                "sql": sql_query,
                "data": clean_data,
                "has_more": has_more,
                "intent": intent,
            }

        except Exception as e:
            return {
                "answer": "### Error\nThe query could not be executed.",
                "error": str(e),
                "sql": sql_query if "sql_query" in locals() else "N/A",
                "data": [],
            }
