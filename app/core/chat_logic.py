import os
import json
import faiss
import re
import numpy as np
from datetime import datetime, date
from decimal import Decimal
from uuid import UUID

from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
db_engine = create_engine(os.getenv("DATABASE_URL"))


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
    lookup_keywords = ["who is", "detail", "info", "pata", "address", "contact",
                       "phone", "mobile", "naam", "name of", "kiska", "kaun"]
    list_keywords   = ["list", "show", "all", "sabhi", "dikhao", "suchi", "record",
                       "members", "sadasya", "give me", "fetch", "report"]

    if any(k in q for k in count_keywords):
        return "count"
    if any(k in q for k in lookup_keywords):
        return "lookup"
    if any(k in q for k in list_keywords):
        return "list"
    # default — treat as list
    return "list"


class DatabaseChatbot:
    def __init__(self, project_path=None):
        if project_path is None:
            project_path = os.getenv("STORAGE_PATH", "./store/current_db")
        self.index = faiss.read_index(os.path.join(project_path, "vector.faiss"))
        with open(os.path.join(project_path, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        with open(os.path.join(project_path, "id_map.json"), "r") as f:
            self.id_map = json.load(f)

    # ─────────────────────────────────────────
    # Vector search → relevant tables + schema
    # ─────────────────────────────────────────
    def get_context(self, user_query):
        query_vec = embed_model.encode([user_query])
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
            sample_vals = info.get("sample_values", {})   # optional: store sample cell values during indexing
            context_str += f"\nTable: `{table_name}`\n  Columns: {', '.join(cols)}\n"
            if sample_vals:
                context_str += f"  Sample values: {json.dumps(sample_vals, ensure_ascii=False)}\n"

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

        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": sql_prompt}],
            temperature=0,          # deterministic SQL
        )
        sql_query = (
            res.choices[0].message.content.strip()
            .replace("```sql", "")
            .replace("```", "")
            .strip()
        )

        # ── 2. SECURITY CHECK ────────────────────────────────────────────────────
        forbidden = [r"\bDELETE\b", r"\bUPDATE\b", r"\bINSERT\b", r"\bDROP\b", r"\bALTER\b"]
        if any(re.search(word, sql_query.upper()) for word in forbidden):
            return {"answer": "### Unauthorized\nThis action is not permitted.", "data": []}

        # ── 3. EXECUTE ───────────────────────────────────────────────────────────
        try:
            with db_engine.connect() as conn:
                result = conn.execute(text(sql_query))
                raw_rows = [dict(row._mapping) for row in result]

            has_more = len(raw_rows) > 25
            display_data = raw_rows[:25]
            clean_data = json.loads(json.dumps(display_data, default=universal_serializer))

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
                final_answer = format_lookup_response(clean_data, f"{header_title} - Lookup")
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
- Then the markdown table immediately — no introductory text.
- Column headers should be human-readable (translate/expand abbreviations if obvious).
- Do NOT add any text after the table except the "more records" note if needed.

Data: {json.dumps(clean_data, ensure_ascii=False)}
"""
            fmt_res = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a markdown table formatter. Output only the header and table."},
                    {"role": "user",   "content": summary_prompt},
                ],
            )
            final_answer = fmt_res.choices[0].message.content.strip()
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
