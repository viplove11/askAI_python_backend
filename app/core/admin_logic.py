import os
import json
import faiss
import numpy as np
from sqlalchemy import create_engine, inspect, text
from groq import Groq
from dotenv import load_dotenv
from app.core.embeddings import get_embedding_service

load_dotenv()

# ─────────────────────────────────────────────
# Lazy singletons — NOT loaded at import time
# ─────────────────────────────────────────────
_groq_client = None


def _log_model_event(purpose: str, model: str, status: str, detail: str = ""):
    suffix = f" | {detail}" if detail else ""
    print(f"[MODEL][{purpose}][{status}] {model}{suffix}")


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def _admin_llm_model() -> str:
    """Primary model used for admin-side description generation."""
    return os.getenv("ADMIN_LLM_MODEL", "qwen/qwen3-32b").strip()


def _admin_llm_fallback_model() -> str:
    """First fallback for admin-side description generation."""
    return os.getenv("ADMIN_LLM_FALLBACK_MODEL", "llama-3.3-70b-versatile").strip()


def _admin_llm_second_fallback_model() -> str:
    """Second fallback for admin-side description generation."""
    return os.getenv("ADMIN_LLM_SECOND_FALLBACK_MODEL", "llama-3.1-8b-instant").strip()


def _fallback_table_description(table_info: dict) -> str:
    """Deterministic fallback if LLM call fails/rate-limits."""
    table_name = table_info.get("table_name", "this table")
    cols = [c.get("name", "") for c in table_info.get("columns", []) if c.get("name")]
    if not cols:
        return f"Stores records in `{table_name}`."
    col_preview = ", ".join(cols[:8])
    return f"Stores records in `{table_name}` with fields like {col_preview}."


# ─────────────────────────────────────────────────────────────────
# Step 1 — Inspect schema + collect sample values from each column
# ─────────────────────────────────────────────────────────────────
def generate_db_inventory(db_url: str, sample_limit: int = 5):
    """
    Inspects the DB and returns table schemas with real sample values.

    sample_limit: how many distinct non-null values to fetch per column.
                  Keep it small (3–5) — it's only used as LLM context.
    """
    engine = create_engine(db_url)
    inspector = inspect(engine)
    inventory = []

    with engine.connect() as conn:
        for table_name in inspector.get_table_names():
            columns = []
            sample_values = {}  # { col_name: [val1, val2, ...] }

            for col in inspector.get_columns(table_name):
                col_name = col["name"]
                col_type = str(col["type"])
                columns.append({"name": col_name, "type": col_type})

                # Fetch a few real values so the LLM knows what the data looks like
                try:
                    query = text(
                        f"SELECT DISTINCT `{col_name}` "
                        f"FROM `{table_name}` "
                        f"WHERE `{col_name}` IS NOT NULL "
                        f"LIMIT {sample_limit}"
                    )
                    rows = conn.execute(query).fetchall()
                    values = []
                    for row in rows:
                        v = row[0]
                        if hasattr(v, "isoformat"):   # date / datetime
                            v = v.isoformat()
                        elif hasattr(v, "__float__"):  # Decimal
                            v = float(v)
                        values.append(str(v))
                    sample_values[col_name] = values
                except Exception:
                    # If a column can't be sampled (e.g. BLOB), skip silently
                    sample_values[col_name] = []

            inventory.append({
                "table_name": table_name,
                "columns": columns,
                "sample_values": sample_values,
            })

    return inventory


# ─────────────────────────────────────────────────────────────────
# Step 2 — Ask LLM for a 1-sentence table description
# ─────────────────────────────────────────────────────────────────
def get_ai_description(table_info: dict) -> str:
    """
    Uses sample values alongside column names so the LLM can understand
    tables whose columns are in Hindi / transliterated Hindi.
    """
    col_lines = []
    for col in table_info["columns"]:
        samples = table_info["sample_values"].get(col["name"], [])[:3]
        trimmed_samples = [str(v)[:40] for v in samples]
        sample_str = f"  (e.g. {', '.join(trimmed_samples)})" if trimmed_samples else ""
        col_lines.append(f"  - {col['name']} ({col['type']}){sample_str}")

    col_block = "\n".join(col_lines)

    prompt = f"""Analyze this database table and its real sample data:

Table Name : {table_info['table_name']}
Columns    :
{col_block}

Note: Column names and values may be in English, Hindi (Devanagari), or
transliterated Hindi (e.g. "naam" = name, "pata" = address, "ward_no" = ward number).

Task: Write ONE sentence describing what kind of data this table stores.
Focus on business meaning (e.g. "Stores BJP party member details including
name, address, ward number and contact information in Hindi.").
Return ONLY that sentence — no extra text."""

    candidate_models = [
        _admin_llm_model(),
        _admin_llm_fallback_model(),
        _admin_llm_second_fallback_model(),
    ]

    # Preserve order but avoid duplicate attempts if env vars repeat values.
    seen = set()
    ordered_models = []
    for model_name in candidate_models:
        if model_name and model_name not in seen:
            seen.add(model_name)
            ordered_models.append(model_name)

    for model_name in ordered_models:
        _log_model_event("admin_description", model_name, "attempt")
        try:
            response = get_groq_client().chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            _log_model_event("admin_description", model_name, "success")
            return response.choices[0].message.content.strip()
        except Exception as e:
            _log_model_event("admin_description", model_name, "failed", str(e))
            print(
                f"[WARN] Description model failed for {table_info.get('table_name')} "
                f"({model_name}): {e}"
            )

    # Do not fail full brain generation if all model calls fail.
    return _fallback_table_description(table_info)


# ─────────────────────────────────────────────────────────────────
# Step 3 — Build FAISS index and save all artefacts
# ─────────────────────────────────────────────────────────────────
def build_and_save_index(db_url: str, out_dir: str):
    """Inspect → Sample → Describe → Vectorize → Save."""

    print("🔍 Inspecting database and collecting sample values...")
    inventory = generate_db_inventory(db_url)

    metadata = {}
    descriptions = []
    table_names = []

    print("🧠 Generating AI descriptions for tables...")
    for table in inventory:
        name = table["table_name"]
        desc = get_ai_description(table)

        metadata[name] = {
            "table_name": name,
            "columns": table["columns"],
            "sample_values": table["sample_values"],
            "description": desc,
        }

        sample_snippets = []
        for col_name, vals in table["sample_values"].items():
            if vals:
                sample_snippets.append(f"{col_name}: {', '.join(vals[:3])}")

        embed_text = (
            f"Table {name}: {desc} "
            f"Columns: {', '.join(c['name'] for c in table['columns'])}. "
            f"Sample data — {'; '.join(sample_snippets[:8])}"
        )
        descriptions.append(embed_text)
        table_names.append(name)

    print("🔢 Vectorizing and building FAISS index...")
    embedder = get_embedding_service()
    vectors = embedder.embed_documents(descriptions)

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    os.makedirs(out_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(out_dir, "vector.faiss"))

    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    with open(os.path.join(out_dir, "id_map.json"), "w", encoding="utf-8") as f:
        json.dump(table_names, f, ensure_ascii=False)

    embedding_info = embedder.get_info(vector_dim=int(dim))
    with open(os.path.join(out_dir, "embedding_info.json"), "w", encoding="utf-8") as f:
        json.dump(embedding_info, f, indent=4, ensure_ascii=False)

    print(f"✅ Brain saved to '{out_dir}' — {len(table_names)} tables indexed.")
    for name in table_names:
        print(f"   • {name}: {metadata[name]['description']}")
