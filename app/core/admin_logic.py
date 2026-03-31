import os
import json
import faiss
import numpy as np
from sqlalchemy import create_engine, inspect, text
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Lazy singletons — NOT loaded at import time
# ─────────────────────────────────────────────
_embed_model = None
_groq_client = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print("[INFO] Loading SentenceTransformer model...")
        _embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("[INFO] SentenceTransformer model loaded.")
    return _embed_model


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


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
        samples = table_info["sample_values"].get(col["name"], [])
        sample_str = f"  (e.g. {', '.join(samples)})" if samples else ""
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

    response = get_groq_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


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
    vectors = get_embed_model().encode(descriptions)   # ← lazy load here

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    os.makedirs(out_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(out_dir, "vector.faiss"))

    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    with open(os.path.join(out_dir, "id_map.json"), "w", encoding="utf-8") as f:
        json.dump(table_names, f, ensure_ascii=False)

    print(f"✅ Brain saved to '{out_dir}' — {len(table_names)} tables indexed.")
    for name in table_names:
        print(f"   • {name}: {metadata[name]['description']}")