import os
import json
import faiss
import numpy as np
from sqlalchemy import create_engine, inspect
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# 1. Setup Clients
# BGE-M3 is the best small open-source embedding model in 2026
embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_db_inventory(db_url):
    """Inspects the DB and returns a list of table schemas."""
    engine = create_engine(db_url)
    inspector = inspect(engine)
    inventory = []

    for table_name in inspector.get_table_names():
        columns = [
            {"name": col['name'], "type": str(col['type'])} 
            for col in inspector.get_columns(table_name)
        ]
        inventory.append({"table_name": table_name, "columns": columns})
    
    return inventory

def get_ai_description(table_info):
    """Ask Qwen-3 to summarize what a table does based on its columns."""
    col_names = [c['name'] for c in table_info['columns']]
    prompt = f"""
    Analyze this database table:
    Table Name: {table_info['table_name']}
    Columns: {', '.join(col_names)}
    
    Task: Write a 1-sentence description explaining what kind of data this table holds. 
    Focus on business logic (e.g., 'Stores patient medical history and prescriptions').
    Return ONLY the sentence.
    """
    
    response = groq_client.chat.completions.create(
        # CHANGE THIS LINE BELOW:
        model="llama-3.3-70b-versatile", 
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def build_and_save_index(db_url, out_dir):
    """The main process: Inspect -> Describe -> Vectorize -> Save."""
    print("🔍 Inspecting database...")
    inventory = generate_db_inventory(db_url)
    
    metadata = {}
    descriptions = []
    table_names = []

    print("🧠 Generating AI descriptions for tables...")
    for table in inventory:
        desc = get_ai_description(table)
        name = table['table_name']
        
        metadata[name] = {
            "table_name": name,
            "columns": table['columns'],
            "description": desc
        }
        descriptions.append(f"Table {name}: {desc}")
        table_names.append(name)

    print("🔢 Vectorizing and saving FAISS index...")
    # Convert descriptions to math (vectors)
    vectors = embed_model.encode(descriptions)
    
    # Create FAISS Index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype('float32'))

    # Ensure directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save the 3 required files
    faiss.write_index(index, os.path.join(out_dir, "vector.faiss"))
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    with open(os.path.join(out_dir, "id_map.json"), "w") as f:
        json.dump(table_names, f)

    print(f"✅ Success! Brain saved to {out_dir}")