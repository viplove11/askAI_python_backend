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

class DatabaseChatbot:
    def __init__(self, project_path=None):
        if project_path is None:
            project_path = os.getenv("STORAGE_PATH", "./store/current_db")
        self.index = faiss.read_index(os.path.join(project_path, "vector.faiss"))
        with open(os.path.join(project_path, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        with open(os.path.join(project_path, "id_map.json"), "r") as f:
            self.id_map = json.load(f)

    def get_context(self, user_query):
        query_vec = embed_model.encode([user_query])
        _, indices = self.index.search(np.array(query_vec).astype('float32'), k=3)
        context_str = "STRICT SCHEMA REFERENCE:\n"
        relevant_tables = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.id_map): continue
            table_name = self.id_map[int(idx)] 
            relevant_tables.append(table_name)
            info = self.metadata[table_name]
            cols = [f"{c['name']} ({c['type']})" for c in info['columns']]
            context_str += f"- Table: {table_name} | Columns: {', '.join(cols)}\n"
        return context_str, relevant_tables

    def ask(self, user_query):
        context, tables = self.get_context(user_query)

        # 1. SQL GENERATION
        sql_prompt = f"""
        You are a MySQL expert. {context}
        Question: {user_query}
        RULES:
        1. ID Logic: JOIN tables for names; never use 'id_column = Name'.
        2. Collation: Use 'COLLATE utf8mb4_unicode_ci' for joins on string columns.
        3. Constraints: LIMIT 26. Return ONLY raw SQL. No backticks.
        """
        
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": sql_prompt}]
        )
        sql_query = res.choices[0].message.content.strip().replace('```sql', '').replace('```', '')

        # Security Check
        forbidden = [r"\bDELETE\b", r"\bUPDATE\b", r"\bINSERT\b", r"\bDROP\b", r"\bALTER\b"]
        if any(re.search(word, sql_query.upper()) for word in forbidden):
            return {"answer": "### 🚫 Unauthorized\nAction not permitted.", "data": []}

        try:
            with db_engine.connect() as conn:
                result = conn.execute(text(sql_query))
                raw_rows = [dict(row._mapping) for row in result]
                has_more = len(raw_rows) > 25
                display_data = raw_rows[:25]
                clean_data = json.loads(json.dumps(display_data, default=universal_serializer))

            # Table Header Logic based on keywords
            header_title = "Data Report"
            if any("bjp" in t.lower() or "sadasyata" in t.lower() for t in tables):
                header_title = "BJP Membership List"
            elif any("mandal" in t.lower() for t in tables):
                header_title = "Mandal Karyakarni List"
            elif any("ward" in t.lower() for t in tables):
                header_title = "Ward/Panchayat Details"

            # 2. NO DATA FALLBACK
            if not clean_data:
                return {
                    "answer": f"### {header_title}\n\n| Category | Information |\n| :--- | :--- |\n| Requested Data | {user_query} |\n| Status | No matching records found. |",
                    "sql": sql_query,
                    "data": []
                }

            # 3. TO-THE-POINT TABLE SUMMARY
            summary_prompt = f"""
            Task: Format this database data into a Markdown table for the user query: {user_query}
            Header to use: ### {header_title}
            
            RULES:
            - Start directly with the header and table.
            - NO introductory text.
            - NO conversational filler.
            - Provide ONLY the table.
            """
            
            summary_res = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a specialized markdown table formatter. You do not talk, you only build tables."},
                    {"role": "user", "content": f"{summary_prompt}\nData: {json.dumps(clean_data)}"}
                ]
            )
            
            final_answer = summary_res.choices[0].message.content
            if has_more:
                final_answer += "\n\n*Note: More than 25 records found. Showing the first 25.*"

            return {
                "answer": final_answer,
                "sql": sql_query,
                "data": clean_data,
                "has_more": has_more
            }
        except Exception as e:
            return {"answer": "### ⚠️ Error\nQuery failed.", "error": str(e), "data": []}