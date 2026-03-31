import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

_embedding_service = None


def _as_int(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _as_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


class EmbeddingService:
    def __init__(self):
        self.provider = os.getenv("EMBEDDING_PROVIDER", "cohere").strip().lower()
        self.batch_size = _as_int(os.getenv("EMBEDDING_BATCH_SIZE"), 64)
        self.max_retries = _as_int(os.getenv("EMBEDDING_MAX_RETRIES"), 3)
        self.retry_base_seconds = _as_float(os.getenv("EMBEDDING_RETRY_BASE_SECONDS"), 1.0)
        self._client = None
        self._model = None

        default_model = (
            "embed-v4.0"
            if self.provider == "cohere"
            else "BAAI/bge-small-en-v1.5"
        )
        self.model_name = os.getenv("EMBEDDING_MODEL", default_model).strip()
        self.cohere_output_dimension = _as_int(
            os.getenv("COHERE_EMBED_OUTPUT_DIMENSION"), 0
        )

        if self.provider not in {"sentence_transformers", "cohere"}:
            raise ValueError(
                "Unsupported EMBEDDING_PROVIDER. Use 'sentence_transformers' or 'cohere'."
            )
        print(f"[MODEL][embeddings][configured] {self.provider}/{self.model_name}")

    def get_info(self, vector_dim: Optional[int] = None) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model_name,
            "vector_dim": vector_dim,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("embed_documents called with empty text list.")
        if self.provider == "cohere":
            input_type = os.getenv("COHERE_EMBED_INPUT_TYPE_DOCUMENT", "search_document")
            return self._embed_with_cohere(texts, input_type=input_type)
        return self._embed_with_sentence_transformers(texts)

    def embed_query(self, text: str) -> np.ndarray:
        if self.provider == "cohere":
            input_type = os.getenv("COHERE_EMBED_INPUT_TYPE_QUERY", "search_query")
            return self._embed_with_cohere([text], input_type=input_type)
        return self._embed_with_sentence_transformers([text])

    def _embed_with_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"[INFO] Loading SentenceTransformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print("[INFO] SentenceTransformer model loaded.")

        vectors = self._model.encode(texts)
        return np.array(vectors, dtype="float32")

    def _get_cohere_client(self):
        if self._client is not None:
            return self._client

        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY is required when EMBEDDING_PROVIDER=cohere.")

        import cohere

        if hasattr(cohere, "ClientV2"):
            self._client = cohere.ClientV2(api_key=api_key)
        else:
            self._client = cohere.Client(api_key=api_key)

        return self._client

    def _embed_with_cohere(self, texts: List[str], input_type: str) -> np.ndarray:
        client = self._get_cohere_client()
        all_vectors: List[List[float]] = []

        for batch in self._chunk(texts, self.batch_size):
            vectors = self._cohere_embed_with_retry(client, batch, input_type=input_type)
            all_vectors.extend(vectors)

        return np.array(all_vectors, dtype="float32")

    def _cohere_embed_with_retry(self, client, texts: List[str], input_type: str) -> List[List[float]]:
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                embed_kwargs = {
                    "model": self.model_name,
                    "texts": texts,
                    "input_type": input_type,
                }
                if self.cohere_output_dimension > 0:
                    embed_kwargs["output_dimension"] = self.cohere_output_dimension

                # Support both v2 and v1 styles.
                response = client.embed(**embed_kwargs)

                if hasattr(response, "embeddings"):
                    emb = response.embeddings
                    if hasattr(emb, "float_") and emb.float_ is not None:
                        return emb.float_
                    if hasattr(emb, "float") and emb.float is not None:
                        return emb.float
                    if isinstance(emb, list):
                        return emb

                if isinstance(response, dict):
                    dict_emb = response.get("embeddings")
                    if isinstance(dict_emb, list):
                        return dict_emb

                raise RuntimeError("Cohere embed response format not recognized.")
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_seconds = self.retry_base_seconds * (2 ** attempt)
                time.sleep(sleep_seconds)

        raise RuntimeError(f"Cohere embedding failed after retries: {last_error}") from last_error

    @staticmethod
    def _chunk(items: List[str], size: int) -> List[List[str]]:
        if size <= 0:
            size = 64
        return [items[i:i + size] for i in range(0, len(items), size)]


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
