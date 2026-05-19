import numpy as np
import logging
from typing import Tuple, Optional, Dict
import faiss

logger = logging.getLogger(__name__)


def search_face(
    query_embedding: np.ndarray,
    faiss_index,
    index_to_code: Dict[int, str],
    k: int = 1,
    threshold: float = 0.6
) -> Tuple[Optional[str], Optional[float]]:
    """
    Search for a face in FAISS index (stateless version).
    """

    if query_embedding.shape != (512,):
        logger.error(f"Invalid embedding shape: {query_embedding.shape}")
        return None, None

    try:
        query_array = np.array([query_embedding], dtype=np.float32)

        D, I = faiss_index.search(query_array, k=k)

        idx = I[0][0]
        distance = D[0][0]

        logger.info(f"FAISS result: idx={idx}, dist={distance:.4f}")

        if distance > threshold:
            return None, distance

        if idx not in index_to_code:
            return None, distance

        return index_to_code[idx], distance

    except Exception as e:
        logger.error(f"FAISS error: {str(e)}")
        return None, None