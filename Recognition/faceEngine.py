import numpy as np
import logging
from typing import Tuple, Optional, Dict
import faiss

logger = logging.getLogger(__name__)

# Global session context
current_session = None


def set_current_session(session_data: Dict) -> None:
    """
    Set the active session with FAISS index and student mapping.
    
    Args:
        session_data: Dictionary containing:
            - faiss_index: FAISS index object
            - index_to_code: Mapping from FAISS index to student_code
            - student_map: Mapping from student_code to student_name
            - student_codes: List of student codes
    """
    global current_session
    current_session = session_data
    logger.info(f"Session set with {len(session_data['student_codes'])} students")


def get_current_session() -> Optional[Dict]:
    """Get the active session."""
    return current_session


def clear_session() -> None:
    """Clear the active session."""
    global current_session
    current_session = None
    logger.info("Session cleared")


def search_face(
    query_embedding: np.ndarray,
    k: int = 1,
    threshold: float = 0.6
) -> Tuple[Optional[str], Optional[float]]:
    """
    Search for a face in the FAISS index.
    
    Args:
        query_embedding: Query embedding vector (512-dimensional)
        k: Number of top results to return (default: 1)
        threshold: Distance threshold for acceptance (default: 0.6)
        
    Returns:
        Tuple of (student_code, distance) or (None, None) if distance exceeds threshold
        
    Raises:
        RuntimeError: If no session is active
    """
    if current_session is None:
        raise RuntimeError("No active session. Please start a session first.")
    
    # Validate embedding shape
    if query_embedding.shape != (512,):
        logger.error(f"Invalid embedding shape: {query_embedding.shape}")
        return None, None
    
    try:
        # Get FAISS index and mappings
        faiss_index = current_session["faiss_index"]
        index_to_code = current_session["index_to_code"]
        
        # Prepare query for FAISS (must be float32 and 2D)
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search in FAISS index
        # D: distances, I: indices
        D, I = faiss_index.search(query_array, k=k)
        
        # Extract results
        idx = I[0][0]  # Top 1 index
        distance = D[0][0]  # Top 1 distance
        
        logger.info(f"FAISS search result: index={idx}, distance={distance:.4f}")
        
        # Check distance threshold
        if distance > threshold:
            logger.warning(
                f"Distance {distance:.4f} exceeds threshold {threshold}. "
                f"Not recognized."
            )
            return None, distance
        
        # Get student code from mapping
        if idx not in index_to_code:
            logger.error(f"Index {idx} not found in mappings")
            return None, distance
        
        student_code = index_to_code[idx]
        logger.info(f"Recognized student: {student_code} (distance: {distance:.4f})")
        
        return student_code, distance
        
    except Exception as e:
        logger.error(f"Error during FAISS search: {str(e)}")
        return None, None


def get_top_candidates(
    query_embedding: np.ndarray,
    k: int = 5,
    threshold: float = 1.0
) -> list:
    """
    Get top-k candidates from FAISS search.
    
    Args:
        query_embedding: Query embedding vector (512-dimensional)
        k: Number of top results to return
        threshold: Distance threshold for inclusion
        
    Returns:
        List of tuples (student_code, distance) sorted by distance
    """
    if current_session is None:
        raise RuntimeError("No active session. Please start a session first.")
    
    try:
        faiss_index = current_session["faiss_index"]
        index_to_code = current_session["index_to_code"]
        
        # Prepare query
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search for top-k
        D, I = faiss_index.search(query_array, k=min(k, faiss_index.ntotal))
        
        results = []
        for i in range(len(I[0])):
            idx = I[0][i]
            distance = D[0][i]
            
            if distance <= threshold and idx in index_to_code:
                student_code = index_to_code[idx]
                results.append((student_code, distance))
        
        logger.info(f"Found {len(results)} candidates within threshold")
        return results
        
    except Exception as e:
        logger.error(f"Error getting candidates: {str(e)}")
        return []


def get_session_stats() -> Dict:
    """Get statistics about the active session."""
    if current_session is None:
        return {"status": "No active session"}
    
    return {
        "session_id": current_session.get("session_id", "N/A"),
        "student_count": len(current_session["student_codes"]),
        "index_size": current_session["faiss_index"].ntotal,
        "embedding_dimension": 512,
        "index_type": "IndexFlatL2"
    }
