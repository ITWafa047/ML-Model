import base64
import cv2
import numpy as np
import logging
from typing import Optional
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


def decode_frame(frame_data: str) -> Optional[np.ndarray]:
    """
    Decode a base64-encoded frame to RGB numpy array.
    
    Args:
        frame_data: Base64-encoded image data (can include data URI prefix)
        
    Returns:
        RGB numpy array or None if decoding fails
        
    Expected input format:
        - Raw base64: "iVBORw0KGgoAAAANSUhEUgAAAAUA..."
        - Data URI: "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA..."
    """
    try:
        # Remove data URI prefix if present
        if isinstance(frame_data, str) and frame_data.startswith("data:"):
            # Format: "data:image/jpeg;base64,<base64_data>"
            frame_data = frame_data.split(",", 1)[1]
        
        # Decode base64
        image_bytes = base64.b64decode(frame_data)
        
        # Convert bytes to numpy array
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        
        # Decode image with OpenCV (reads as BGR)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            logger.error("Failed to decode image with cv2.imdecode")
            # Try PIL as fallback
            return decode_frame_pil(image_bytes)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        logger.info(f"Frame decoded successfully: shape={image_rgb.shape}")
        return image_rgb
        
    except Exception as e:
        logger.error(f"Error decoding frame: {str(e)}")
        return None


def decode_frame_pil(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Fallback method to decode image using PIL.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        RGB numpy array or None
    """
    try:
        image_pil = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        
        image_rgb = np.array(image_pil)
        logger.info(f"Frame decoded with PIL: shape={image_rgb.shape}")
        return image_rgb
        
    except Exception as e:
        logger.error(f"Error decoding frame with PIL: {str(e)}")
        return None


def validate_frame(frame: Optional[np.ndarray]) -> bool:
    """
    Validate that frame is a valid RGB array.
    
    Args:
        frame: Numpy array to validate
        
    Returns:
        True if valid, False otherwise
    """
    if frame is None:
        logger.error("Frame is None")
        return False
    
    if not isinstance(frame, np.ndarray):
        logger.error(f"Frame is not numpy array: {type(frame)}")
        return False
    
    if len(frame.shape) != 3:
        logger.error(f"Invalid frame shape: {frame.shape}")
        return False
    
    if frame.shape[2] != 3:
        logger.error(f"Frame must have 3 channels, got {frame.shape[2]}")
        return False
    
    if frame.dtype != np.uint8:
        logger.error(f"Frame must be uint8, got {frame.dtype}")
        return False
    
    return True


def frame_to_bytes(frame: np.ndarray, format: str = "JPEG") -> bytes:
    """
    Convert RGB frame to bytes (for storage or transmission).
    
    Args:
        frame: RGB numpy array
        format: Image format ("JPEG", "PNG")
        
    Returns:
        Image bytes
    """
    try:
        # Convert RGB to BGR for cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Encode to bytes
        _, buffer = cv2.imencode(f".{format.lower()}", frame_bgr)
        return buffer.tobytes()
        
    except Exception as e:
        logger.error(f"Error converting frame to bytes: {str(e)}")
        return b""


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize frame to specified dimensions.
    
    Args:
        frame: RGB numpy array
        width: Target width
        height: Target height
        
    Returns:
        Resized RGB numpy array
    """
    try:
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized
    except Exception as e:
        logger.error(f"Error resizing frame: {str(e)}")
        return frame


def extract_face_region(
    frame: np.ndarray,
    bbox: tuple
) -> Optional[np.ndarray]:
    """
    Extract face region from frame using bounding box.
    
    Args:
        frame: RGB frame
        bbox: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Cropped RGB frame or None
    """
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Ensure bounds are within frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        face_region = frame[y1:y2, x1:x2]
        return face_region
        
    except Exception as e:
        logger.error(f"Error extracting face region: {str(e)}")
        return None
