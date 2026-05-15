import numpy as np
import threading
import cv2
import os
import logging
import onnxruntime as ort
from typing import Optional, Union, Tuple, List, Dict


class FaceProcessor:
    """
    Process faces and extract embeddings using ArcFace model.

    Features:
        - Robust embedding extraction with augmentations
        - ONNX Runtime with GPU acceleration
        - Thread-safe model caching
        - Numerical stability
    """

    def __init__(self):
        self._session = None
        self._current_model_path = None
        self._session_lock = threading.Lock()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        available_providers = ort.get_available_providers()
        preferred_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.providers = [p for p in preferred_providers if p in available_providers]
        if not self.providers:
            self.providers = ["CPUExecutionProvider"]

        self.model_path = os.path.join(self.BASE_DIR, "buffalo_l", "w600k_r50.onnx")
        self.logger.info("FaceProcessor initialized with providers: %s", self.providers)

    def preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """Preprocess face for ArcFace model."""

        # Validate input
        if aligned_face.shape != (112, 112, 3):
            raise ValueError(f"Expected (112, 112, 3), got {aligned_face.shape}")

        # Validate dtype
        if aligned_face.dtype != np.uint8:
            raise ValueError(f"Expected uint8, got {aligned_face.dtype}")


        image = aligned_face.astype(np.float32) # Convert to float32 for processing
        image = image / 127.5 - 1.0 # Normalize to [-1, 1]
        image = np.transpose(image, (2, 0, 1)) # Change to (C, H, W)
        image = np.expand_dims(image, axis=0) # Add batch dimension

        return image # (1, 3, 112, 112) float32

    def extract_embedding(
        self, preprocessed_image: np.ndarray, model_name: str = None
    ) -> np.ndarray:
        """
        Extract embedding using w600k_r50.onnx model.

        FIXED: Validates model path exists and uses numerical stability.
        """

        # FIXED: Validate model path
        model_name = model_name or self.model_path

        if not os.path.exists(model_name):
            raise FileNotFoundError(
                f"ArcFace model not found: {model_name}\n"
                f"Download from: https://github.com/deepinsight/insightface/releases"
                f"Place in project root or specify correct path."
            )

        # Thread-safe model loading and inference
        with self._session_lock:
            if (
                self._session is None
                or self._current_model_path != model_name
            ):
                self.logger.info(f"Loading ArcFace model: {model_name}")
                self._session = ort.InferenceSession(
                    model_name, providers=self.providers
                )
                self._current_model_path = model_name
                self.logger.info(
                    f"Model loaded with providers: {self._session.get_providers()}"
                )

            input_name = self._session.get_inputs()[0].name
            output_name = self._session.get_outputs()[0].name

            output = self._session.run(
                [output_name], {input_name: preprocessed_image}
            )

        embedding = output[0][0]

        # FIXED: Numerical stability in normalization
        norm = np.linalg.norm(embedding)
        embedding = embedding / (norm + 1e-8)

        return embedding

    
    def generate_embedding(
        self, face: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate robust embedding with augmentations.

        Augmentations:
            1. Original
            2. Horizontally flipped
            3. Brightness +40
            4. Brightness -40
            5. Rotated +10°
            6. Rotated -10°

        Args:
            face: Aligned face (112, 112, 3) uint8 RGB

        Returns:
            - mean_embedding: (512,) normalized average
            - embeddings_stack: (6, 512) all embeddings
        """

        # Validate input
        if face.shape != (112, 112, 3):
            raise ValueError(f"Expected shape (112, 112, 3), got {face.shape}")

        if face.dtype != np.uint8:
            raise ValueError(f"Expected uint8, got {face.dtype}")

        self.logger.info("Generating embeddings with augmentations...")

        # Create augmentations
        augmentations = []

        # 1. Original
        augmentations.append(face)

        # 2. Flip
        augmentations.append(cv2.flip(face, 1))

        # 3. Brightness +40
        bright_plus = np.clip(face.astype(np.int16) + 40, 0, 255).astype(np.uint8)
        augmentations.append(bright_plus)

        # 4. Brightness -40
        bright_minus = np.clip(face.astype(np.int16) - 40, 0, 255).astype(np.uint8)
        augmentations.append(bright_minus)

        # 5. Rotate +10°
        center = (56, 56)
        rot_matrix_pos = cv2.getRotationMatrix2D(center, 10, 1.0)
        rotated_pos = cv2.warpAffine(
            face,
            rot_matrix_pos,
            (112, 112),
            borderMode=cv2.BORDER_REFLECT_101,
            flags=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        augmentations.append(rotated_pos)

        # 6. Rotate -10°
        rot_matrix_neg = cv2.getRotationMatrix2D(center, -10, 1.0)
        rotated_neg = cv2.warpAffine(
            face,
            rot_matrix_neg,
            (112, 112),
            borderMode=cv2.BORDER_REFLECT_101,
            flags=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        augmentations.append(rotated_neg)

        # Extract embeddings
        embeddings = []
        for i, aug_image in enumerate(augmentations):
            preprocessed = self.preprocess(aug_image)
            embedding = self.extract_embedding(preprocessed, self.model_path)
            embeddings.append(embedding)
            self.logger.info(f"Extracted embedding {i+1}/6")

        # Stack and aggregate
        embeddings_stack = np.stack(embeddings, axis=0)  # (6, 512)
        mean_embedding = np.mean(embeddings_stack, axis=0)  # (512,)


        # Normalize
        norm = np.linalg.norm(mean_embedding)
        normalized_embedding = mean_embedding / (norm + 1e-8)

        self.logger.info("Embedding generation completed")
        return normalized_embedding,  embeddings_stack
