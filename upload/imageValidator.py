import numpy as np
import cv2
import logging
import onnxruntime as ort
from fastapi import UploadFile, HTTPException
from typing import Optional, Union, Tuple, List, Dict
from insightface.app import FaceAnalysis


class ImageValidator:
    """
    ImageValidator is a class that provides functionality to validate images based on specific criteria such as size, format, and content. It uses OpenCV for image processing and can be integrated into applications that require image validation before further processing or storage.

    ImageValidator Pipelines:
        1. Format validation (MIME type, extension)
        2. Load image (bytes → RGB array)
        3. Size validation (minimum resolution)
        4. Face detection
        5. Single face validation
        6. Face quality checks (size, ratio)
        7. Background validation (white, uniform)
        8. Face alignment (eye-based rotation)
        9. Blur validation (sharpness)
        10. Brightness validation
    """

    def __init__(self):
        """
        Initializes the ImageValidator class. This constructor can be expanded to include any necessary configuration or parameters for validation.
        """
        # File format validation
        self.allow_formats = {"jpg", "jpeg", "png"}
        self.allow_mime_types = {"image/jpeg", "image/png", "image/jpg"}
        self.max_file_size = 10 * 1024 * 1024  # 10 MB

        # Image dimension validation
        self.min_width = 500
        self.min_height = 650

        # Face size validation
        self.min_face_width = 80
        self.min_face_height = 80
        self.min_face_ratio = 0.02
        self.min_confidence = 0.65  # or 0.75

        # lighting condition validation
        self.min_brightness = 60  # 🔥 FIX: reduced from 80 for low-light conditions
        self.max_brightness = 220
        self.blur_threshold = 100

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        available_providers = ort.get_available_providers()
        preferred_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.providers = [p for p in preferred_providers if p in available_providers]
        if not self.providers:
            self.providers = ["CPUExecutionProvider"]

        # initialize the face analysis model
        self.face_detector = FaceAnalysis(name="buffalo_l", providers=self.providers)

        # buffalo_l: a large model with high accuracy, suitable for face detection and recognition tasks. It is based on the RetinaFace architecture and provides robust performance in various conditions.

        # CUDAExecutionProvider: This provider allows the model to utilize NVIDIA GPUs for computation, significantly improving the speed of face detection and embedding extraction compared to CPU execution. It is essential for handling larger images or processing multiple images efficiently.

        # prepare the model with GPU support
        self.face_detector.prepare(
            ctx_id=0, det_size=(1024, 1024)
        )  # set det_size to (1024, 1024) for better accuracy on high-resolution images, but it can be adjusted based on performance needs

        self.logger.info(
            "ImageValidator initialized with providers: %s", self.providers
        )

    async def validate_format(self, file: UploadFile) -> bool:
        """
        Validates the format of the uploaded file based on its MIME type and extension.
        """
        # Check if the file has a valid filename
        if not file.filename or file.filename.strip() == "":
            raise HTTPException(
                status_code=400, detail="Invalid file: No filename provided."
            )

        # check if the file has an extension
        if "." not in file.filename:
            raise HTTPException(
                status_code=400, detail="Invalid file: No file extension found."
            )

        # extract the file extension and validate it
        extension = file.filename.rsplit(".", 1)[-1].lower()
        if extension not in self.allow_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format: .{extension} is not allowed. Allowed formats: {', '.join(self.allow_formats)}.",
            )

        # validate the MIME type of the file
        if file.content_type not in self.allow_mime_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid MIME type: {file.content_type} is not allowed.Allowed MIME types: {', '.join(self.allow_mime_types)}.",
            )

        # Log the successful validation of the file format
        self.logger.info(
            f"File format validation passed. Extension: {extension}, MIME type: {file.content_type}"
        )

        # If all checks pass, return True
        return True

    async def load_image(self, file: Optional[UploadFile] = None) -> np.ndarray:
        """
        Loads an image from an UploadFile object and converts it to a NumPy array in RGB format.
        """

        # check if the file is none
        if file is None or file.filename == "":
            raise HTTPException(status_code=400, detail="error: No file provided.")

        # validate the file size
        if file.size and file.size > self.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds the maximum limit of {self.max_file_size / (1024 * 1024):.2f} MB.",
            )

        # read the file content as bytes
        try:
            file_content = await file.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

        # convert the bytes to a numpy array
        try:
            byte_array = np.frombuffer(file_content, dtype=np.uint8)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error converting file to byte array: {str(e)}"
            )

        # decode the bytes into an image using OpenCV
        image_bgr = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(
                status_code=400, detail="Invalid image format or corrupted file."
            )

        # convert the image from BGR to RGB format
        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error converting image to RGB: {str(e)}"
            )

        # log the successful loading of the image
        self.logger.info(
            f"Image loaded successfully. Filename: {file.filename}, Size: {len(file_content) / (1024 * 1024):.2f} MB"
        )

        return image_rgb

    def size_validation(self, image: np.ndarray) -> bool:
        """
        Validates the dimensions of the image to ensure it meets the minimum width and height requirements.
        """
        # Get the height and width of the image
        height, width = image.shape[:2]

        # check if the image dimensions are smaller than the minimum requirements
        if width < self.min_width or height < self.min_height:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions are too small. Minimum required size is {self.min_width}x{self.min_height} pixels. Uploaded image size is {width}x{height} pixels.",
            )

        # Log the successful validation of the image size
        self.logger.info(
            f"Image size validation passed. Image dimensions: {width}x{height} pixels."
        )

        # If all checks pass, return True
        return True

    def faces_detection(self, image: np.ndarray) -> Dict[str, Union[int, List[Dict]]]:
        """
        Detects faces in the given image using OpenCV's Haar Cascade classifier. Returns a list of detected faces with their coordinates and confidence scores.
        """

        # detect faces using insightface.app.FaceAnalysis
        try:
            detections = self.face_detector.get(image)
            self.logger.info(
                f"Face detection completed. Number of faces detected: {len(detections)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error during face detection: {str(e)}"
            )

        # Handle empty detections
        if not detections or len(detections) == 0:
            raise HTTPException(
                status_code=404, detail="No faces detected in the image."
            )

        # Convert insightface Face objects to dicts
        # insightface returns list of Face objects with attributes:
        # - bbox: [x1, y1, x2, y2]
        # - det_score: detection confidence
        # - kps: keypoints array (5 points: 2 eyes, nose, 2 mouth corners)

        faces = []
        for det in detections:
            try:
                # Normalize bbox to [x1, y1, x2, y2] format
                bbox = det.bbox.astype(int).tolist()

                # Extract landmarks from keypoints (kps)
                # kps is shape (5, 2): [[x0, y0], [x1, y1], ..., [x4, y4]]
                kps = det.kps.astype(int)
                landmarks = {
                    "left_eye": tuple(kps[0]),
                    "right_eye": tuple(kps[1]),
                    "nose": tuple(kps[2]),
                    "left_mouth": tuple(kps[3]),
                    "right_mouth": tuple(kps[4]),
                }

                # Append the face information to the dict
                face_dict = {
                    "bbox": bbox,
                    "bbox_format": "xyxy",  # Explicit format marker [x1, y1, x2, y2]
                    "score": float(det.det_score),
                    "landmarks": landmarks,
                }

                faces.append(face_dict)
            except Exception as e:
                self.logger.error(f"Error processing detected face: {str(e)}")
                continue

        # check if any valid faces never detected after processing
        if not faces:
            raise HTTPException(
                status_code=404, detail="No valid faces detected in the image."
            )

        # Filter by confidence threshold
        confident_faces = [
            face for face in faces if face["score"] >= self.min_confidence
        ]

        # check if any faces meet the confidence threshold
        if not confident_faces:
            raise HTTPException(
                status_code=404,
                detail=f"No faces detected with confidence above {self.min_confidence}.",
            )

        self.logger.info(
            f"Number of faces after confidence filtering: {len(confident_faces)} confident face(s) out of {len(faces)}"
        )

        # Return the results in a structured format
        return {
            "faces_count": len(confident_faces),
            "faces": confident_faces,
        }

    def single_face_validation(self, faces_info: List[Dict]) -> Dict:
        """
        Validates that there is exactly one face detected in the image. If multiple faces are detected, it raises an HTTPException with a 400 status code.
        """

        # check if the number of faces detected is equal to 0
        if len(faces_info) == 0:
            raise HTTPException(
                status_code=404, detail="No faces detected in the image."
            )

        # check if the number of faces detected is greater than 1
        if len(faces_info) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"Multiple faces detected in the image. Number of faces detected: {len(faces_info)}. Please upload an image with only one face.",
            )

        # Log the successful validation of a single face
        self.logger.info("Single face validation passed. Exactly one face detected.")

        # If exactly one face is detected, return the face information
        return faces_info[0]

    def face_quality_checks(self, image: np.ndarray, face_info: Dict) -> np.ndarray:
        """Performs quality checks on the detected face, including size, aspect ratio, and confidence score. If the face does not meet the quality criteria, it raises an HTTPException with a 400 status code."""

        # check if image is None or empty
        if image is None or image.size == 0:
            raise HTTPException(
                status_code=400, detail="Invalid image: Image data is empty."
            )

        # check if face_info is None or empty
        if face_info is None or not isinstance(face_info, dict):
            raise HTTPException(
                status_code=400,
                detail="Invalid face information: Face info is empty or not a dictionary.",
            )

        # Get the height and width of the image
        image_height, image_width = image.shape[:2]

        # Extract the bounding box coordinates from the face information
        bbox = face_info.get("bbox")
        if bbox is None:
            raise ValueError("Bounding box information is missing in face_info.")

        # Validate the format of the bounding box
        try:
            x1, y1, x2, y2 = [
                int(v) for v in bbox
            ]  # v is expected to be in [x1, y1, x2, y2] format
            width = x2 - x1
            height = y2 - y1
        except Exception as e:
            raise ValueError(
                f"Invalid bounding box format: {str(e)}. Expected format is [x1, y1, x2, y2]."
            )

        # validate bbox is within image bounds
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            raise HTTPException(
                status_code=400,
                detail=f"Bounding box coordinates are out of image bounds. Image dimensions: {image_width}x{image_height}, Bounding box: [{x1}, {y1}, {x2}, {y2}]",
            )

        # validate bbox has positive width and height
        if width <= 0 or height <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid bounding box dimensions. Width and height must be positive. Bounding box: [{x1}, {y1}, {x2}, {y2}]",
            )

        # validate face size
        face_area = width * height
        image_area = image_width * image_height
        face_ratio = face_area / image_area

        # check if the face area is smaller than the minimum required area
        if face_ratio < self.min_face_ratio:
            raise ValueError(
                f"Face area is too small relative to the image size. Minimum required face ratio: {self.min_face_ratio:.4f}%, Detected face ratio: {face_ratio:.4f}%."
            )

        # Crop face region from the image using the bounding box coordinates
        face_region = image[y1:y2, x1:x2]
        if face_region.size == 0:
            raise HTTPException(
                status_code=400,
                detail="Cropped face region is empty. Please ensure the bounding box coordinates are correct.",
            )

        # Validate detection confidence
        confidence_detected = face_info.get("score")
        if confidence_detected is None:
            raise ValueError("Detection confidence score is missing in face_info.")

        if float(confidence_detected) < self.min_confidence:
            raise ValueError(
                f"Face detection confidence is too low. Minimum required confidence: {self.min_confidence}, Detected confidence: {confidence_detected}."
            )

        # Log the successful validation of face quality
        self.logger.info(
            f"Face quality checks passed. Face area ratio: {face_ratio:.4f}%, Detection confidence: {confidence_detected}."
        )

        return face_region

    def face_alignment(self, image: np.ndarray, face_info: Dict) -> np.ndarray:
        """Aligns the detected face in the image based on the positions of the eyes. This function uses the coordinates of the left and right eyes to calculate the angle of rotation needed to align the face horizontally. It then applies an affine transformation to rotate the image accordingly. If the alignment process fails, it raises an HTTPException with a 500 status code."""

        # Extract and validate landmarks
        landmarks = face_info.get("landmarks")

        # check if landmarks are present and valid
        if landmarks is None or not isinstance(landmarks, dict):
            raise HTTPException(
                status_code=400,
                detail="Landmark information is missing or invalid in face_info.",
            )

        # Extract eye coordinates and validate their presence
        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")
        if left_eye is None or right_eye is None:
            raise HTTPException(
                status_code=400,
                detail="Eye landmarks are missing in face_info. Both left and right eye coordinates are required for alignment.",
            )

        # Calculate the center points of the eyes
        try:
            left_eye_center = (float(left_eye[0]), float(left_eye[1]))
            right_eye_center = (float(right_eye[0]), float(right_eye[1]))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid eye landmark coordinates: {str(e)}. Eye coordinates should be in the format (x, y).",
            )

        # Validate landmarks are within image bounds
        image_height, image_width = image.shape[:2]

        # loop through all landmarks and check if they are within image bounds
        for point, name in [
            (left_eye_center, "left_eye"),
            (right_eye_center, "right_eye"),
        ]:
            if not (0 <= point[0] < image_width and 0 <= point[1] < image_height):
                raise HTTPException(
                    status_code=400,
                    detail=f"{name} landmark coordinates are out of image bounds. Image dimensions: {image_width}x{image_height}, {name} coordinates: {point}",
                )

        # Compute rotation angle from eye positions
        delta_x = right_eye_center[0] - left_eye_center[0]
        delta_y = right_eye_center[1] - left_eye_center[1]
        angle = float(np.degrees(np.arctan2(delta_y, delta_x)))

        # Rotation center = midpoint between eyes
        center = (
            (left_eye_center[0] + right_eye_center[0]) / 2.0,
            (left_eye_center[1] + right_eye_center[1]) / 2.0,
        )

        # Build rotation matrix and rotate image
        h, w = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        aligned_image = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Extract and transform bbox
        bbox = face_info.get("bbox")
        if bbox is None:
            raise ValueError("Bounding box not found")

        try:
            # bbox is [x1, y1, x2, y2] from insightface
            x1_orig, y1_orig, x2_orig, y2_orig = [int(v) for v in bbox]
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid bounding box: {bbox}. Expected format is [x1, y1, x2, y2]."
            )

        # Transform bbox corners through rotation matrix
        corners = np.array(
            [
                [x1_orig, y1_orig, 1],
                [x2_orig, y1_orig, 1],
                [x1_orig, y2_orig, 1],
                [x2_orig, y2_orig, 1],
            ]
        ).T  # Shape: (3, 4)

        rotated_corners = rotation_matrix @ corners

        # Get new bbox from rotated corners
        x1 = int(np.floor(np.min(rotated_corners[0, :])))
        y1 = int(np.floor(np.min(rotated_corners[1, :])))
        x2 = int(np.ceil(np.max(rotated_corners[0, :])))
        y2 = int(np.ceil(np.max(rotated_corners[1, :])))

        # Clamp to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # validate the new bbox has positive width and height after transformation
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid transformed bbox: [{x1}, {y1}, {x2}, {y2}]")

        # Extract aligned face region
        aligned_face = aligned_image[y1:y2, x1:x2]

        # Resize to ArcFace standard (112x112)
        aligned_face = cv2.resize(
            aligned_face, (112, 112), interpolation=cv2.INTER_LINEAR
        )

        self.logger.info(
            f"Face alignment completed. Rotation angle: {angle:.2f} degrees. Aligned face size: {aligned_face.shape[1]}x{aligned_face.shape[0]} pixels."
        )

        return aligned_face

    def blur_validation(self, face_region: np.ndarray) -> bool:
        """
        Validates the blur level of the face region.
        """
        # check if face_region is None or empty
        if face_region is None or face_region.size == 0:
            raise ValueError("Invalid face region: empty image")

        # Convert to grayscale
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_region.copy()  # already grayscale

        # Compute Laplacian variance (sharpness metric)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # check if the variance is below the blur threshold
        if variance < self.blur_threshold:
            raise ValueError(
                f"Face region is too blurry. Variance of Laplacian: {variance:.2f}, Threshold: {self.blur_threshold}."
            )

        self.logger.info(
            f"Blur validation passed. Variance of Laplacian: {variance:.2f}."
        )

        return True

    def brightness_validation(self, face_region: np.ndarray) -> bool:
        """
        Validates the brightness level of the face region.
        """

        # check if face_region is None or empty
        if face_region is None or face_region.size == 0:
            raise ValueError("Invalid face region: empty image")

        # Convert to grayscale
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_region.copy()  # already grayscale

        # Measure brightness using mean pixel intensity
        mean_intensity = float(gray.mean())
        if mean_intensity < self.min_brightness or mean_intensity > self.max_brightness:
            raise ValueError(
                f"Face region brightness is not within the acceptable range. Minimum brightness: {self.min_brightness}, Maximum brightness: {self.max_brightness}, Detected mean brightness: {mean_intensity:.2f}."
            )

        self.logger.info(
            f"Brightness validation passed. Mean intensity: {mean_intensity:.2f}."
        )

        return True
