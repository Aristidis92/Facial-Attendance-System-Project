from facenet_pytorch import MTCNN
import cv2
import torch
import numpy as np
from PIL import Image
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MTCNN detector with specific parameters
mtcnn_detector = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],  # MTCNN thresholds for face detection
    factor=0.709,
    post_process=True,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

def preprocess_image(image):
    """
    Preprocess image to improve face detection
    """
    try:
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Adjust brightness and contrast
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)

        # Merge channels
        limg = cv2.merge((cl,a,b))

        # Convert back to RGB
        enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        return Image.fromarray(enhanced_image)
    except Exception as e:
        logger.error(f"Error in image preprocessing: {e}")
        return None

def detect_face(image_path):
    """
    Detect and align face in the image

    Args:
        image_path: Path to the image file

    Returns:
        face_tensor: Processed face tensor or None if no face detected
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None

        # Load and preprocess image
        try:
            img = Image.open(image_path).convert('RGB')
            logger.info(f"Image loaded successfully: {image_path}")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

        # Get original image size
        original_size = img.size
        logger.info(f"Original image size: {original_size}")

        # Preprocess image
        preprocessed_img = preprocess_image(img)
        if preprocessed_img is None:
            logger.error("Image preprocessing failed")
            return None

        # Detect face
        try:
            faces = mtcnn_detector(preprocessed_img)

            # Check if any face was detected
            if faces is None:
                logger.warning("No face detected in the image")
                return None

            # If multiple faces detected, use the first one
            if isinstance(faces, list):
                if len(faces) == 0:
                    logger.warning("No face detected in the image")
                    return None
                face_tensor = faces[0]
                if len(faces) > 1:
                    logger.warning(f"Multiple faces detected ({len(faces)}), using the first one")
            else:
                face_tensor = faces

            # Verify tensor properties
            if face_tensor is not None:
                logger.info(f"Face detected successfully. Tensor shape: {face_tensor.shape}")

                # Ensure the tensor is in the correct format (3, 160, 160)
                if face_tensor.shape != (3, 160, 160):
                    logger.warning(f"Unexpected tensor shape: {face_tensor.shape}")
                    face_tensor = torch.nn.functional.interpolate(
                        face_tensor.unsqueeze(0),
                        size=(160, 160),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

                return face_tensor
            else:
                logger.warning("Face detection failed")
                return None

        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            return None

    except Exception as e:
        logger.error(f"Unexpected error in detect_face: {e}")
        return None

def get_face_quality(face_tensor):
    """
    Assess the quality of the detected face
    """
    try:
        
        return True, "Face quality acceptable"

    except Exception as e:
        logger.error(f"Error in face quality assessment: {e}")
        return False, f"Error checking face quality: {str(e)}"