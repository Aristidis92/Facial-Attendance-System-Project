import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import numpy as np
import logging
import os
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load pretrained FaceNet model
try:
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    logger.info("FaceNet model loaded successfully")
except Exception as e:
    logger.error(f"Error loading FaceNet model: {e}")
    raise

def get_face_embedding(face_tensor):
    """
    Generate embedding for a face tensor

    Args:
        face_tensor: PyTorch tensor of shape (3, 160, 160)

    Returns:
        numpy array: Face embedding vector of shape (512,)
    """
    try:
        if face_tensor is None:
            logger.error("No face tensor provided")
            return None

        # Verify tensor shape
        if face_tensor.shape != (3, 160, 160):
            logger.error(f"Invalid tensor shape: {face_tensor.shape}, expected (3, 160, 160)")
            return None

        # Ensure tensor is on the correct device
        face_tensor = face_tensor.to(device)

        # Normalize tensor
        face_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(face_tensor)

        # Add batch dimension and get embedding
        with torch.no_grad():
            face_tensor = face_tensor.unsqueeze(0)
            embedding = facenet(face_tensor)

        # Convert to numpy array
        embedding_np = embedding.cpu().detach().numpy()[0]

        # Normalize embedding
        embedding_np = embedding_np / np.linalg.norm(embedding_np)

        logger.info("Face embedding generated successfully")
        return embedding_np

    except Exception as e:
        logger.error(f"Error generating face embedding: {e}")
        return None

def compare_embeddings(embedding1, embedding2):
    """
    Compare two face embeddings using cosine similarity

    Args:
        embedding1: First face embedding numpy array
        embedding2: Second face embedding numpy array

    Returns:
        float: Similarity score between 0 and 1
    """
    try:
        if embedding1 is None or embedding2 is None:
            logger.error("Invalid embeddings provided")
            return 0.0

        # Reshape embeddings if needed
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0][0]

        logger.info(f"Similarity score: {similarity:.4f}")
        return similarity

    except Exception as e:
        logger.error(f"Error comparing embeddings: {e}")
        return 0.0

def save_embedding(embedding, filepath):
    """
    Save face embedding to file

    Args:
        embedding: Face embedding numpy array
        filepath: Path to save the embedding

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save embedding
        np.save(filepath, embedding)
        logger.info(f"Embedding saved successfully to {filepath}")
        return True

    except Exception as e:
        logger.error(f"Error saving embedding: {e}")
        return False

def load_embedding(filepath):
    """
    Load face embedding from file

    Args:
        filepath: Path to the embedding file

    Returns:
        numpy array: Face embedding vector or None if error
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"Embedding file not found: {filepath}")
            return None

        embedding = np.load(filepath)
        logger.info(f"Embedding loaded successfully from {filepath}")
        return embedding

    except Exception as e:
        logger.error(f"Error loading embedding: {e}")
        return None

def verify_face(new_embedding, stored_embedding_path, threshold=0.8):
    """
    Verify if two faces match

    Args:
        new_embedding: New face embedding numpy array
        stored_embedding_path: Path to stored embedding file
        threshold: Similarity threshold for verification (default: 0.8)

    Returns:
        tuple: (bool, float) - (is_match, similarity_score)
    """
    try:
        stored_embedding = load_embedding(stored_embedding_path)
        if stored_embedding is None:
            return False, 0.0

        similarity = compare_embeddings(new_embedding, stored_embedding)
        is_match = similarity > threshold

        logger.info(f"Face verification result: {'Match' if is_match else 'No match'} "
                   f"(similarity: {similarity:.4f})")
        return is_match, similarity

    except Exception as e:
        logger.error(f"Error in face verification: {e}")
        return False, 0.0