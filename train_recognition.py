import os
import numpy as np
from sklearn.svm import SVC
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define paths
embedding_dir = "media/embeddings/"
model_output_path = "media/models/face_classifier.pkl"

# Create folders if they don't exist
os.makedirs(embedding_dir, exist_ok=True)
os.makedirs("media/models/", exist_ok=True)

logger.debug(f"Checking embedding directory: {embedding_dir}")
logger.debug(f"Files in directory: {os.listdir(embedding_dir)}")

# Collect embeddings and labels
labels = []
embeddings = []

for file in os.listdir(embedding_dir):
    if file.endswith(".npy"):
        try:
            file_path = os.path.join(embedding_dir, file)
            logger.debug(f"Processing file: {file_path}")

            emb = np.load(file_path)
            logger.debug(f"Loaded embedding shape: {emb.shape}")

            # Check if it's a valid embedding
            if emb.ndim == 1 and emb.shape[0] == 512:  # Update size to 512 for InceptionResnetV1
                embeddings.append(emb)
                student_id = file.replace(".npy", "")
                labels.append(student_id)
                logger.debug(f"Valid embedding found for student: {student_id}")
            else:
                logger.warning(f"Invalid embedding shape in {file}: {emb.shape}")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}", exc_info=True)

# Convert to NumPy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

logger.debug(f"Total embeddings collected: {len(embeddings)}")
logger.debug(f"Labels collected: {labels}")

# Check if we have enough data
if len(embeddings) == 0:
    raise ValueError("❌ No valid embeddings found in 'media/embeddings/'. Train data by registering students first.")

# Train SVM classifier
logger.info("Training face recognition model...")
clf = SVC(probability=True)
clf.fit(embeddings, labels)

# Save the trained classifier
joblib.dump(clf, model_output_path)
logger.info(f"Model trained and saved to: {model_output_path}")