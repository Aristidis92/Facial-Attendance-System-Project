import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import logging
import os
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LivenessCNN(nn.Module):
    def __init__(self):
        super(LivenessCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 20 * 20, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)  # Add dropout for regularization

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

class LivenessDetector:
    def __init__(self, model_path='media/models/liveness_cnn.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LivenessCNN().to(self.device)

        try:
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info("✅ Liveness model loaded successfully")
            else:
                logger.error(f"❌ Model file not found at {model_path}")
        except Exception as e:
            logger.error(f"❌ Error loading liveness model: {e}")

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def preprocess_tensor(self, face_tensor):
        """Preprocess the input tensor for liveness detection"""
        try:
            # If tensor is on GPU, move it to CPU
            if face_tensor.device.type == 'cuda':
                face_tensor = face_tensor.cpu()

            # Ensure tensor is float and normalized
            if face_tensor.dtype != torch.float32:
                face_tensor = face_tensor.float()

            if face_tensor.max() > 1.0:
                face_tensor = face_tensor / 255.0

            # Ensure correct shape
            if face_tensor.shape != (3, 160, 160):
                if face_tensor.shape[0] != 3:
                    face_tensor = face_tensor.permute(2, 0, 1)
                face_tensor = F.interpolate(
                    face_tensor.unsqueeze(0),
                    size=(160, 160),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            # Normalize
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            face_tensor = normalize(face_tensor)

            return face_tensor

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return None

    def detect_liveness(self, face_tensor):
        """
        Detect if a face is live or not
        Returns: (is_live, confidence, message)
        """
        try:
            # Preprocess the input
            processed_tensor = self.preprocess_tensor(face_tensor)
            if processed_tensor is None:
                return False, 0.0, "Preprocessing failed"

            # Add batch dimension and move to device
            input_tensor = processed_tensor.unsqueeze(0).to(self.device)

            # Get prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                confidence = output.item()

                # Log raw confidence
                logger.info(f"Raw confidence score: {confidence:.4f}")

                # Adjust confidence threshold
                is_live = confidence > 0.3  # Lowered threshold for testing

                if is_live:
                    message = f"Live face detected (confidence: {confidence:.2f})"
                    logger.info(f"✅ {message}")
                else:
                    message = f"Spoof detected (confidence: {confidence:.2f})"
                    logger.warning(f"⚠️ {message}")

                return is_live, confidence, message

        except Exception as e:
            logger.error(f"❌ Error in liveness detection: {e}")
            return False, 0.0, f"Error processing image: {str(e)}"

def is_live_face(face_tensor):
    """Wrapper function for liveness detection"""
    try:
        detector = LivenessDetector()
        is_live, confidence, message = detector.detect_liveness(face_tensor)

        if not is_live:
            logger.warning(f"⚠️ Liveness check failed: {message}")
            return False

        logger.info(f"✅ Liveness check passed: {message}")
        return True

    except Exception as e:
        logger.error(f"❌ Error in liveness check: {e}")
        return False

# Function to test the model
def test_model(image_path):
    """Test the liveness detection model on a single image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor()
        ])
        face_tensor = transform(image)

        # Test liveness detection
        detector = LivenessDetector()
        is_live, confidence, message = detector.detect_liveness(face_tensor)

        print(f"\nTest Results for {image_path}:")
        print(f"Is Live: {is_live}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Message: {message}")

    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    # Test the model if running this file directly
    test_image_path = "path_to_test_image.jpg"  # Replace with actual test image path
    if os.path.exists(test_image_path):
        test_model(test_image_path)
    else:
        print("Please provide a valid test image path")