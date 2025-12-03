import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Step 1: Define the model
class LivenessNet(nn.Module):
    def __init__(self):
        super(LivenessNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 20 * 20, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x

# Step 2: Define the detector
class LivenessDetector:
    def __init__(self, model_path='models/simple_liveness_net.pt'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LivenessNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        # If path string is provided, open it
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        # If it's already a tensor, skip transforms
        if isinstance(image, torch.Tensor):
            input_tensor = image.unsqueeze(0).to(self.device)
        else:
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            prob = output.item()
            return prob > 0.6, prob


# Step 3: Simple helper
def is_live_face(image_path):
    detector = LivenessDetector()
    is_live, confidence = detector.predict(image_path)

    if is_live:
        print(f"✅ Live face detected! (Confidence: {confidence:.4f})")
    else:
        print(f"⚠️ Spoof detected! (Confidence: {confidence:.4f})")

    return is_live
