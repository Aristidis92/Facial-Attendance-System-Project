import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColorDepthDataset(Dataset):
    def __init__(self, color_dir, depth_dir, color_transform=None, depth_transform=None):
        """
        Args:
            color_dir (str): Path to the color dataset directory
            depth_dir (str): Path to the depth dataset directory
            color_transform (callable, optional): Transform to be applied on color images
            depth_transform (callable, optional): Transform to be applied on depth images
        """
        self.color_dir = color_dir
        self.depth_dir = depth_dir
        self.color_transform = color_transform
        self.depth_transform = depth_transform

        # Verify directories exist
        if not os.path.exists(color_dir):
            raise ValueError(f"Color directory not found: {color_dir}")
        if not os.path.exists(depth_dir):
            raise ValueError(f"Depth directory not found: {depth_dir}")

        # Get all color and depth image paths
        self.color_images = []
        self.depth_images = []
        self.labels = []

        # Supported image extensions
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

        for label, class_name in enumerate(['live', 'spoof']):
            color_class_dir = os.path.join(color_dir, class_name)
            depth_class_dir = os.path.join(depth_dir, class_name)

            # Verify class directories exist
            if not os.path.exists(color_class_dir):
                raise ValueError(f"Color class directory not found: {color_class_dir}")
            if not os.path.exists(depth_class_dir):
                raise ValueError(f"Depth class directory not found: {depth_class_dir}")

            # Get list of files in both directories
            color_files = [f for f in os.listdir(color_class_dir)
                         if f.lower().endswith(self.image_extensions)]
            depth_files = [f for f in os.listdir(depth_class_dir)
                         if f.lower().endswith(self.image_extensions)]

            logger.info(f"Found {len(color_files)} color images and {len(depth_files)} "
                       f"depth images in {class_name} class")

            # Sort files to ensure alignment
            color_files.sort()
            depth_files.sort()

            # Match color and depth files
            for color_file in color_files:
                color_base = os.path.splitext(color_file)[0]

                # Try to find matching depth file with any supported extension
                depth_file = None
                for ext in self.image_extensions:
                    potential_depth_file = color_base + ext
                    if potential_depth_file in depth_files:
                        depth_file = potential_depth_file
                        break

                if depth_file:
                    color_path = os.path.join(color_class_dir, color_file)
                    depth_path = os.path.join(depth_class_dir, depth_file)

                    # Verify files are valid images
                    try:
                        with Image.open(color_path) as img:
                            if img.mode not in ['RGB', 'L']:
                                logger.warning(f"Skipping invalid color image: {color_path}")
                                continue
                        with Image.open(depth_path) as img:
                            if img.mode not in ['RGB', 'L']:
                                logger.warning(f"Skipping invalid depth image: {depth_path}")
                                continue
                    except Exception as e:
                        logger.warning(f"Error reading image pair: {str(e)}")
                        continue

                    self.color_images.append(color_path)
                    self.depth_images.append(depth_path)
                    self.labels.append(label)

        # Verify we found valid pairs
        if len(self.color_images) == 0:
            raise ValueError("No valid image pairs found in the dataset directories")

        logger.info(f"Successfully loaded {len(self.color_images)} valid image pairs")
        logger.info(f"Class distribution - Real: {self.labels.count(0)}, Fake: {self.labels.count(1)}")

    def __len__(self):
        return len(self.color_images)

    def __getitem__(self, idx):
        try:
            # Load color image
            color_image = Image.open(self.color_images[idx]).convert('RGB')

            # Load depth image as grayscale
            depth_image = Image.open(self.depth_images[idx]).convert('L')

            # Apply transforms if available
            if self.color_transform:
                color_image = self.color_transform(color_image)
            if self.depth_transform:
                depth_image = self.depth_transform(depth_image)

            # Add unsqueeze to make label shape match model output
            label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)

            return color_image, depth_image, label

        except Exception as e:
            logger.error(f"Error loading image pair at index {idx}: {str(e)}")
            raise