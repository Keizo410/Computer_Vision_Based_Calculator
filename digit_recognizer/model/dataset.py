import os
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        self.images = self.load_dataset()

    def load_dataset(self):
        images = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                images.append((img_path, self.class_to_idx[cls_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        # image = Image.open(img_path).convert('RGB')
        image = self.segment_elements(img_path)

        # Check data type before inversion (optional)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)  # Convert to uint8 if necessary

        image = 255 - image  # Invert color assuming uint8 data type

        if image.shape[-1] == 1:
            image = image  # Already grayscale, no conversion needed
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Assuming BGR

        image = cv2.GaussianBlur(image, (3, 3), 0)

        # image = Image.fromarray(blur_image)

        if self.transform:
            image = self.transform(image)

        return image, label
        
    def get_class_name(self, idx):
        return self.idx_to_class[idx]
    
    def segment_elements(self, image_path):
        # Segment elements from the image
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Image at {image_path} could not be read")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the largest contour assuming it's the character
        c = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        x, y, w, h = cv2.boundingRect(hull)

        aspect_ratio = w / h
        if aspect_ratio > 2:
            square_size = max(w, h)
            center_x = x + w // 2
            center_y = y + h // 2

            top_left_x = max(0, center_x - square_size // 2)
            top_left_y = max(0, center_y - square_size // 2)
            bottom_right_x = min(image.shape[1], top_left_x + square_size)
            bottom_right_y = min(image.shape[0], top_left_y + square_size)

            cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        else:
            cropped_image = image[y:y + h, x:x + w]

        return cropped_image
    

class InferenceDataset(Dataset):
    def __init__(self, image_arrays, transform=None):
        self.transform = transform
        self.image_arrays = image_arrays

    def __len__(self):
        return len(self.image_arrays)

    def __getitem__(self, idx):

        image = self.image_arrays[idx]

        # Check data type before inversion (optional)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)  # Convert to uint8 if necessary

        image = 255 - image  # Invert color assuming uint8 data type
        
        # print(image.shape)
        # if image.shape[-1] == 1:
        #     image = image  # Already grayscale, no conversion needed
        # else:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAT)  # Assuming BGR
        # Convert to grayscale if necessary
        if len(image.shape) == 2:  # Check if image is grayscale
            image = image[:, :, np.newaxis]  # Add channel dimension
        elif len(image.shape) == 3 and image.shape[2] in {3, 4}:  # Check for RGB or RGBA
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
            image = image[:, :, np.newaxis]  # Add channel dimension
        else:
            raise ValueError("Unexpected number of channels in input image.")

        image = cv2.GaussianBlur(image, (3, 3), 0)
        if self.transform:
            image = self.transform(image)  # Apply transformations on NumPy array

        return image

