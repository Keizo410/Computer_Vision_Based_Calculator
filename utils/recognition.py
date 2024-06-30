from uu import Error
import matplotlib
import torch
import cv2
from digit_recognizer.model.cnn import Model
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from digit_recognizer.model.dataset import InferenceDataset, CustomDataset

class HandwritingRecognizer:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Model().to(self.device)

        # Load the trained model and class mappings
        checkpoint_path = "./digit_recognizer/model/handwritten_digit_operator_recognizer.pth"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Checkpoint structure check and loading
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.class_to_idx = checkpoint['class_to_idx']
        else:
            self.model.load_state_dict(checkpoint)
            self.class_to_idx = checkpoint.get('class_to_idx', {'%': 0, '+': 1, '-': 2, '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10, '8': 11, '9': 12, '[': 13, ']': 14, '_': 15})

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.model.eval()

        # No need to define transform here; it will be handled by InferenceDataset

    def recognize(self, image_tensor):

        image_tensor = image_tensor.to(self.device)

        # Predict the class (digit/operator) using the model
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_label = self.idx_to_class[predicted_class]

        return predicted_label

    #     return crop_characters
    def segment_elements(self, image):
        # Segment elements and load into InferenceDataset
        if type(image) == np.ndarray:
            image = image
        else:
            image = cv2.imread(image)

        # Ensure image is in grayscale
        if len(image.shape) == 3 and image.shape[2] in {3, 4}:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        elif len(image.shape) == 2:
            gray_image = image.astype(np.uint8)  # Already grayscale
        else:
            raise ValueError("Unexpected number of channels in input image.")
        
        print(image.shape)
         # Convert image to grayscale
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crop_characters = []

        for c in self.sort_contours(contours):
            c = cv2.convexHull(c)
            (x, y, w, h) = cv2.boundingRect(c)
            # Aspect ratio check
            aspect_ratio = w / h
            if aspect_ratio > 2: 
               # Square box size based on width
                square_size = w

                # Calculate center coordinates for square box
                center_x = x + w // 2
                center_y = y + h // 2

                # Adjust top-left corner coordinates (considering padding if needed)
                top_left_x = center_x - square_size // 2
                top_left_y = center_y - square_size // 2

                # Ensure coordinates stay within image boundaries
                top_left_x = max(0, top_left_x)  # Non-negative x
                top_left_y = max(0, top_left_y)  # Non-negative y
                bottom_right_x = min(image.shape[1], top_left_x + square_size)
                bottom_right_y = min(image.shape[0], top_left_y + square_size)

                # Crop character with square box
                cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                crop_characters.append(cropped_image)
            else:
                crop_characters.append(image[y:y+h+1, x:x+w+1])

        print(f"Detected {len(crop_characters)} elements...")
        
        return crop_characters

    @staticmethod
    def sort_contours(contours, reverse=False):
        i = 0
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        sorted_contours = sorted(zip(contours, bounding_boxes), key=lambda b: b[1][i], reverse=reverse)
        return [cnt for cnt, bbox in sorted_contours]
    
    def visualize_predictions(self, image):
        cropped_characters = self.segment_elements(image)
        predictions = []
        print("hello")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((28, 28)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Try to catch any errors here
        try:
            inference_dataset = InferenceDataset(image_arrays=cropped_characters, transform=transform)
            print("InferenceDataset created successfully with {} images.".format(len(inference_dataset)))
        except Exception as e:
            print(f"Error creating InferenceDataset: {e}")
            return
        
        for i in range(inference_dataset.__len__()):
            try:
                print(f"Processing image {i + 1}/{inference_dataset.__len__()}")
                char_img = inference_dataset.__getitem__(i)
                print(f"Image {i + 1} loaded successfully. Type: {type(char_img)}")
                
                char_tensor = char_img.unsqueeze(0)  # Need this for the model
                
                recognized_label = self.recognize(char_tensor)  # Perform recognition
                predictions.append(recognized_label)
                print(f"Pred: {recognized_label}")
            except Exception as e:
                print(f"Error during prediction for image {i + 1}: {e}")

        # fig, axes = plt.subplots(1, len(predictions), figsize=(15, 5))
        # for ax, char_img, pred in zip(axes, cropped_characters, predictions):
        #     ax.imshow(char_img.squeeze(), cmap='gray')
        #     ax.set_title(f"Pred: {pred}")
        #     ax.axis('off')
        
        # plt.show()

        return predictions
