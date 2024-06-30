import os
import random
from shutil import copyfile

from matplotlib import gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from model.cnn import Model
from PIL import Image, ImageOps
import numpy as np
import cv2
from sklearn.metrics import precision_score, f1_score
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from model.dataset import CustomDataset

def split_dataset(dataset_path):

    # Define paths to your dataset
    source_dir = dataset_path
    train_dir = 'dataset/dataset/train'
    val_dir = 'dataset/dataset/validation'

    # Create directories for training and validation data if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Define the percentage of data to use for validation
    val_split = 0.3

    # Iterate over each class directory
    for class_name in os.listdir(source_dir):

        class_dir = os.path.join(source_dir, class_name)
        
        # Create directories for this class in training and validation sets
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Get a list of all images in this class
        images = os.listdir(class_dir)
        
        # Shuffle the list of images
        random.shuffle(images)
        
        # Split the images into training and validation sets
        num_val = int(len(images) * val_split)
        val_images = images[:num_val]
        train_images = images[num_val:]
        
        # Copy images to their respective directories
        for image in train_images:
            source_path = os.path.join(class_dir, image)
            target_path = os.path.join(train_class_dir, image)
            copyfile(source_path, target_path)
            
        for image in val_images:
            source_path = os.path.join(class_dir, image)
            target_path = os.path.join(val_class_dir, image)
            copyfile(source_path, target_path)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_precisions, val_precisions = [], []
    train_f1_scores, val_f1_scores = [], []

    for epoch in range(num_epochs):
        print("model train at epoch = ", epoch)
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store labels and predictions for precision and F1 score
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        epoch_precision = precision_score(all_labels, all_predictions, average='macro')
        epoch_f1_score = f1_score(all_labels, all_predictions, average='macro')

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        train_precisions.append(epoch_precision)
        train_f1_scores.append(epoch_f1_score)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Precision: {epoch_precision:.4f}, F1 Score: {epoch_f1_score:.4f}")

        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_predictions = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Store labels and predictions for precision and F1 score
                val_labels.extend(labels.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_accuracy = val_correct / val_total
        val_epoch_precision = precision_score(val_labels, val_predictions, average='macro')
        val_epoch_f1_score = f1_score(val_labels, val_predictions, average='macro')

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        val_precisions.append(val_epoch_precision)
        val_f1_scores.append(val_epoch_f1_score)

        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}, Precision: {val_epoch_precision:.4f}, F1 Score: {val_epoch_f1_score:.4f}")

    # Plotting accuracy and loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epochs')
    plt.savefig('loss_plot.png')  # Save the loss plot

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs Epochs')
    plt.savefig('accuracy_plot.png')  # Save the accuracy plot

    plt.close()  # Close the plot to avoid displaying it

    # Plotting precision and F1 score
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_precisions, label='Train Precision')
    plt.plot(range(1, num_epochs + 1), val_precisions, label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Precision vs Epochs')
    plt.savefig('precision_plot.png')  # Save the precision plot

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_f1_scores, label='Train F1 Score')
    plt.plot(range(1, num_epochs + 1), val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score vs Epochs')
    plt.savefig('f1_score_plot.png')  # Save the F1 score plot

    plt.close()  # Close the plot to avoid displaying it

def check():

    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize images to 28x28
        transforms.ToTensor(),  # Convert PIL Image to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor
    ])
    dataset = CustomDataset('dataset/train', transform = transform)
    
    # Accessing class_to_idx and idx_to_class
    class_to_idx = dataset.class_to_idx
    idx_to_class = dataset.idx_to_class

    print("clssToInd: ",class_to_idx)
    print("idToClass: ",idx_to_class)

    cols = 3
    rows = 3

    fig = plt.figure(figsize=(10, 10))
    plt.rcParams.update({"font.size":15})
    grid = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig)

    np.random.seed(45)
    rand = np.random.randint(0, dataset.__len__(), size= (cols*rows))

    for i in range(cols*rows):
        fig.add_subplot(grid[i])
        image, label = dataset.__getitem__(rand[i])
        plt.title(dataset.get_class_name(label))
        plt.axis(False)
        plt.imshow(image.permute(1, 2, 0).numpy())  # Convert CHW to HWC format for imshow
    
    plt.show()

def main():
    

    # # Split the dataset
    # split_dataset(dataset_path)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to tensor
        transforms.Resize((28, 28)),  # Resize images to 28x28
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor
    ])

    # Load datasets
    print("Loading datasets...")
    train_dataset = CustomDataset('dataset/train', transform = transform)
    val_dataset = CustomDataset('dataset/validation', transform=transform)

    # Define dataloaders
    print("Defining dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4*torch.cuda.device_count(), pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4*torch.cuda.device_count(), pin_memory = True)

    # Load pre-trained ResNet model
    print("Loading pre-trained weights & setting some stuff...")
    
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Train the model
    print("Training starts...")
    train_model(model, criterion, optimizer, train_loader, val_loader)

    # Save the trained model
    print("Saving the model parameter...")
    torch.save(model.state_dict(), 'model/handwritten_digit_operator_recognizer.pth')

if __name__ == "__main__":
    main()
    # check()
