import os
import numpy as np
import torch
from torch import nn
import math
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils import data
from PIL import Image

class AlexNet(torch.nn.Module):   
    def  __init__(self, input_channels = 1, output_size = 10):
        super().__init__()
        self.conv1 = torch.nn.Sequential(   #input_size = 227*227*1
            torch.nn.Conv2d(input_channels, 96, 11, 4, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2) #output_size = 27*27*96
        )
        self.conv2 = torch.nn.Sequential(   #input_size = 27*27*96
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)    #output_size = 13*13*256
        )
        self.conv3 = torch.nn.Sequential(   #input_size = 13*13*256
            torch.nn.Conv2d(256, 384, 3, 1, 1), 
            torch.nn.ReLU(),    #output_size = 13*13*384
        )
        self.conv4 = torch.nn.Sequential(   #input_size = 13*13*384
            torch.nn.Conv2d(384, 384, 3, 1, 1),
            torch.nn.ReLU(),    #output_size = 13*13*384
        )
        self.conv5 = torch.nn.Sequential(   #input_size = 13*13*384
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)    #output_size = 6*6*256
        )
 
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, output_size)
        )
 
    def forward(self, x): 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        conv5_out = self.conv5(x)
        x = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(x)
        return out

def resize_image(input_size = 28):
    transform = transforms.Compose([
    transforms.Resize(input_size), 
    transforms.ToTensor()
    ])
    return transform

def view_datasets(image_loader, objective_list):
    objective_list = np.array(objective_list)
    images, labels = next(iter(image_loader))
    img = torchvision.utils.make_grid(images)
    img = img.numpy().transpose(1, 2, 0)
    print(objective_list[labels.tolist()])
    # plt.axis('off')
    # plt.imshow(img)
    return (images, objective_list[labels.tolist()])

def create_network(network):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res = network
    net = res.to(device)
    return net

def train_model(net, train_loader, LR, epochs=1, number_of_images=None):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    
    # Learning rate scheduler: Decay every 20 epochs by a factor of 0.5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    for epoch in range(epochs):
        net.train()  # Set model to training mode
        sum_loss = 0.0
        total_images = 0

        for i, (inputs, labels) in enumerate(train_loader):
            if number_of_images is not None and total_images >= number_of_images:
                break  # Stop after processing the desired number of images

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients
            outputs = net(inputs)  # Forward pass
            loss = loss_function(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            sum_loss += loss.item() * inputs.size(0)
            total_images += inputs.size(0)

            if i % 100 == 99:
                print('[Epoch %d, Batch %d] Loss: %.03f' %
                      (epoch + 1, i + 1, sum_loss / total_images))
                sum_loss = 0.0
        
        # Step the learning rate scheduler at the end of each epoch
        scheduler.step()
        
        # Log the current learning rate and average loss for the epoch
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = sum_loss / total_images
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

    return net


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
def test_model(net, test_loader, number_of_images=None, class_label=None):
    net.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    all_labels = []
    all_predictions = []
    
    total_images_tested = 0

    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        if class_label is not None:
            mask = (labels == class_label)
            images = images[mask]
            labels = labels[mask]
            if len(labels) == 0:
                continue
        
        if number_of_images is not None and total_images_tested >= number_of_images:
            break

        output_test = net(images)
        _, predicted = torch.max(output_test, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        total_images_tested += labels.size(0)
    
    if total_images_tested == 0:
        print(f"No data found for class {class_label}")
        return {
            "Accuracy": "N/A",
            "Precision": "N/A",
            "Recall": "N/A",
            "F1-Score": "N/A",
            "FP": "N/A",
            "FN": "N/A",
            "TP": "N/A",
            "TN": "N/A",
            "Total_Images": 0
        }

    # Define all possible class labels
    num_classes = len(set([0, 1, 2, 3])) 
    cm = confusion_matrix(all_labels, all_predictions, labels=range(num_classes))
    print("Confusion Matrix:\n", cm)

    if class_label is not None:
        # Single-class evaluation
        tp = cm[class_label, class_label]
        fp = cm[:, class_label].sum() - tp
        fn = cm[class_label, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
    else:
        # Multi-class evaluation
        tp = cm.diagonal()
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (fp + fn + tp)
        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        tn = tn.sum()

    num_samples = len(all_labels)
    num_predictions = sum([1 for pred in all_predictions if pred == class_label]) if class_label is not None else "All"

    return {
        "Accuracy": accuracy_score(all_labels, all_predictions),
        "Precision": precision_score(all_labels, all_predictions, average='weighted', zero_division=0),
        "Recall": recall_score(all_labels, all_predictions, average='weighted', zero_division=0),
        "F1-Score": f1_score(all_labels, all_predictions, average='weighted', zero_division=0),
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "TN": tn,
        "Total_Images": num_samples
    }


def predict_image(net, input_image, objective_list, num_of_prediction=1, true_label=None, dataset=None, image_index=None):
    net.eval()

    # Create reverse mapping for class indices to class labels
    idx_to_class = {v: k for k, v in objective_list.items()}

    # Ensure input_image has the correct shape
    if len(input_image.shape) == 3:  # If (C, H, W), add batch dimension
        input_image = input_image.unsqueeze(0)

    # Move input and model to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    images = input_image.to(device)

    # Get model predictions
    output_test = net(images)
    _, predicted = torch.topk(output_test, num_of_prediction)  # Get top predictions

    # Map predicted indices to class labels
    predicted_classes = [idx_to_class[idx.item()] for idx in predicted[0]]

    # Calculate confidence scores for the predictions
    confidence_scores = torch.softmax(output_test, dim=1)[0][predicted[0]].cpu().detach().numpy()

    # Retrieve true label
    if true_label is None and dataset is not None and image_index is not None:
        true_label_index = dataset.dataset.targets[dataset.indices[image_index]]
        true_label = idx_to_class[true_label_index]

    print(f"Predicted result(s): {predicted_classes}")
    print(f"Confidence: {[f'{conf*100:.2f}%' for conf in confidence_scores]}")
    if true_label:
        print(f"True Label: {true_label}")

    return {
        "Predicted Classes": predicted_classes,
        "Confidence": confidence_scores,
        "True Label": true_label
    }
