import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
from datetime import datetime

class FaceExpressionCNN(nn.Module):
    """
    Face Expression Analyzer using MobileNet-V3-Small backbone
    Classifies facial expressions into Calm/Not-Calm for meditation detection
    """
    def __init__(self, num_classes=2):
        super(FaceExpressionCNN, self).__init__()
        # Use MobileNet-V3-Small as backbone
        self.backbone = mobilenet_v3_small(pretrained=True)
        
        # Modify the first conv layer to accept grayscale input
        self.backbone.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Modify classifier for binary classification
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class FaceExpressionDataset(Dataset):
    """Dataset class for face expression data"""
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        # Load your dataset here - this is a placeholder
        self.samples = []  # [(image_path, label), ...]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class FaceExpressionAnalyzer:
    """Main class for face expression analysis"""
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FaceExpressionCNN().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
    
    def train_model(self, train_loader, val_loader, epochs=100, learning_rate=0.001):
        """Train the face expression model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        # TensorBoard logging
        writer = SummaryWriter(f'runs/face_expression_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            # Calculate metrics
            train_loss_avg = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            val_loss_avg = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # Store metrics
            train_losses.append(train_loss_avg)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss_avg)
            val_accuracies.append(val_acc)
            
            # Log to TensorBoard
            writer.add_scalar('Loss/Train', train_loss_avg, epoch)
            writer.add_scalar('Loss/Validation', val_loss_avg, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
        
        writer.close()
        
        # Save training metrics
        metrics = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
        
        return metrics
    
    def predict(self, image):
        """Predict if facial expression is calm or not"""
        self.model.eval()
        
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
            probabilities = F.softmax(output, dim=1)
            confidence = probabilities.max().item()
            prediction = probabilities.argmax().item()
        
        return prediction, confidence  # 0: Not-Calm, 1: Calm
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
