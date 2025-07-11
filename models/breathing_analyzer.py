import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import scipy.signal
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.fft import fft, fftfreq
import time

class BreathingCNN(nn.Module):
    """
    Breathing Analyzer using 1D CNN for respiratory pattern analysis
    Analyzes breathing patterns from video or audio signals
    """
    def __init__(self, input_channels=1, sequence_length=300, num_classes=2):
        super(BreathingCNN, self).__init__()
        
        # 1D CNN layers for temporal pattern recognition
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=20, stride=1, padding=10)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=20, stride=1, padding=10)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=10, stride=1, padding=5)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate the size after convolution and pooling
        self.conv_output_size = self._get_conv_output_size(sequence_length)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, num_classes)
        
    def _get_conv_output_size(self, input_size):
        """Calculate output size after convolution layers"""
        size = input_size
        # After conv1 and pool1
        size = size // 2
        # After conv2 and pool2
        size = size // 2
        # After conv3 and pool3
        size = size // 2
        return size * 128
    
    def forward(self, x):
        # x: (batch_size, channels, sequence_length)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class BreathingAnalyzer:
    """Main class for breathing pattern analysis"""
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BreathingCNN().to(self.device)
        
        # Signal processing parameters
        self.sample_rate = 20  # 20 Hz for video analysis
        self.window_size = 15  # 15 seconds window
        self.sequence_length = self.sample_rate * self.window_size
        
        # Breathing signal buffer
        self.signal_buffer = []
        self.timestamps = []
        
        # ROI for chest movement detection
        self.roi = None
        self.background = None
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
    
    def extract_breathing_signal_video(self, image):
        """Extract breathing signal from video using optical flow"""
        if self.roi is None:
            # Auto-detect chest region (simplified)
            h, w = image.shape[:2]
            self.roi = (w//4, h//3, w//2, h//3)  # Center chest region
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract ROI
        x, y, w, h = self.roi
        roi_gray = gray[y:y+h, x:x+w]
        
        # Apply Gaussian blur to reduce noise
        roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        
        # Calculate optical flow magnitude in ROI
        if len(self.signal_buffer) > 0:
            # Compare with previous frame
            prev_roi = self.prev_roi
            flow = cv2.calcOpticalFlowPyrLK(
                prev_roi, roi_gray, None, None,
                winSize=(15, 15), maxLevel=2
            )
            
            # Calculate average flow magnitude
            if flow[0] is not None:
                flow_magnitude = np.mean(np.sqrt(flow[0]**2).sum(axis=2))
            else:
                flow_magnitude = 0
        else:
            flow_magnitude = 0
        
        self.prev_roi = roi_gray.copy()
        
        # Add to buffer
        self.signal_buffer.append(flow_magnitude)
        self.timestamps.append(time.time())
        
        # Maintain buffer size
        if len(self.signal_buffer) > self.sequence_length:
            self.signal_buffer.pop(0)
            self.timestamps.pop(0)
        
        return flow_magnitude
    
    def predict(self, image=None, audio=None):
        """Predict breathing pattern quality"""
        if image is not None:
            # Extract signal from video
            signal_val = self.extract_breathing_signal_video(image)
        else:
            return 0, 0.5
        
        # Need enough data for prediction
        if len(self.signal_buffer) < self.sequence_length:
            return 0, 0.5
        
        # Prepare input tensor
        signal = np.array(self.signal_buffer[-self.sequence_length:])
        # Normalize signal
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        signal = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(signal)
            probabilities = F.softmax(output, dim=1)
            confidence = probabilities.max().item()
            prediction = probabilities.argmax().item()
        
        return prediction, confidence  # 0: Irregular, 1: Regular
