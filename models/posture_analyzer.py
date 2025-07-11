import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import mediapipe as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime

class PostureLSTM(nn.Module):
    """
    Posture Analyzer using Bi-LSTM for temporal pose analysis
    Analyzes body posture for meditation alignment
    """
    def __init__(self, input_size=99, hidden_size=128, num_layers=2, num_classes=2):
        super(PostureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bi-LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classify
        output = self.classifier(attended_output)
        return output

class PostureAnalyzer:
    """Main class for posture analysis using MediaPipe and LSTM"""
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PostureLSTM().to(self.device)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Sequence buffer for temporal analysis
        self.sequence_length = 50
        self.pose_buffer = []
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
    
    def extract_pose_features(self, image):
        """Extract pose landmarks from image using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        if results.pose_landmarks:
            # Extract landmark coordinates
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Calculate additional features
            features = self.calculate_posture_features(landmarks)
            return np.array(features)
        else:
            return np.zeros(99)  # Return zero vector if no pose detected
    
    def calculate_posture_features(self, landmarks):
        """Calculate posture-specific features from landmarks"""
        # Convert to numpy array and reshape
        points = np.array(landmarks).reshape(-1, 3)
        
        # Key point indices (MediaPipe pose landmarks)
        nose = points[0]
        left_shoulder = points[11]
        right_shoulder = points[12]
        left_hip = points[23]
        right_hip = points[24]
        left_knee = points[25]
        right_knee = points[26]
        left_ankle = points[27]
        right_ankle = points[28]
        
        features = []
        
        # 1. Spine alignment (shoulder to hip angle)
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        spine_vector = shoulder_center - hip_center
        spine_angle = np.arctan2(spine_vector[1], spine_vector[0])
        features.append(spine_angle)
        
        # 2. Shoulder levelness
        shoulder_diff = left_shoulder[1] - right_shoulder[1]
        features.append(shoulder_diff)
        
        # 3. Hip levelness
        hip_diff = left_hip[1] - right_hip[1]
        features.append(hip_diff)
        
        # 4. Cross-legged sitting detection
        # Angle between thighs
        left_thigh = left_knee - left_hip
        right_thigh = right_knee - right_hip
        thigh_angle = np.arccos(np.dot(left_thigh, right_thigh) / 
                               (np.linalg.norm(left_thigh) * np.linalg.norm(right_thigh)))
        features.append(thigh_angle)
        
        # 5. Knee positions relative to hips
        left_knee_hip_diff = left_knee[1] - left_hip[1]
        right_knee_hip_diff = right_knee[1] - right_hip[1]
        features.extend([left_knee_hip_diff, right_knee_hip_diff])
        
        # 6. Head position relative to shoulders
        head_shoulder_diff = nose[1] - shoulder_center[1]
        features.append(head_shoulder_diff)
        
        # 7. Symmetry measures
        left_side_length = np.linalg.norm(left_ankle - left_shoulder)
        right_side_length = np.linalg.norm(right_ankle - right_shoulder)
        symmetry = abs(left_side_length - right_side_length)
        features.append(symmetry)
        
        # 8. Add raw landmark coordinates (normalized)
        features.extend(landmarks)
        
        return features
    
    def predict(self, image):
        """Predict posture quality from image"""
        # Extract pose features
        features = self.extract_pose_features(image)
        
        # Add to buffer
        self.pose_buffer.append(features)
        if len(self.pose_buffer) > self.sequence_length:
            self.pose_buffer.pop(0)
        
        # Need enough frames for prediction
        if len(self.pose_buffer) < self.sequence_length:
            return 0, 0.5  # Default to not-meditation with low confidence
        
        # Prepare input tensor
        sequence = np.array(self.pose_buffer)
        sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(sequence)
            probabilities = F.softmax(output, dim=1)
            confidence = probabilities.max().item()
            prediction = probabilities.argmax().item()
        
        return prediction, confidence  # 0: Bad-Posture, 1: Good-Posture
    
    def visualize_pose(self, image):
        """Visualize pose landmarks on image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
        
        return image
