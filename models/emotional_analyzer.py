import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from datetime import datetime

class EmotionalCNN(nn.Module):
    """
    Emotional Analyzer using ResNet-18 backbone for vision
    Analyzes emotional stability and relaxation from facial features
    """
    def __init__(self, num_classes=2):
        super(EmotionalCNN, self).__init__()
        self.backbone = resnet18(pretrained=True)
        
        # Modify for grayscale input
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Add temporal squeeze-excite layer
        self.temporal_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 512, 1),
            nn.Sigmoid()
        )
        
        # Modify classifier
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Apply temporal squeeze-excite
        se = self.temporal_se(x)
        x = x * se
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        
        return x

class AudioEmotionNet(nn.Module):
    """
    Audio-based emotion analysis using Bi-GRU
    Processes log-mel spectrograms for emotional stability
    """
    def __init__(self, input_size=64, hidden_size=128, num_layers=3, num_classes=2):
        super(AudioEmotionNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.bi_gru = nn.GRU(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        out, _ = self.bi_gru(x)
        # Take the last output
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class EmotionalAnalyzer:
    """Main class for emotional analysis combining vision and audio"""
    def __init__(self, vision_model_path=None, audio_model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Vision model
        self.vision_model = EmotionalCNN().to(self.device)
        self.vision_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        # Audio model
        self.audio_model = AudioEmotionNet().to(self.device)
        
        # Load pre-trained models if available
        if vision_model_path and os.path.exists(vision_model_path):
            self.vision_model.load_state_dict(torch.load(vision_model_path, map_location=self.device))
            
        if audio_model_path and os.path.exists(audio_model_path):
            self.audio_model.load_state_dict(torch.load(audio_model_path, map_location=self.device))
    
    def extract_audio_features(self, audio_data, sr=16000):
        """Extract log-mel spectrogram features from audio"""
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, sr=sr, n_mels=64, fmax=8000
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel = (log_mel - log_mel.mean()) / log_mel.std()
        
        return log_mel.T  # (time_steps, n_mels)
    
    def predict(self, image=None, audio=None):
        """Predict emotional stability from image and/or audio"""
        predictions = []
        confidences = []
        
        # Vision prediction
        if image is not None:
            self.vision_model.eval()
            
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = self.vision_transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.vision_model(image)
                probabilities = F.softmax(output, dim=1)
                confidence = probabilities.max().item()
                prediction = probabilities.argmax().item()
                
                predictions.append(prediction)
                confidences.append(confidence)
        
        # Audio prediction
        if audio is not None:
            self.audio_model.eval()
            
            # Extract features
            features = self.extract_audio_features(audio)
            features = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.audio_model(features)
                probabilities = F.softmax(output, dim=1)
                confidence = probabilities.max().item()
                prediction = probabilities.argmax().item()
                
                predictions.append(prediction)
                confidences.append(confidence)
        
        # Combine predictions if both modalities are available
        if len(predictions) == 2:
            # Weighted average based on confidence
            weights = np.array(confidences)
            weights = weights / weights.sum()
            
            final_prediction = int(np.average(predictions, weights=weights) > 0.5)
            final_confidence = np.average(confidences, weights=weights)
        else:
            final_prediction = predictions[0]
            final_confidence = confidences[0]
        
        return final_prediction, final_confidence  # 0: Not-Stable, 1: Stable
