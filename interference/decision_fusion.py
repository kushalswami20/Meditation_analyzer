import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
import json
from datetime import datetime

class MeditationDecisionFusion:
    """
    Main decision fusion class that combines predictions from all four models
    """
    def __init__(self, fusion_model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model weights (learned from calibration data)
        self.model_weights = {
            'face_expression': 0.25,
            'emotional': 0.25,
            'posture': 0.25,
            'breathing': 0.25
        }
        
        # Confidence thresholds
        self.confidence_threshold = 0.7
        self.uncertainty_threshold = 0.1
        
        # History for temporal smoothing
        self.prediction_history = []
        self.history_length = 5
    
    def weighted_average_fusion(self, predictions, confidences):
        """
        Combine predictions using weighted average
        predictions: dict with keys ['face_expression', 'emotional', 'posture', 'breathing']
        confidences: dict with same keys
        """
        weighted_sum = 0
        total_weight = 0
        
        for model_name, prediction in predictions.items():
            if model_name in self.model_weights:
                weight = self.model_weights[model_name] * confidences[model_name]
                weighted_sum += weight * prediction
                total_weight += weight
        
        if total_weight > 0:
            final_prediction = weighted_sum / total_weight
        else:
            final_prediction = 0.5  # Default uncertain prediction
        
        return final_prediction
    
    def temporal_smoothing(self, prediction, confidence):
        """
        Apply temporal smoothing to reduce jitter
        """
        # Add current prediction to history
        self.prediction_history.append((prediction, confidence))
        
        # Maintain history length
        if len(self.prediction_history) > self.history_length:
            self.prediction_history.pop(0)
        
        # Calculate exponential moving average
        weights = np.exp(np.linspace(-2, 0, len(self.prediction_history)))
        weights = weights / np.sum(weights)
        
        smoothed_prediction = 0
        smoothed_confidence = 0
        
        for i, (pred, conf) in enumerate(self.prediction_history):
            smoothed_prediction += weights[i] * pred
            smoothed_confidence += weights[i] * conf
        
        return smoothed_prediction, smoothed_confidence
    
    def make_decision(self, model_predictions, use_temporal_smoothing=True):
        """
        Make final meditation decision based on all model predictions
        
        model_predictions: dict with structure:
        {
            'face_expression': {'prediction': 0/1, 'confidence': 0.0-1.0},
            'emotional': {'prediction': 0/1, 'confidence': 0.0-1.0},
            'posture': {'prediction': 0/1, 'confidence': 0.0-1.0},
            'breathing': {'prediction': 0/1, 'confidence': 0.0-1.0}
        }
        """
        
        # Extract predictions and confidences
        predictions = {}
        confidences = {}
        
        for model_name, result in model_predictions.items():
            predictions[model_name] = result['prediction']
            confidences[model_name] = result['confidence']
        
        # Weighted average fusion
        final_pred_continuous = self.weighted_average_fusion(predictions, confidences)
        final_confidence = np.mean(list(confidences.values()))
        
        # Apply temporal smoothing
        if use_temporal_smoothing:
            final_pred_continuous, final_confidence = self.temporal_smoothing(
                final_pred_continuous, final_confidence
            )
        
        # Make binary decision
        if final_confidence < self.confidence_threshold:
            decision = "Uncertain"
            binary_prediction = 0
        else:
            binary_prediction = int(final_pred_continuous > 0.5)
            decision = "Meditating" if binary_prediction == 1 else "Not Meditating"
        
        # Prepare detailed result
        result = {
            'decision': decision,
            'binary_prediction': binary_prediction,
            'confidence': final_confidence,
            'continuous_prediction': final_pred_continuous,
            'individual_predictions': model_predictions,
            'model_weights': self.model_weights,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
