import cv2
import numpy as np
import threading
import time
from datetime import datetime
import json

# Import our analyzers
# from models.face_expression_analyzer import FaceExpressionAnalyzer
# from models.emotional_analyzer import EmotionalAnalyzer  
# from models.posture_analyzer import PostureAnalyzer
# from models.breathing_analyzer import BreathingAnalyzer
# from inference.decision_fusion import MeditationDecisionFusion

class RealTimeMeditationAnalyzer:
    """
    Real-time meditation analysis system
    Combines all four models for live video analysis
    """
    def __init__(self, model_paths=None):
        # Initialize all analyzers
        self.face_analyzer = FaceExpressionAnalyzer(
            model_paths.get('face_expression') if model_paths else None
        )
        self.emotional_analyzer = EmotionalAnalyzer(
            model_paths.get('emotional_vision') if model_paths else None,
            model_paths.get('emotional_audio') if model_paths else None
        )
        self.posture_analyzer = PostureAnalyzer(
            model_paths.get('posture') if model_paths else None
        )
        self.breathing_analyzer = BreathingAnalyzer(
            model_paths.get('breathing') if model_paths else None
        )
        self.decision_fusion = MeditationDecisionFusion(
            model_paths.get('fusion') if model_paths else None
        )
        
        # Performance monitoring
        self.fps_counter = 0
        self.start_time = time.time()
        self.frame_times = []
        
        # Results storage
        self.current_result = None
        self.results_history = []
        
    def analyze_frame(self, frame, audio_data=None):
        """
        Analyze a single frame with all models
        """
        start_time = time.time()
        
        # Get predictions from all models
        face_pred, face_conf = self.face_analyzer.predict(frame)
        emotional_pred, emotional_conf = self.emotional_analyzer.predict(
            image=frame, audio=audio_data
        )
        posture_pred, posture_conf = self.posture_analyzer.predict(frame)
        breathing_pred, breathing_conf = self.breathing_analyzer.predict(frame)
        
        # Prepare model predictions
        model_predictions = {
            'face_expression': {'prediction': face_pred, 'confidence': face_conf},
            'emotional': {'prediction': emotional_pred, 'confidence': emotional_conf},
            'posture': {'prediction': posture_pred, 'confidence': posture_conf},
            'breathing': {'prediction': breathing_pred, 'confidence': breathing_conf}
        }
        
        # Make final decision
        final_result = self.decision_fusion.make_decision(model_predictions)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.frame_times.append(processing_time)
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
        
        # Store result
        final_result['processing_time'] = processing_time
        final_result['fps'] = 1.0 / processing_time if processing_time > 0 else 0
        
        self.current_result = final_result
        self.results_history.append(final_result)
        
        # Keep only last 100 results
        if len(self.results_history) > 100:
            self.results_history.pop(0)
        
        return final_result
    
    def visualize_results(self, frame, result):
        """
        Visualize analysis results on frame
        """
        # Make a copy to avoid modifying original
        vis_frame = frame.copy()
        
        # Draw pose landmarks
        vis_frame = self.posture_analyzer.visualize_pose(vis_frame)
        
        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Main decision
        decision_text = f"Status: {result['decision']}"
        confidence_text = f"Confidence: {result['confidence']:.2f}"
        
        # Choose color based on decision
        if result['decision'] == "Meditating":
            color = (0, 255, 0)  # Green
        elif result['decision'] == "Not Meditating":
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 255)  # Yellow for uncertain
        
        cv2.putText(vis_frame, decision_text, (10, 30), font, 1, color, 2)
        cv2.putText(vis_frame, confidence_text, (10, 70), font, 0.7, color, 2)
        
        # Individual model predictions
        y_offset = 110
        for model_name, pred_info in result['individual_predictions'].items():
            pred_text = f"{model_name}: {pred_info['prediction']} ({pred_info['confidence']:.2f})"
            cv2.putText(vis_frame, pred_text, (10, y_offset), font, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Performance info
        fps_text = f"FPS: {result.get('fps', 0):.1f}"
        cv2.putText(vis_frame, fps_text, (10, vis_frame.shape[0] - 10), font, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def run_webcam_analysis(self, camera_index=0):
        """
        Run real-time analysis on webcam feed
        """
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting webcam analysis. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze frame
            result = self.analyze_frame(frame)
            
            # Visualize results
            vis_frame = self.visualize_results(frame, result)
            
            # Display
            cv2.imshow('Meditation Analyzer', vis_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            self.fps_counter += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        total_time = time.time() - self.start_time
        avg_fps = self.fps_counter / total_time if total_time > 0 else 0
        avg_processing_time = np.mean(self.frame_times) if self.frame_times else 0
        
        print(f"\\nSession Statistics:")
        print(f"Total frames processed: {self.fps_counter}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average processing time: {avg_processing_time*1000:.2f}ms")
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        if not self.frame_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.frame_times),
            'max_processing_time': np.max(self.frame_times),
            'min_processing_time': np.min(self.frame_times),
            'fps': 1.0 / np.mean(self.frame_times),
            'total_frames': self.fps_counter
        }

# Example usage
if __name__ == "__main__":
    # Model paths (update with your actual model paths)
    model_paths = {
        'face_expression': 'models/face_expression_model.pth',
        'emotional_vision': 'models/emotional_vision_model.pth',
        'emotional_audio': 'models/emotional_audio_model.pth',
        'posture': 'models/posture_model.pth',
        'breathing': 'models/breathing_model.pth',
        'fusion': 'models/decision_fusion_model.pth'
    }
    
    # Initialize analyzer
    analyzer = RealTimeMeditationAnalyzer(model_paths)
    
    # Run webcam analysis
    analyzer.run_webcam_analysis()
