import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

def create_training_plots(metrics_file, output_dir):
    """
    Create comprehensive training visualization plots
    """
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    model_names = ['face_expression', 'emotional', 'posture', 'breathing']
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Meditation Analyzer Training Results', fontsize=16, fontweight='bold')
    
    for i, model_name in enumerate(model_names):
        if model_name not in metrics:
            continue
            
        model_metrics = metrics[model_name]
        epochs = range(1, len(model_metrics['train_losses']) + 1)
        
        # Plot accuracy
        axes[0, i].plot(epochs, model_metrics['train_accs'], 
                       label='Train Accuracy', color='#2E86C1', linewidth=2)
        axes[0, i].plot(epochs, model_metrics['val_accs'], 
                       label='Validation Accuracy', color='#E74C3C', linewidth=2)
        axes[0, i].set_title(f'{model_name.replace("_", " ").title()}\\nAccuracy vs Epochs', 
                            fontweight='bold')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Accuracy (%)')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_ylim([0, 100])
        
        # Plot loss
        axes[1, i].plot(epochs, model_metrics['train_losses'], 
                       label='Train Loss', color='#2E86C1', linewidth=2)
        axes[1, i].plot(epochs, model_metrics['val_losses'], 
                       label='Validation Loss', color='#E74C3C', linewidth=2)
        axes[1, i].set_title(f'{model_name.replace("_", " ").title()}\\nLoss vs Epochs', 
                            fontweight='bold')
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Loss')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comprehensive plot
    comprehensive_path = os.path.join(output_dir, 'comprehensive_training_plots.png')
    plt.savefig(comprehensive_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual plots for each model
    for model_name in model_names:
        if model_name not in metrics:
            continue
            
        model_metrics = metrics[model_name]
        epochs = range(1, len(model_metrics['train_losses']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{model_name.replace("_", " ").title()} Training Progress', 
                     fontsize=14, fontweight='bold')
        
        # Accuracy plot
        ax1.plot(epochs, model_metrics['train_accs'], 
                label='Train Accuracy', color='#2E86C1', linewidth=2, marker='o', markersize=3)
        ax1.plot(epochs, model_metrics['val_accs'], 
                label='Validation Accuracy', color='#E74C3C', linewidth=2, marker='s', markersize=3)
        ax1.set_title('Accuracy vs Epochs', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 100])
        
        # Loss plot
        ax2.plot(epochs, model_metrics['train_losses'], 
                label='Train Loss', color='#2E86C1', linewidth=2, marker='o', markersize=3)
        ax2.plot(epochs, model_metrics['val_losses'], 
                label='Validation Loss', color='#E74C3C', linewidth=2, marker='s', markersize=3)
        ax2.set_title('Loss vs Epochs', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save individual plot
        individual_path = os.path.join(output_dir, f'{model_name}_training_progress.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training plot for {model_name}")
    
    return comprehensive_path

def generate_model_comparison_plot(metrics_file, output_dir):
    """
    Generate comparison plot showing final performance of all models
    """
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    model_names = []
    final_train_accs = []
    final_val_accs = []
    
    for model_name in ['face_expression', 'emotional', 'posture', 'breathing']:
        if model_name in metrics:
            model_names.append(model_name.replace('_', ' ').title())
            final_train_accs.append(metrics[model_name]['train_accs'][-1])
            final_val_accs.append(metrics[model_name]['val_accs'][-1])
    
    # Create comparison plot
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width/2, final_train_accs, width, 
                   label='Training Accuracy', color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x + width/2, final_val_accs, width, 
                   label='Validation Accuracy', color='#E67E22', alpha=0.8)
    
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_title('Final Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    comparison_path = os.path.join(output_dir, 'model_performance_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_path

if __name__ == "__main__":
    # Example usage
    metrics_file = "outputs/training_metrics.json"
    output_dir = "outputs/plots"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    comprehensive_plot = create_training_plots(metrics_file, output_dir)
    comparison_plot = generate_model_comparison_plot(metrics_file, output_dir)
    
    print(f"Training visualizations saved to {output_dir}")
