import torch
import torch.nn as nn
import torch.nn.functional as F
import os,sys
parent_dir = os.path.abspath(os.path.join(".."))
if not parent_dir in sys.path:
    sys.path.append(parent_dir)
from main.extra.alignment_module import AlignmentModule, ContrastiveLoss

# Test function
def test_alignment_module(path):
    """Test the alignment module with dummy data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model=AlignmentModule.load_model(path, device)
    # Create dummy data
    batch_size = 8
    visual_features = torch.randn(batch_size, 512).to(device)
    text_features = torch.randn(batch_size, 512).to(device)
    
    # Forward pass
    aligned_visual, aligned_text = model(visual_features, text_features)
    
    # Compute loss
    criterion = ContrastiveLoss()
    loss, similarity = criterion(aligned_visual, aligned_text)
    
    print(f"Input shapes - Visual: {visual_features.shape}, Text: {text_features.shape}")
    print(f"Aligned shapes - Visual: {aligned_visual.shape}, Text: {aligned_text.shape}")
    print(similarity)
    print(f"Loss: {loss.item():.4f}")


print("Testing Alignment Module...")
test_alignment_module("../saved/AlignmentModule/alignment_module.pth")