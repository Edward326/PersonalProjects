import torch
import torch.nn as nn
import torch.nn.functional as F
default_device = 'cpu'


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for alignment training
    Based on CLIP's contrastive learning objective
    """
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature 
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, visual_features, text_features):
        """
        Compute contrastive loss between visual and text features
        
        Args:
            visual_features: [batch_size, feature_dim]
            text_features: [batch_size, feature_dim]
            
        Returns:
            loss: Scalar loss value
        """
        batch_size = visual_features.size(0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(visual_features, text_features.T) / self.temperature
        
        # Labels are diagonal (positive pairs)
        labels = torch.arange(batch_size, device=visual_features.device)
        
        # Compute loss for both directions (visual->text and text->visual)
        loss_v2t = self.cross_entropy(similarity_matrix, labels)
        loss_t2v = self.cross_entropy(similarity_matrix.T, labels)
        
        # Average the losses
        total_loss = (loss_v2t + loss_t2v) / 2
        
        return total_loss, similarity_matrix

class AlignmentModule(nn.Module):
    """
    Alignment Module for LightCap
    Based on the paper architecture but adapted for CLIP ViT-B/32 (512-dim embeddings input)
    Original paper: 2048 × 1024(text emb) and 1024 × 1024(img emb) linear blocks for ViT-ResNet50 compatibility
    My version: 512 × 512(hidden) × 512(text emb) and 512 × 512(hidden) × 512(img emb) for CLIP ViT-B/32 compatibility
    """
    def __init__(self, 
                 input_dim=512,      # CLIP ViT-B/32 embedding dimension
                 hidden_dim=512,    # Hidden dimension for alignment
                 output_dim=512,     # Output aligned dimension
                 dropout_rate=0.1,
                 device=default_device):
        super(AlignmentModule, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        
        # Visual alignment branch
        self.visual_alignment = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Text alignment branch  
        self.text_alignment = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, visual_features, text_features):
        """
        Forward pass through alignment module
        
        Args:
            visual_features: Tensor of shape [batch_size, input_dim] - CLIP visual embeddings
            text_features: Tensor of shape [batch_size, input_dim] - CLIP text embeddings
            
        Returns:
            aligned_visual: Tensor of shape [batch_size, output_dim]
            aligned_text: Tensor of shape [batch_size, output_dim]
        """
        # Apply alignment transformations
        aligned_visual = self.visual_alignment(visual_features)
        aligned_text = self.text_alignment(text_features)
        
        # L2 normalize aligned features
        aligned_visual = F.normalize(aligned_visual, p=2, dim=-1)
        aligned_text = F.normalize(aligned_text, p=2, dim=-1)
        
        return aligned_visual, aligned_text
    
    def encode_image(self, image_features):
        """
        Encode image features through the alignment module
        
        Args:
            image_features: Tensor of shape [batch_size, input_dim]
            
        Returns:
            aligned_image: Tensor of shape [batch_size, output_dim]
        """
        aligned_visual = self.visual_alignment(image_features)
        aligned_image = F.normalize(aligned_visual, p=2, dim=-1)
        return aligned_image
    
    def encode_text(self, text_features):
        """
        Encode text features through the alignment module
        
        Args:
            text_features: Tensor of shape [batch_size, input_dim]
            
        Returns:
            aligned_text: Tensor of shape [batch_size, output_dim]
        """
        aligned_text = self.text_alignment(text_features)
        aligned_text = F.normalize(aligned_text, p=2, dim=-1)
        return aligned_text

    def compute_similarity(self, aligned_visual, aligned_text):
        """
        Compute cosine similarity between aligned features
        
        Args:
            aligned_visual: Tensor of shape [batch_size, output_dim]
            aligned_text: Tensor of shape [batch_size, output_dim]
            
        Returns:
            similarity: Tensor of shape [batch_size, batch_size]
        """
        # Compute cosine similarity matrix
        similarity = torch.matmul(aligned_visual, aligned_text.T)
        return similarity
    
    def save_model(self, path):
        """Save the alignment module"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
        }, path)
        print(f"Alignment module saved to: {path}")
    
    @classmethod
    def load_model(cls, path, device=default_device):
        """Load the alignment module"""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=checkpoint['output_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"Alignment module loaded from: {path}")
        return model