import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small
import clip
import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2
from ultralytics import YOLO
from PIL import Image
import json

# Configuration
IMAGE_SIZE = 640  # Input resolution for YOLOv8
ROI_SIZE = 224    # Input size for MobileNetV3 and CLIP
K_SIMILAR = 5     # Number of most similar regions for captioning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet class names (simplified - you should load the complete list)
IMAGENET_CLASSES = [
    "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead_shark",
    "electric_ray", "stingray", "cock", "hen", "ostrich", "brambling", "goldfinch",
    # ... Add all 1000 ImageNet classes here
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"
    # This is just a sample - you need the full list
]

class ForegroundDetector(nn.Module):
    """
    Modified YOLOv8n with single-class (foreground) detection
    - Freezes pretrained COCO weights
    - Treats all detections as foreground (class=1)
    """
    def __init__(self, confidence_threshold=0.25):
        super().__init__()
        # Load pretrained YOLOv8n
        self.yolo_model = YOLO('yolov8n.pt')
        self.confidence_threshold = confidence_threshold
        
        # Freeze all parameters
        for param in self.yolo_model.model.parameters():
            param.requires_grad = False
            
        self.yolo_model.model.eval()
        
    def preprocess_image(self, image):
        """
        Preprocess image for YOLOv8
        Input: PIL Image or numpy array
        Output: Preprocessed tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
        
    def forward(self, image):
        """
        Input: PIL Image or numpy array
        Output: List of detections with bbox coordinates (xyxy format)
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Run detection
        results = self.yolo_model(processed_image, conf=self.confidence_threshold)
        
        # Process detections - treat all as foreground
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf.item()
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': 1,  # All detections are foreground
                        'id': i
                    })
        return detections

class ROIProcessor(nn.Module):
    """
    Handles ROI cropping and processing:
    1. Crops regions based on YOLO detections
    2. Resizes to ROI_SIZE
    3. Normalizes for MobileNetV3 and CLIP
    """
    def __init__(self):
        super().__init__()
        self.mobilenet_transform = T.Compose([
            T.Resize((ROI_SIZE, ROI_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Separate transform for CLIP (it has its own preprocessing)
        self.clip_transform = T.Compose([
            T.Resize((ROI_SIZE, ROI_SIZE)),
            T.ToTensor(),
        ])
        
    def crop_and_resize(self, image, detections):
        """
        Input:
            image - PIL Image
            detections - list of bbox dicts
        Output:
            mobilenet_rois: List of ROI tensors for MobileNetV3
            clip_rois: List of ROI tensors for CLIP
            valid_detections: List of valid detections
        """
        mobilenet_rois = []
        clip_rois = []
        valid_detections = []
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        w, h = image.size
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure valid coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Crop ROI
            roi_image = image.crop((x1, y1, x2, y2))
            
            if roi_image.size[0] > 0 and roi_image.size[1] > 0:
                # Process for MobileNet
                mobilenet_roi = self.mobilenet_transform(roi_image)
                mobilenet_rois.append(mobilenet_roi)
                
                # Process for CLIP
                clip_roi = self.clip_transform(roi_image)
                clip_rois.append(clip_roi)
                
                valid_detections.append(det)
                
        return mobilenet_rois, clip_rois, valid_detections

class ObjectClassifier(nn.Module):
    """
    MobileNetV3-Small for ROI classification
    - Freezes backbone with ImageNet1k weights
    - Keeps original 1000-class head
    """
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, rois):
        """
        Input: List of ROI tensors [3, ROI_SIZE, ROI_SIZE]
        Output: List of class predictions
        """
        if not rois:
            return []
            
        batch = torch.stack(rois).to(DEVICE)
        
        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1)
            
        classifications = []
        for i, prob in enumerate(probs):
            top5_probs, top5_indices = torch.topk(prob, 5)
            classifications.append({
                'top_class': top5_indices[0].item(),
                'top_confidence': top5_probs[0].item(),
                'top5_classes': top5_indices.tolist(),
                'top5_confidences': top5_probs.tolist(),
                'class_name': IMAGENET_CLASSES[top5_indices[0].item()] if top5_indices[0].item() < len(IMAGENET_CLASSES) else f"class_{top5_indices[0].item()}"
            })
        return classifications

class AlignmentLayer(nn.Module):
    """Aligns visual and text features to a common space"""
    def __init__(self, visual_dim=512, text_dim=512, hidden_dim=256):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, visual_features, text_features):
        aligned_visual = self.visual_proj(visual_features)
        aligned_text = self.text_proj(text_features)
        return aligned_visual, aligned_text

class CrossAttentionLayer(nn.Module):
    """Cross-attention between visual features"""
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.multihead_attn = MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, visual_features, context_features=None):
        if context_features is None:
            context_features = visual_features
            
        attended, _ = self.multihead_attn(
            visual_features, context_features, context_features
        )
        return self.norm(attended + visual_features)  # Residual connection

class TinyBERT(nn.Module):
    """Simplified BERT-like decoder for caption generation"""
    def __init__(self, vocab_size=30522, embed_dim=256, num_layers=3, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(512, embed_dim)  # Max sequence length
        
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids, visual_context):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.pos_embedding(positions)
        hidden_states = token_embeds + pos_embeds
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, visual_context)
            
        # Output projection
        logits = self.output_projection(hidden_states)
        return logits

class LightCapCaptioner(nn.Module):
    """
    Enhanced LightCap with proper CLIP feature alignment and cross-attention
    """
    def __init__(self):
        super().__init__()
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # LightCap components
        self.alignment = AlignmentLayer(visual_dim=512, text_dim=512, hidden_dim=256)
        self.cross_attn = CrossAttentionLayer(embed_dim=256, num_heads=8)
        self.caption_generator = TinyBERT(vocab_size=30522, embed_dim=256)
        
        # Move to device
        self.to(DEVICE)
        
    def extract_visual_features(self, clip_rois):
        """Extract CLIP visual features for ROIs"""
        if not clip_rois:
            return torch.empty(0, 512, device=DEVICE)
            
        features = []
        for roi in clip_rois:
            # Convert tensor to PIL for CLIP preprocessing
            roi_pil = T.ToPILImage()(roi.cpu())
            roi_processed = self.clip_preprocess(roi_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                feat = self.clip_model.encode_image(roi_processed).float()
            features.append(feat.squeeze(0))
            
        return torch.stack(features) if features else torch.empty(0, 512, device=DEVICE)
    
    def select_salient_regions(self, features, k=K_SIMILAR):
        """
        Select top-k regions based on visual diversity and feature magnitude
        """
        if len(features) <= k:
            return features, torch.arange(len(features))
        
        # Calculate feature magnitude (importance)
        magnitude_scores = torch.norm(features, dim=1)
        
        # Calculate visual diversity (avoid redundant regions)
        similarity_matrix = F.cosine_similarity(
            features.unsqueeze(1), features.unsqueeze(0), dim=-1
        )
        diversity_scores = 1 - (similarity_matrix.sum(dim=1) - 1) / (len(features) - 1)
        
        # Combine scores
        combined_scores = magnitude_scores * 0.7 + diversity_scores * 0.3
        
        # Select top-k
        _, top_indices = torch.topk(combined_scores, k)
        return features[top_indices], top_indices

    def generate_caption(self, visual_features, classifications=None, max_length=30):
        """Generate caption from attended visual features"""
        if len(visual_features) == 0:
            return "No objects detected."
            
        # Project visual features to caption space
        aligned_visual, _ = self.alignment(
            visual_features.unsqueeze(0),
            visual_features.unsqueeze(0)  # Self-alignment
        )
        
        # Cross-attention
        attended = self.cross_attn(aligned_visual)
        
        # Simple template-based caption generation (you can replace with proper BERT decoding)
        caption = self.generate_template_caption(classifications)
        return caption
    
    def generate_template_caption(self, classifications):
        """Generate caption using templates and classifications"""
        if not classifications or len(classifications) == 0:
            return "I see some objects in the image."
        
        # Get top detected objects
        objects = []
        for cls in classifications[:3]:  # Top 3 objects
            objects.append(cls['class_name'])
        
        # Simple template-based generation
        if len(objects) == 1:
            return f"I see {objects[0]} in the image."
        elif len(objects) == 2:
            return f"I see {objects[0]} and {objects[1]} in the image."
        else:
            return f"I see {objects[0]}, {objects[1]}, and {objects[2]} in the image."

class VisionAssistModel(nn.Module):
    """
    Complete Vision Assist Model combining YOLOv8n, MobileNetV3, and LightCap
    """
    def __init__(self):
        super().__init__()
        self.detector = ForegroundDetector()
        self.roi_processor = ROIProcessor()
        self.classifier = ObjectClassifier()
        self.captioner = LightCapCaptioner()
        
        # Move to device
        self.to(DEVICE)
    
    def annotate_image(self, image, detections, classifications):
        """Annotate image with bounding boxes and labels"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        annotated = image.copy()
        
        for i, (det, cls) in enumerate(zip(detections, classifications)):
            x1, y1, x2, y2 = det['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{cls['class_name']}: {cls['top_confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1-20), (x1+label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated
    
    def forward(self, image):
        """
        Main forward pass
        Input: PIL Image or numpy array
        Output: Dictionary with caption, annotated image, and metadata
        """
        # Step 1: Object Detection
        detections = self.detector(image)
        
        if not detections:
            return {
                'caption': "No objects detected in the image.",
                'annotated_image': np.array(image) if isinstance(image, Image.Image) else image,
                'detections': [],
                'classifications': []
            }
        
        # Step 2: ROI Processing
        mobilenet_rois, clip_rois, valid_detections = self.roi_processor.crop_and_resize(
            image, detections
        )
        
        if not mobilenet_rois:
            return {
                'caption': "No valid regions detected.",
                'annotated_image': np.array(image) if isinstance(image, Image.Image) else image,
                'detections': detections,
                'classifications': []
            }
        
        # Step 3: Classification
        classifications = self.classifier(mobilenet_rois)
        
        # Step 4: Caption Generation
        visual_features = self.captioner.extract_visual_features(clip_rois)
        salient_features, _ = self.captioner.select_salient_regions(visual_features)
        caption = self.captioner.generate_caption(salient_features, classifications)
        
        # Step 5: Image Annotation
        annotated_image = self.annotate_image(image, valid_detections, classifications)
        
        return {
            'caption': caption,
            'annotated_image': annotated_image,
            'detections': valid_detections,
            'classifications': classifications,
            'num_objects': len(valid_detections)
        }

# Example usage and testing
def test_model():
    """Test the complete model pipeline"""
    model = VisionAssistModel()
    
    # Load a test image
    # test_image = Image.open("test_image.jpg")
    # results = model(test_image)
    
    print("Model initialized successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    return model

if __name__ == "__main__":
    model = test_model()