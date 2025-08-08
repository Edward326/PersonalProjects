import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small
import clip
import numpy as np
from typing import List, Tuple, Dict
import cv2

# Configuration
K_SIMILAR = 5  # Number of most similar regions to select
IMAGE_SIZE = 640
ROI_SIZE = 224

class C2f(nn.Module):
    """C2f module from YOLOv8"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Bottleneck(nn.Module):
    """Bottleneck module"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, k[0] // 2)
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, k[1] // 2, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class SPPF(nn.Module):
    """SPPF module"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Conv(nn.Module):
    """Standard convolution with BatchNorm and SiLU activation"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class YOLOv8Nano(nn.Module):
    """
    Modified YOLOv8 Nano for ROI Detection Only (No Classification)
    Based on https://arxiv.org/pdf/2408.15857
    """
    def __init__(self):
        super().__init__()
        
        # Backbone - YOLOv8n architecture
        self.stem = Conv(3, 16, 3, 2)  # P1/2
        
        self.stage1 = nn.Sequential(
            Conv(16, 32, 3, 2),  # P2/4
            C2f(32, 32, 1, True)
        )
        
        self.stage2 = nn.Sequential(
            Conv(32, 64, 3, 2),  # P3/8
            C2f(64, 64, 2, True)
        )
        
        self.stage3 = nn.Sequential(
            Conv(64, 128, 3, 2),  # P4/16
            C2f(128, 128, 2, True)
        )
        
        self.stage4 = nn.Sequential(
            Conv(128, 256, 3, 2),  # P5/32
            C2f(256, 256, 1, True),
            SPPF(256, 256, 5)
        )
        
        # Neck - FPN + PAN
        self.neck_up1 = nn.Upsample(scale_factor=2)
        self.neck_up2 = nn.Upsample(scale_factor=2)
        
        self.neck_c4 = C2f(256 + 128, 128, 1, False)
        self.neck_c3 = C2f(128 + 64, 64, 1, False)
        
        self.neck_down1 = Conv(64, 64, 3, 2)
        self.neck_down2 = Conv(128, 128, 3, 2)
        
        self.neck_p4 = C2f(64 + 128, 128, 1, False)
        self.neck_p5 = C2f(128 + 256, 256, 1, False)
        
        # Detection heads - ONLY regression + objectness (NO classification)
        self.reg_head_p3 = nn.Conv2d(64, 4, 1)    # Bounding box regression
        self.reg_head_p4 = nn.Conv2d(128, 4, 1)   # [x, y, w, h]
        self.reg_head_p5 = nn.Conv2d(256, 4, 1)
        
        self.obj_head_p3 = nn.Conv2d(64, 1, 1)    # Objectness score
        self.obj_head_p4 = nn.Conv2d(128, 1, 1)   # [object/background]
        self.obj_head_p5 = nn.Conv2d(256, 1, 1)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Backbone
        x = self.stem(x)
        
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        
        # Neck - Top-down
        p5 = c5
        p4 = self.neck_c4(torch.cat([self.neck_up1(p5), c4], 1))
        p3 = self.neck_c3(torch.cat([self.neck_up2(p4), c3], 1))
        
        # Neck - Bottom-up
        p4 = self.neck_p4(torch.cat([self.neck_down1(p3), p4], 1))
        p5 = self.neck_p5(torch.cat([self.neck_down2(p4), p5], 1))
        
        # Detection heads - Only regression + objectness
        reg_p3 = self.reg_head_p3(p3)
        reg_p4 = self.reg_head_p4(p4) 
        reg_p5 = self.reg_head_p5(p5)
        
        obj_p3 = self.obj_head_p3(p3)
        obj_p4 = self.obj_head_p4(p4)
        obj_p5 = self.obj_head_p5(p5)
        
        return {
            'regression': [reg_p3, reg_p4, reg_p5],
            'objectness': [obj_p3, obj_p4, obj_p5]
        }

class AlignmentLayer(nn.Module):
    """Alignment layer for visual-textual feature alignment"""
    def __init__(self, visual_dim=512, text_dim=512, hidden_dim=256):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, visual_features, text_features):
        v_aligned = self.norm(self.visual_proj(visual_features))
        t_aligned = self.norm(self.text_proj(text_features))
        return v_aligned, t_aligned

class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for visual-textual interaction"""
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.cross_attn = MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, visual_features, text_features):
        # Cross attention: visual queries, text keys/values
        attn_out, _ = self.cross_attn(visual_features, text_features, text_features)
        visual_features = self.norm1(visual_features + attn_out)
        
        # Feed forward
        ffn_out = self.ffn(visual_features)
        visual_features = self.norm2(visual_features + ffn_out)
        
        return visual_features

class TinyBERT(nn.Module):
    """Simplified BERT-like model for caption generation"""
    def __init__(self, vocab_size=30522, embed_dim=256, num_layers=6, num_heads=8, max_seq_len=50):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids, visual_features, attention_mask=None):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Token + position embeddings
        embeddings = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Pass through transformer layers with visual features as memory
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, visual_features)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

class ROIClassifier(nn.Module):
    """MobileNetV3 for ROI classification"""
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = mobilenet_v3_small(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class LightCapModel(nn.Module):
    """
    Complete LightCap model with integrated YOLOv8 and MobileNetV3
    """
    def __init__(self, num_classes=80, vocab_size=30522):
        super().__init__()
        
        # Core models
        self.yolo = YOLOv8Nano()
        self.roi_classifier = ROIClassifier(num_classes)
        
        # Load CLIP for visual feature extraction
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
        self.clip_model.eval()
        
        # Feature alignment and interaction
        self.alignment_layer = AlignmentLayer(visual_dim=512, text_dim=512, hidden_dim=256)
        self.cross_attention = CrossAttentionLayer(embed_dim=256, num_heads=8)
        
        # Caption generation
        self.caption_generator = TinyBERT(vocab_size=vocab_size, embed_dim=256)
        
        # Transforms
        self.yolo_transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.roi_transform = T.Compose([
            T.Resize((ROI_SIZE, ROI_SIZE)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def detect_objects(self, image):
        """Detect objects using modified YOLOv8"""
        # Preprocess image for YOLO
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Forward through YOLO
        with torch.no_grad():
            outputs = self.yolo(image)
        
        # Process outputs to get bounding boxes
        bboxes = self.process_yolo_outputs(outputs, image.shape)
        return bboxes
    
    def process_yolo_outputs(self, outputs, image_shape):
        """Process YOLO outputs to extract bounding boxes"""
        regression_outputs = outputs['regression']
        objectness_outputs = outputs['objectness']
        
        all_boxes = []
        
        for i, (reg, obj) in enumerate(zip(regression_outputs, objectness_outputs)):
            # Get predictions above confidence threshold
            obj_scores = torch.sigmoid(obj).squeeze()
            valid_mask = obj_scores > 0.5
            
            if valid_mask.sum() == 0:
                continue
                
            # Extract valid boxes
            valid_reg = reg[:, :, valid_mask]
            valid_scores = obj_scores[valid_mask]
            
            # Convert to absolute coordinates
            boxes = self.reg_to_boxes(valid_reg, image_shape, stride=8 * (2 ** i))
            
            for j, (box, score) in enumerate(zip(boxes, valid_scores)):
                all_boxes.append({
                    'bbox': box,
                    'confidence': score.item(),
                    'class': 1  # Foreground class
                })
        
        # Apply NMS
        final_boxes = self.apply_nms(all_boxes)
        return final_boxes
    
    def reg_to_boxes(self, reg_output, image_shape, stride):
        """Convert regression output to bounding boxes"""
        # This is a simplified version - in practice you'd need proper anchor handling
        h, w = reg_output.shape[-2:]
        boxes = []
        
        for i in range(h):
            for j in range(w):
                # Center coordinates
                cx = (j + 0.5) * stride
                cy = (i + 0.5) * stride
                
                # Width and height
                bw = reg_output[0, 2, i, j].item() * stride
                bh = reg_output[0, 3, i, j].item() * stride
                
                # Convert to x1, y1, x2, y2
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2
                
                boxes.append([x1, y1, x2, y2])
        
        return torch.tensor(boxes)
    
    def apply_nms(self, boxes, iou_threshold=0.5):
        """Apply Non-Maximum Suppression"""
        if not boxes:
            return []
        
        # Sort by confidence
        boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
        
        final_boxes = []
        while boxes:
            best_box = boxes.pop(0)
            final_boxes.append(best_box)
            
            # Remove overlapping boxes
            boxes = [box for box in boxes if self.calculate_iou(best_box['bbox'], box['bbox']) < iou_threshold]
        
        return final_boxes
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def crop_rois(self, image, bboxes):
        """Crop regions of interest from image"""
        rois = []
        for box_info in bboxes:
            bbox = box_info['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within image bounds
            h, w = image.shape[-2:]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Crop ROI
            roi = image[:, :, y1:y2, x1:x2]
            rois.append(roi)
        
        return rois
    
    def classify_rois(self, rois):
        """Classify cropped ROIs using MobileNetV3"""
        if not rois:
            return []
        
        classifications = []
        with torch.no_grad():
            for roi in rois:
                if roi.numel() == 0:  # Skip empty ROIs
                    classifications.append({'class': 0, 'confidence': 0.0})
                    continue
                    
                # Resize and normalize ROI
                roi_processed = F.interpolate(roi, size=(ROI_SIZE, ROI_SIZE), mode='bilinear')
                roi_processed = self.roi_transform(roi_processed.squeeze(0)).unsqueeze(0)
                
                # Classify
                logits = self.roi_classifier(roi_processed)
                probs = F.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()
                
                classifications.append({
                    'class': pred_class,
                    'confidence': confidence
                })
        
        return classifications
    
    def extract_visual_features(self, rois):
        """Extract visual features using CLIP"""
        if not rois:
            return torch.empty(0, 512)
        
        visual_features = []
        with torch.no_grad():
            for roi in rois:
                if roi.numel() == 0:
                    # Handle empty ROIs
                    visual_features.append(torch.zeros(512))
                    continue
                
                # Process ROI for CLIP
                roi_pil = T.ToPILImage()(roi.squeeze(0))
                roi_processed = self.clip_preprocess(roi_pil).unsqueeze(0)
                
                # Extract features
                features = self.clip_model.encode_image(roi_processed)
                visual_features.append(features.squeeze(0))
        
        return torch.stack(visual_features) if visual_features else torch.empty(0, 512)
    
    def select_k_similar(self, features, k=K_SIMILAR):
        """Select k most similar features based on cosine similarity"""
        if len(features) <= k:
            return features, list(range(len(features)))
        
        # Compute pairwise cosine similarities
        similarities = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        
        # Sum similarities for each feature (excluding self-similarity)
        similarity_scores = similarities.sum(dim=1) - 1  # Subtract self-similarity
        
        # Select top k
        _, top_k_indices = torch.topk(similarity_scores, k)
        selected_features = features[top_k_indices]
        
        return selected_features, top_k_indices.tolist()
    
    def generate_caption(self, visual_features, max_length=50):
        """Generate caption using TinyBERT"""
        if visual_features.numel() == 0:
            return "No objects detected."
        
        # Start with [CLS] token (assuming token id 101)
        input_ids = torch.tensor([[101]], dtype=torch.long)
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Generate next token
                logits = self.caption_generator(input_ids, visual_features.unsqueeze(0))
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                # Append token
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                # Stop if [SEP] token (assuming token id 102)
                if next_token.item() == 102:
                    break
        
        # Convert to text (simplified - you'd need proper tokenizer)
        caption = f"Generated caption with {len(visual_features)} objects"
        return caption
    
    def draw_bboxes(self, image, bboxes, classifications):
        """Draw bounding boxes on image"""
        if isinstance(image, torch.Tensor):
            image_np = (image.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            image_np = image.copy()
        
        for i, (box_info, classification) in enumerate(zip(bboxes, classifications)):
            bbox = box_info['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw rectangle
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Class: {classification['class']}, Conf: {classification['confidence']:.2f}"
            cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image_np
    
    def forward(self, image):
        """
        Main forward pass
        Returns: caption, annotated_image, bboxes, classifications
        """
        # 1. Detect objects with YOLO
        bboxes = self.detect_objects(image)
        
        if not bboxes:
            return "No objects detected.", image, [], []
        
        # 2. Crop ROIs
        rois = self.crop_rois(image, bboxes)
        
        # 3. Classify ROIs with MobileNetV3
        classifications = self.classify_rois(rois)
        
        # 4. Extract visual features with CLIP
        visual_features = self.extract_visual_features(rois)
        
        # 5. Select k most similar regions
        if len(visual_features) > 0:
            selected_features, selected_indices = self.select_k_similar(visual_features, K_SIMILAR)
            
            # 6. Apply alignment and cross-attention
            # For simplicity, using dummy text features here
            dummy_text_features = torch.randn(1, len(selected_features), 512)
            v_aligned, t_aligned = self.alignment_layer(selected_features.unsqueeze(0), dummy_text_features)
            
            # 7. Cross attention
            attended_features = self.cross_attention(v_aligned, t_aligned)
            
            # 8. Generate caption
            caption = self.generate_caption(attended_features.squeeze(0))
        else:
            caption = "No valid objects for captioning."
        
        # 9. Draw bounding boxes
        annotated_image = self.draw_bboxes(image, bboxes, classifications)
        
        return caption, annotated_image, bboxes, classifications

def create_model():
    """Factory function to create the complete model"""
    model = LightCapModel(num_classes=80, vocab_size=30522)
    return model

# Example usage
if __name__ == "__main__":
    # Create model
    model = create_model()
    model.eval()
    
    # Dummy input (3, 640, 640)
    dummy_image = torch.randn(1, 3, 640, 640)
    
    # Forward pass
    caption, annotated_image, bboxes, classifications = model(dummy_image)
    
    print(f"Caption: {caption}")
    print(f"Detected {len(bboxes)} objects")
    for i, (bbox, classification) in enumerate(zip(bboxes, classifications)):
        print(f"Object {i+1}: Class {classification['class']}, Confidence {classification['confidence']:.3f}")