import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO
import clip
from transformers import AutoModel, AutoTokenizer
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
from extra.alignment_module import AlignmentModule
import cv2
default_device='cpu'
yolo_model_param='../saved/YoloV8/yolov8n.pt'
alignmment_module_path='../../saved/AlignmentModule/alignment_model.pth'
clip_model_base_arh='ViT-B/32'
base_dataset='coco'

# COCO Classes dictionary (same as before)
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}


class CrossModalModulator(nn.Module):
    """Cross-modal modulator as described in LightCap paper"""
    def __init__(self, concept_embed_dim=256, feature_dim=256, hidden_dim=312):  # Updated dims for aligned features
        super().__init__()
        # Embedding layer for visual concepts (shared with TinyBERT)
        self.concept_embedding = nn.Embedding(len(COCO_CLASSES), concept_embed_dim)
        
        # Two FC layers with ReLU and Sigmoid as per paper
        self.modulator = nn.Sequential(
            nn.Linear(concept_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, visual_concepts, region_features):
        """
        Args:
            visual_concepts: List of concept IDs for each region [num_regions]
            region_features: Visual features from alignment module [num_regions, feature_dim]
        Returns:
            modulated_features: Channel-wise modulated features [num_regions, feature_dim]
        """
        if len(visual_concepts) == 0:
            return region_features
            
        # Convert to tensor if needed
        if isinstance(visual_concepts, list):
            visual_concepts = torch.tensor(visual_concepts, device=region_features.device)
            
        # Get concept embeddings
        concept_embeds = self.concept_embedding(visual_concepts)  # [num_regions, concept_embed_dim]
        
        # Generate channel weights
        channel_weights = self.modulator(concept_embeds)  # [num_regions, feature_dim]
        
        # Apply channel-wise multiplication
        modulated_features = region_features * channel_weights
        
        return modulated_features

class HybridLightCapYOLOv8(nn.Module):
    """Optimized Hybrid LightCap model with separate alignment encoders"""
    
    def __init__(self, device=default_device):
        super().__init__()
        self.device = device
        
        # Load YOLOv8 (frozen)
        self.yolo = YOLO(yolo_model_param)
        self.yolo.model.eval()
        for param in self.yolo.model.parameters():
            param.requires_grad = False
            
        # Load CLIP (frozen)
        self.clip_model, self.clip_preprocess = clip.load(clip_model_base_arh, device=self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Load trained alignment module
        self.alignment_module = AlignmentModule.load_model(alignmment_module_path, self.device)
        self.alignment_module.eval()
        for param in self.alignment_module.parameters():
            param.requires_grad = False
        
        # Cross-modal modulator (updated for aligned feature dimensions)
        self.cross_modal_modulator = CrossModalModulator(
            concept_embed_dim=self.alignment_module.output_dim,
            feature_dim=self.alignment_module.output_dim
        )
        
        # TinyBERT for caption generation
        self.tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        self.bert_model = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        # Caption generation components (updated for aligned feature dimensions)
        self.caption_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.alignment_module.output_dim, nhead=8, batch_first=True),
            num_layers=3
        )
        self.vocab_size = self.tokenizer.vocab_size
        self.output_head = nn.Linear(self.alignment_module.output_dim, self.vocab_size)
        self.text_embedding = nn.Embedding(self.vocab_size, self.alignment_module.output_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, self.alignment_module.output_dim))

        # Create visual concept vocabulary (text embeddings for all COCO classes)
        self._create_visual_concept_vocabulary()
        self.to(self.device)
        
    def _create_visual_concept_vocabulary(self):
        """
        Create visual concept vocabulary using separate text encoder
        Computed once at initialization for maximum efficiency
        """
        concept_texts = [f"a photo of a {class_name}" for class_name in COCO_CLASSES.values()]
    
        with torch.no_grad():
            # Step 1: CLIP text encoding
            text_tokens = clip.tokenize(concept_texts).to(self.device)
            clip_text_embeddings = self.clip_model.encode_text(text_tokens)  # [80, 512]
            
            # Step 2: Process through alignment module TEXT ENCODER ONLY
            aligned_text_embeddings = self.alignment_module.encode_text(clip_text_embeddings)
        
        # Store as buffer (not trainable) - these are our visual concept vocabulary
        self.register_buffer('concept_vocabulary', aligned_text_embeddings)
        print(f"✅ Created visual concept vocabulary based on {base_dataset}")
        print(f"   Shape: {aligned_text_embeddings.shape}")  # [80, aligned_dim]
        print("🚀 Visual concepts pre-computed and cached for fast inference!")
        
    def detect_objects(self, image):
        """Detect objects using YOLOv8"""
        with torch.no_grad():
            results = self.yolo(image)
            
        detections = []
        if len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, score, cls in zip(boxes, scores, classes):
                    detections.append({
                        'bbox': box,
                        'score': float(score),
                        'class_id': int(cls),
                        'class_name': COCO_CLASSES.get(int(cls), 'unknown')
                    })
                    
        return detections
    
    def extract_roi_features(self, image, detections):
        """
        Extract and align ROI features using VISUAL ENCODER ONLY
        Much faster since we don't process text concepts repeatedly
        """
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        if not detections:
            return torch.zeros((0, self.alignment_module.output_dim)).to(self.device), []
            
        roi_features = []
        class_labels = []
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # Ensure valid crop coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Crop region
            roi = image.crop((x1, y1, x2, y2))
            
            # Preprocess for CLIP
            roi_tensor = self.clip_preprocess(roi).unsqueeze(0).to(self.device)
            
            # Extract visual features using CLIP
            with torch.no_grad():
                visual_features = self.clip_model.encode_image(roi_tensor).squeeze(0)
                
            roi_features.append(visual_features)
            class_labels.append(det['class_name'])
                
        if not roi_features:
            return torch.zeros((0, self.alignment_module.output_dim)).to(self.device), []
            
        # Stack visual features
        roi_features = torch.stack(roi_features)  # [num_regions, 512]
        
        # Process through alignment module VISUAL ENCODER ONLY
        with torch.no_grad():
            aligned_visual = self.alignment_module.encode_visual(roi_features)
            # aligned_visual: [num_regions, aligned_dim] - same space as concept_vocabulary
            
        return aligned_visual, class_labels
    
    def retrieve_visual_concepts(self, aligned_visual_features):
        """
        Retrieve most similar visual concepts using CLIP-style similarity computation
        Now with probability scores using softmax
        """
        if aligned_visual_features.size(0) == 0:
            return [], []
            
        # Compute similarity matrix (both features are L2 normalized from alignment module)
        similarity_matrix = torch.matmul(aligned_visual_features, self.concept_vocabulary.T)  
        # [num_regions, 80] - comparing aligned visual with aligned text concepts
        
        # Convert to probabilities using softmax (CLIP-style)
        probabilities = F.softmax(similarity_matrix, dim=-1)  # [num_regions, 80]
        
        # Get most similar concept for each region
        most_similar_indices = torch.argmax(similarity_matrix, dim=1)  # [num_regions]
        
        # Get confidence scores for selected concepts
        confidence_scores = []
        for i, concept_idx in enumerate(most_similar_indices):
            confidence = probabilities[i, concept_idx].item()
            confidence_scores.append(confidence)
        
        return most_similar_indices.tolist(), confidence_scores
    
    def apply_cross_modal_modulation(self, aligned_visual_features, visual_concepts):
        """Apply cross-modal modulation as described in the paper"""
        if len(visual_concepts) == 0:
            return aligned_visual_features
            
        modulated_features = self.cross_modal_modulator(visual_concepts, aligned_visual_features)
        return modulated_features
    
    def prepare_multimodal_input(self, modulated_features, visual_concepts):
        """
        Prepare multimodal input using pre-computed concept embeddings
        No text processing needed - just indexing into cached vocabulary!
        """
        if len(visual_concepts) == 0:
            return torch.zeros((1, self.alignment_module.output_dim)).to(self.device)
            
        # Use modulated features directly
        concatenated_features = modulated_features  # [num_regions, aligned_dim]
        
        # Get embeddings from pre-computed vocabulary (super fast!)
        if visual_concepts:
            concept_embeddings = self.concept_vocabulary[visual_concepts]  # [num_concepts, aligned_dim]
            
            # Concatenate visual features and concept embeddings
            multimodal_input = torch.cat([concatenated_features, concept_embeddings], dim=0)
        else:
            multimodal_input = concatenated_features
            
        return multimodal_input.unsqueeze(0)  # [1, sequence_length, aligned_dim]
    
    def generate_caption(self, multimodal_input, max_length=50):
        """Generate caption using the prepared multimodal input"""
        if multimodal_input.size(1) == 0:
            return "No objects detected"
            
        batch_size = 1
        
        # Start token
        start_token = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.pad_token_id
        generated = torch.tensor([[start_token]]).to(self.device)
        
        # Initial embedding
        tgt_emb = self.text_embedding(generated)
        tgt_emb = tgt_emb + self.positional_encoding[:, :1, :]
        
        caption_tokens = []
        
        for _ in range(max_length):
            # Decode
            output = self.caption_decoder(tgt_emb, multimodal_input)
            logits = self.output_head(output[:, -1, :])
            
            # Sampling
            next_token = torch.argmax(logits, dim=-1)
            caption_tokens.append(next_token.item())
            
            # Stop if we encounter end token
            if next_token.item() == self.tokenizer.sep_token_id:
                break
                
            # Update embedding for next step
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            tgt_emb = self.text_embedding(generated)
            tgt_emb = tgt_emb + self.positional_encoding[:, :generated.size(1), :]
            
        # Decode caption
        caption = self.tokenizer.decode(caption_tokens, skip_special_tokens=True)
        return caption
    
    def forward(self, image):
        """Complete forward pass implementing optimized LightCap pipeline"""
        # Convert image to proper format
        if isinstance(image, np.ndarray):
            original_image = image.copy()
            image_pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            original_image = np.array(image)
            image_pil = image
        else:
            raise ValueError("Input must be numpy array or PIL Image")
            
        # Step 1: Detect objects using YOLOv8
        detections = self.detect_objects(original_image)
        
        if not detections:
            return {
                'image_with_bboxes': original_image,
                'detections': [],
                'caption': "No objects detected",
                'roi_features': torch.zeros((0, self.alignment_module.output_dim)).to(self.device)
            }
        
        # Step 2: Extract ROI features and process through alignment module VISUAL ENCODER ONLY
        aligned_visual_features, detected_classes = self.extract_roi_features(image_pil, detections)
        
        if aligned_visual_features.size(0) == 0:
            return {
                'image_with_bboxes': original_image,
                'detections': detections,
                'caption': "No valid regions detected",
                'roi_features': torch.zeros((0, self.alignment_module.output_dim)).to(self.device)
            }
        
        # Step 3: Retrieve visual concepts using similarity matrix with confidence scores
        visual_concepts, concept_confidences = self.retrieve_visual_concepts(aligned_visual_features)
        
        # Step 4: Apply cross-modal modulation
        modulated_features = self.apply_cross_modal_modulation(aligned_visual_features, visual_concepts)
        
        # Step 5: Prepare multimodal input using pre-computed concept embeddings
        multimodal_input = self.prepare_multimodal_input(modulated_features, visual_concepts)
        
        # Step 6: Generate caption
        caption = self.generate_caption(multimodal_input)
        
        # Draw bounding boxes
        output_image = self.draw_bboxes(original_image, detections)
        
        return {
            'image_with_bboxes': output_image,
            'detections': detections,
            'caption': caption,
            'roi_features': modulated_features,
            'visual_concepts': visual_concepts,
            'concept_confidences': concept_confidences,
            'aligned_features': aligned_visual_features
        }
    
    def draw_bboxes(self, image, detections):
        """Draw bounding boxes on image"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            image = (image * 255).astype(np.uint8)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            
        output_image = image.copy()
        
        # Colors for different scales
        scale_colors = {
            80: (255, 0, 0),    # Red for scale 80x80
            40: (0, 255, 0),    # Green for scale 40x40  
            20: (0, 0, 255)     # Blue for scale 20x20
        }
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # Determine scale based on bbox size
            area = (x2 - x1) * (y2 - y1)
            if area < 1600:  # 40x40
                color = scale_colors[20]
            elif area < 6400:  # 80x80
                color = scale_colors[40]
            else:
                color = scale_colors[80]
                
            # Draw bbox
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            
            # Add class name
            label = f"{det['class_name']}: {det['score']:.2f}"
            cv2.putText(output_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
        return output_image
    
    def process_live(self, frame):
        """Process live frame from camera"""
        return self.forward(frame)
    
    def process_static(self, image_path):
        """Process static image from file"""
        image = Image.open(image_path).convert('RGB')
        return self.forward(np.array(image))
    
    def profile_performance(self):
        """Display performance optimizations"""
        print("🚀 Performance Optimizations Applied:")
        print("✅ Visual concept vocabulary pre-computed at initialization")
        print("✅ Only visual encoder used during inference (encode_visual)")
        print("✅ Text concepts retrieved via fast tensor indexing")
        print("✅ CLIP-style probability computation with softmax")
        print("✅ Significant speedup for batch processing")
        print(f"📊 Concept vocabulary size: {self.concept_vocabulary.shape}")
        print(f"📊 Aligned feature dimension: {self.alignment_module.output_dim}")