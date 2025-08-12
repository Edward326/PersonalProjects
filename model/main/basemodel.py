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
import cv2

# Dicționar clase COCO
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


class AlignmentModule(nn.Module):
    """Modul de aliniere între feature-urile vizuale și textuale"""
    def __init__(self, visual_dim=512, text_dim=768, hidden_dim=512):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.alignment_head = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, visual_features, text_features):
        # Proiectare în spațiu comun
        visual_proj = self.visual_proj(visual_features)
        text_proj = self.text_proj(text_features)
        
        # Aliniere prin atenție
        aligned, _ = self.alignment_head(text_proj, visual_proj, visual_proj)
        aligned = self.norm(aligned + text_proj)
        
        return aligned

class HybridLightCapYOLOv8(nn.Module):
    """Model hibrid ce combină LightCap cu YOLOv8 pentru image captioning și object detection"""
    
    def __init__(self, yolo_model='yolov8n.pt', clip_model='ViT-B/32', device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Încărcare YOLOv8 pre-antrenat (frozen)
        self.yolo = YOLO(yolo_model)
        self.yolo.model.eval()
        for param in self.yolo.model.parameters():
            param.requires_grad = False
            
        # Încărcare CLIP pre-antrenat
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # TinyBERT pentru generare caption
        self.tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        self.bert_model = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        
        # Module de aliniere și fuziune
        self.alignment_module = AlignmentModule(
            visual_dim=512,  # CLIP output dim
            text_dim=312,    # TinyBERT hidden dim
            hidden_dim=512
        )
        
        # Decoder pentru generare caption
        self.caption_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=3
        )
        
        # Head pentru generare text
        self.vocab_size = self.tokenizer.vocab_size
        self.output_head = nn.Linear(512, self.vocab_size)
        
        # Embedding pentru text
        self.text_embedding = nn.Embedding(self.vocab_size, 512)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, 512))
        
        self.to(self.device)
        
    def detect_objects(self, image):
        """Detectează obiecte folosind YOLOv8"""
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
        """Extrage feature-uri CLIP pentru regiunile detectate"""
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        roi_features = []
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            roi = image.crop((x1, y1, x2, y2))
            
            # Preprocesat pentru CLIP
            roi_tensor = self.clip_preprocess(roi).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.clip_model.encode_image(roi_tensor)
                roi_features.append(features.squeeze(0))
                
        if roi_features:
            return torch.stack(roi_features)
        return torch.zeros((1, 512)).to(self.device)
    
    def generate_caption(self, visual_features, max_length=50):
        """Generează caption folosind feature-urile vizuale"""
        batch_size = 1
        
        # Token de start
        start_token = self.tokenizer.cls_token_id
        generated = torch.tensor([[start_token]]).to(self.device)
        
        # Embedding inițial
        tgt_emb = self.text_embedding(generated)
        tgt_emb = tgt_emb + self.positional_encoding[:, :1, :]
        
        # Expandare feature-uri vizuale pentru decoder
        visual_features = visual_features.unsqueeze(0) if visual_features.dim() == 2 else visual_features
        
        caption_tokens = []
        
        for _ in range(max_length):
            # Decodare
            output = self.caption_decoder(tgt_emb, visual_features)
            logits = self.output_head(output[:, -1, :])
            
            # Sampling
            next_token = torch.argmax(logits, dim=-1)
            caption_tokens.append(next_token.item())
            
            # Stop dacă întâlnim token de final
            if next_token.item() == self.tokenizer.sep_token_id:
                break
                
            # Actualizare embedding pentru următorul pas
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            tgt_emb = self.text_embedding(generated)
            tgt_emb = tgt_emb + self.positional_encoding[:, :generated.size(1), :]
            
        # Decodare caption
        caption = self.tokenizer.decode(caption_tokens, skip_special_tokens=True)
        return caption
    
    def draw_bboxes(self, image, detections):
        """Desenează bounding box-uri pe imagine"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            image = (image * 255).astype(np.uint8)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            
        output_image = image.copy()
        
        # Culori pentru diferite scale
        scale_colors = {
            80: (255, 0, 0),    # Roșu pentru scale 80x80
            40: (0, 255, 0),    # Verde pentru scale 40x40  
            20: (0, 0, 255)     # Albastru pentru scale 20x20
        }
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # Determină scala bazată pe dimensiunea bbox
            area = (x2 - x1) * (y2 - y1)
            if area < 1600:  # 40x40
                color = scale_colors[20]
            elif area < 6400:  # 80x80
                color = scale_colors[40]
            else:
                color = scale_colors[80]
                
            # Desenare bbox
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            
            # Adăugare text cu numele clasei
            label = f"{det['class_name']}: {det['score']:.2f}"
            cv2.putText(output_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
        return output_image
    
    def forward(self, image):
        """Forward pass complet"""
        # Resize imagine pentru YOLO (640x640)
        if isinstance(image, np.ndarray):
            original_image = image.copy()
            image_pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            original_image = np.array(image)
            image_pil = image
        else:
            raise ValueError("Input trebuie să fie numpy array sau PIL Image")
            
        # Detectare obiecte
        detections = self.detect_objects(original_image)
        
        # Extragere feature-uri ROI
        roi_features = self.extract_roi_features(image_pil, detections)
        
        # Generare caption
        caption = self.generate_caption(roi_features)
        
        # Desenare bounding box-uri
        output_image = self.draw_bboxes(original_image, detections)
        
        return {
            'image_with_bboxes': output_image,
            'detections': detections,
            'caption': caption,
            'roi_features': roi_features
        }
    
    def process_live(self, frame):
        """Procesare pentru modul live (frame de la cameră)"""
        return self.forward(frame)
    
    def process_static(self, image_path):
        """Procesare pentru modul static (imagine salvată)"""
        image = Image.open(image_path).convert('RGB')
        return self.forward(np.array(image))