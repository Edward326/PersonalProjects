import os,sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import clip
import numpy as np
import json
import requests
import zipfile
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# Import custom modules
from main.extra.alignment_module import AlignmentModule, ContrastiveLoss
# Configuration parameters - Optimized
epochs = 80
learning_rate = 4e-5
batch_size = 2048  # Increased for better GPU utilization
data_dir = '../datasets/coco_data'
embeddings_cache_dir = '../datasets/cached_embeddings'
max_train_samples = 60000
max_val_samples = 5000
patience=5


coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PrecomputedCOCODataset(Dataset):
    """
    COCO Dataset with precomputed embeddings for fast training
    """
    def __init__(self, 
                 data_dir,
                 split,
                 max_samples,
                 clip_model,
                 clip_preprocess,
                 device,
                 cache_dir=None):
        
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples
        self.device = device
        self.coco_classes = coco_classes
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, f'{split}_embeddings_{max_samples}.pt')
        else:
            self.cache_file = None
        
        # Freeze CLIP parameters
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Load or compute embeddings
        if self.cache_file and os.path.exists(self.cache_file):
            print(f"Loading precomputed embeddings from {self.cache_file}")
            self.embeddings_data = torch.load(self.cache_file)
            print(f"Loaded {len(self.embeddings_data)} precomputed samples")
        else:
            print("Computing embeddings (this will take time, but only once)...")
            self._setup_coco()
            self._compute_all_embeddings()
            if self.cache_file:
                print(f"Saving embeddings to {self.cache_file}")
                torch.save(self.embeddings_data, self.cache_file)
        
        print(f"Dataset initialized with {len(self.embeddings_data)} samples")
    
    def _download_coco(self):
        """Download COCO dataset"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        urls = {
            'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
            'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
            'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        }
        
        # Download annotations
        ann_path = os.path.join(self.data_dir, 'annotations_trainval2017.zip')
        if not os.path.exists(ann_path):
            print("Downloading COCO annotations...")
            self._download_file(urls['annotations'], ann_path)
            self._extract_zip(ann_path, self.data_dir)
        
        # Download images
        if self.split == 'train':
            img_path = os.path.join(self.data_dir, 'train2017.zip')
            if not os.path.exists(os.path.join(self.data_dir, 'train2017')):
                print("Downloading COCO train images...")
                self._download_file(urls['train_images'], img_path)
                self._extract_zip(img_path, self.data_dir)
        else:
            img_path = os.path.join(self.data_dir, 'val2017.zip')
            if not os.path.exists(os.path.join(self.data_dir, 'val2017')):
                print("Downloading COCO val images...")
                self._download_file(urls['val_images'], img_path)
                self._extract_zip(img_path, self.data_dir)
    
    def _download_file(self, url, filename):
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as file, tqdm(
                desc=filename.split('/')[-1],
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    bar.update(len(data))
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            raise
    
    def _extract_zip(self, zip_path, extract_to):
        """Extract zip file"""
        print(f"Extracting {zip_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")
            raise
    
    def _setup_coco(self):
        """Setup COCO API"""
        ann_file = os.path.join(self.data_dir, 'annotations', f'instances_{self.split}2017.json')
        if not os.path.exists(ann_file):
            self._download_coco()
        
        try:
            self.coco = COCO(ann_file)
            self.img_dir = os.path.join(self.data_dir, f'{self.split}2017')
            
            self.category_ids = self.coco.getCatIds()
            self.categories = self.coco.loadCats(self.category_ids)
            self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.categories}
        except Exception as e:
            print(f"Error setting up COCO API: {e}")
            raise
    
    def _compute_all_embeddings(self):
        """Precompute all embeddings in batches for efficiency"""
        # Get samples
        samples = self._prepare_samples()
        
        self.embeddings_data = []
        batch_size =64  # Batch size for embedding computation
        
        print(f"Computing embeddings for {len(samples)} samples...")
        
        for i in tqdm(range(0, len(samples), batch_size), desc="Computing embeddings"):
            batch_samples = samples[i:i + batch_size]
            
            # Process batch
            images = []
            texts = []
            valid_indices = []
            
            for j, sample in enumerate(batch_samples):
                try:
                    # Load and crop image
                    image = Image.open(sample['img_path']).convert('RGB')
                    x1, y1, x2, y2 = [int(coord) for coord in sample['bbox']]
                    
                    # Ensure valid coordinates
                    w, h = image.size
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        roi = image.crop((x1, y1, x2, y2))
                        images.append(self.clip_preprocess(roi))
                        texts.append(f"a photo of a {sample['category_name']}")
                        valid_indices.append(j)
                        
                except Exception as e:
                    continue
            
            if images:
                try:
                    # Compute embeddings in batch
                    with torch.no_grad():
                        # Visual embeddings
                        image_batch = torch.stack(images).to(self.device)
                        visual_embeddings = self.clip_model.encode_image(image_batch).float()
                        
                        # Text embeddings
                        text_tokens = clip.tokenize(texts).to(self.device)
                        text_embeddings = self.clip_model.encode_text(text_tokens).float()
                        
                        # Store embeddings
                        for k, valid_idx in enumerate(valid_indices):
                            sample_idx = i + valid_idx
                            self.embeddings_data.append({
                                'visual_embedding': visual_embeddings[k].cpu(),
                                'text_embedding': text_embeddings[k].cpu(),
                                'category_name': batch_samples[valid_idx]['category_name']
                            })
                            
                except Exception as e:
                    print(f"Error computing batch embeddings: {e}")
                    continue
    
    def _prepare_samples(self):
        """Prepare samples list from COCO"""
        samples = []
        
        img_ids = self.coco.getImgIds()
        if self.max_samples:
            img_ids = img_ids[:self.max_samples]
        
        print(f"Preparing samples from {len(img_ids)} images...")
        
        for img_id in tqdm(img_ids, desc="Processing images"):
            try:
                img_info = self.coco.loadImgs(img_id)[0]
                img_path = os.path.join(self.img_dir, img_info['file_name'])
                
                if not os.path.exists(img_path):
                    continue
                
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                
                for ann in anns:
                    if ann['area'] < 100 or ann['iscrowd']:
                        continue
                    
                    bbox = ann['bbox']
                    category_id = ann['category_id']
                    
                    x1, y1, w, h = bbox
                    x2, y2 = x1 + w, y1 + h
                    
                    if category_id in self.cat_id_to_name:
                        category_name = self.cat_id_to_name[category_id]
                        
                        samples.append({
                            'img_path': img_path,
                            'bbox': [x1, y1, x2, y2],
                            'category_name': category_name,
                            'category_id': category_id,
                            'img_id': img_id
                        })
            except Exception as e:
                continue
        
        return samples
    
    def __len__(self):
        return len(self.embeddings_data)
    
    def __getitem__(self, idx):
        """Get precomputed embeddings - FAST!"""
        sample = self.embeddings_data[idx]
        
        return {
            'visual_embedding': sample['visual_embedding'].to(self.device),
            'text_embedding': sample['text_embedding'].to(self.device),
            'category_name': sample['category_name']
        }


class OptimizedAlignmentTrainer:
    """
    Optimized Trainer for Alignment Module with mixed precision
    """
    def __init__(self, model_save_path='../saved/AlignmentModule'):
        
        self.device = device
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)
        
        print(f"Training on device: {self.device}")
        
        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if torch.cuda.is_available() else None
        print(f"Mixed precision enabled: {self.scaler is not None}")
        
        # Initialize CLIP
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        print("CLIP model loaded successfully")

        # Initialize alignment module
        self.alignment_model = AlignmentModule(
            device=self.device
        ).to(self.device)
        
        # Loss function
        self.criterion = ContrastiveLoss(temperature=0.07)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def create_datasets(self, data_dir, max_train_samples, max_val_samples):
        """Create precomputed datasets"""
        print("Creating precomputed datasets...")
        
        # Training dataset
        self.train_dataset = PrecomputedCOCODataset(
            data_dir=data_dir,
            split='train',
            max_samples=max_train_samples,
            clip_model=self.clip_model,
            clip_preprocess=self.clip_preprocess,
            device=self.device,
            cache_dir=embeddings_cache_dir
        )
        
        # Validation dataset
        self.val_dataset = PrecomputedCOCODataset(
            data_dir=data_dir,
            split='val',
            max_samples=max_val_samples,
            clip_model=self.clip_model,
            clip_preprocess=self.clip_preprocess,
            device=self.device,
            cache_dir=embeddings_cache_dir
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
    
    def create_dataloaders(self, batch_size=1024):
        """Create optimized data loaders"""
        num_workers = 0
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        )
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
    
    def train_epoch(self, optimizer):
        """Optimized training loop with mixed precision"""
        self.alignment_model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            try:
                # Move data to GPU with non_blocking
                visual_emb = batch['visual_embedding'].to(self.device, non_blocking=True)
                text_emb = batch['text_embedding'].to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if self.scaler:
                    with autocast('cuda'):
                        aligned_visual, aligned_text = self.alignment_model(visual_emb, text_emb)
                        loss, similarity = self.criterion(aligned_visual, aligned_text)
                    
                    # Mixed precision backward pass
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.alignment_model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Regular precision
                    aligned_visual, aligned_text = self.alignment_model(visual_emb, text_emb)
                    loss, similarity = self.criterion(aligned_visual, aligned_text)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.alignment_model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress less frequently for speed
                if num_batches % 10 == 0:
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Avg': f'{total_loss/num_batches:.4f}'
                    })
                    
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def validate(self):
        """Optimized validation"""
        self.alignment_model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                try:
                    visual_emb = batch['visual_embedding'].to(self.device, non_blocking=True)
                    text_emb = batch['text_embedding'].to(self.device, non_blocking=True)
                    
                    if self.scaler:
                        with autocast('cuda'):
                            aligned_visual, aligned_text = self.alignment_model(visual_emb, text_emb)
                            loss, _ = self.criterion(aligned_visual, aligned_text)
                    else:
                        aligned_visual, aligned_text = self.alignment_model(visual_emb, text_emb)
                        loss, _ = self.criterion(aligned_visual, aligned_text)
                    
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        return total_loss / max(num_batches, 1)
    
    def train(self, 
              epochs=epochs,
              learning_rate=learning_rate,
              batch_size=batch_size,
              data_dir=data_dir,
              max_train_samples=max_train_samples,
              max_val_samples=max_val_samples):
        """Full optimized training pipeline"""
        
        print("\nDataset initialization...")
        self.create_datasets(data_dir, max_train_samples, max_val_samples)
        
        self.create_dataloaders(batch_size)
        
        patience_counter=0 # early stopping

        # Optimized optimizer with weight decay
        optimizer = optim.AdamW(
            self.alignment_model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs,
            eta_min=learning_rate * 0.01
        )
        
        print("\n\nTraining phase...")
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}:")
            
            # Train
            train_loss = self.train_epoch(optimizer)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(self.model_save_path, 'alignment_model_min.pth')
                self.alignment_model.save_model(best_model_path)
                print(f"New best model saved! Val Loss: {val_loss:.4f}")
            else:   
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print("\n\nTRAINING COMPLETED!")
        self.plot_training_curves()
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Models saved in: {self.model_save_path}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Optimized Alignment Module Training Curves')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.model_save_path, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()


# Main training execution
if __name__ == "__main__":
    print("OPTIMIZED COCO Alignment Module Training")
    print("=" * 50)
    
    try:
        trainer = OptimizedAlignmentTrainer()   
        trainer.train()
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()