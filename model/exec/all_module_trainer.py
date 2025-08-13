import os

# DEPENDENCIES - Fix: Check torch availability after import, not before
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

print("Installing core dependencies...")
if torch_available and torch.cuda.is_available():
    os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
else:
    os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")

# OpenAI CLIP dependencies
os.system("pip install ftfy regex tqdm")
os.system("pip install git+https://github.com/openai/CLIP.git")

# Image processing
os.system("pip install Pillow")

# Data handling and utilities
os.system("pip install numpy")
os.system("pip install requests")
os.system("pip install matplotlib")

# COCO API
os.system("pip install pycocotools")

# Optional but recommended for better performance
os.system("pip install opencv-python")
os.system("pip install scikit-learn")

print("All dependencies installed!")

import time
time.sleep(2)

# Now import everything after installation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
import sys

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import custom modules
from main.extra.alignment_module import AlignmentModule, ContrastiveLoss

# Configuration parameters - Fix: Remove trailing commas
epochs = 60
learning_rate = 1e-5
batch_size = 32
save_every = 10
data_dir = '../datasets/coco_data'
max_train_samples = 10000
max_val_samples = 2000

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


class COCOAlignmentDataset(Dataset):
    """
    COCO Dataset for Alignment Module Training
    Downloads COCO, crops regions, and prepares CLIP embeddings
    """
    def __init__(self, 
                 data_dir,
                 split,
                 max_samples,
                 clip_model,
                 clip_preprocess):  # Fix: Add missing clip_preprocess parameter
        
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples
        self.device = device
        self.coco_classes = coco_classes
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess  # Fix: Store clip_preprocess
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Download and setup COCO
        self._setup_coco()
        
        # Prepare samples
        self.samples = self._prepare_samples()
        
        print(f"Dataset initialized with {len(self.samples)} samples")
    
    def _download_coco(self):
        """Download COCO dataset"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        # URLs for COCO 2017
        urls = {
            'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
            'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
            'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        }
        
        # Download annotations (always needed)
        ann_path = os.path.join(self.data_dir, 'annotations_trainval2017.zip')
        if not os.path.exists(ann_path):
            print("Downloading COCO annotations...")
            self._download_file(urls['annotations'], ann_path)
            self._extract_zip(ann_path, self.data_dir)
        
        # Download images based on split
        if self.split == 'train':
            img_path = os.path.join(self.data_dir, 'train2017.zip')
            if not os.path.exists(os.path.join(self.data_dir, 'train2017')):
                print("Downloading COCO train images (this may take a while)...")
                self._download_file(urls['train_images'], img_path)
                self._extract_zip(img_path, self.data_dir)
        else:  # validation
            img_path = os.path.join(self.data_dir, 'val2017.zip')
            if not os.path.exists(os.path.join(self.data_dir, 'val2017')):
                print("Downloading COCO val images...")
                self._download_file(urls['val_images'], img_path)
                self._extract_zip(img_path, self.data_dir)
    
    def _download_file(self, url, filename):
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Fix: Add error checking
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
        # Download COCO if not exists
        ann_file = os.path.join(self.data_dir, 'annotations', f'instances_{self.split}2017.json')
        if not os.path.exists(ann_file):
            self._download_coco()
        
        # Initialize COCO API
        try:
            self.coco = COCO(ann_file)
            self.img_dir = os.path.join(self.data_dir, f'{self.split}2017')
            
            # Get category info
            self.category_ids = self.coco.getCatIds()
            self.categories = self.coco.loadCats(self.category_ids)
            self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.categories}
        except Exception as e:
            print(f"Error setting up COCO API: {e}")
            raise
    
    def _prepare_samples(self):
        """Prepare training samples from COCO"""
        samples = []
        
        # Get all image IDs
        img_ids = self.coco.getImgIds()
        
        # Limit samples for faster training/testing
        if self.max_samples:
            img_ids = img_ids[:self.max_samples]
        
        print(f"Preparing samples from {len(img_ids)} images...")
        
        for img_id in tqdm(img_ids, desc="Processing images"):
            try:
                # Get image info
                img_info = self.coco.loadImgs(img_id)[0]
                img_path = os.path.join(self.img_dir, img_info['file_name'])
                
                # Skip if image doesn't exist
                if not os.path.exists(img_path):
                    continue
                
                # Get annotations for this image
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                
                # Process each annotation
                for ann in anns:
                    # Skip if annotation has issues
                    if ann['area'] < 100 or ann['iscrowd']:
                        continue
                    
                    bbox = ann['bbox']  # [x, y, width, height]
                    category_id = ann['category_id']
                    
                    # Convert to [x1, y1, x2, y2]
                    x1, y1, w, h = bbox
                    x2, y2 = x1 + w, y1 + h
                    
                    # Get category name
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
                print(f"Error processing image {img_id}: {e}")
                continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a training sample"""
        sample = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(sample['img_path']).convert('RGB')
            
            # Crop region
            x1, y1, x2, y2 = [int(coord) for coord in sample['bbox']]
            
            # Ensure valid coordinates
            w, h = image.size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return self._get_dummy_sample()
            
            # Crop region
            roi = image.crop((x1, y1, x2, y2))
            
            # Preprocess for CLIP
            roi_processed = self.clip_preprocess(roi).unsqueeze(0).to(self.device)
            
            # Get visual embedding
            with torch.no_grad():
                visual_embedding = self.clip_model.encode_image(roi_processed).float().squeeze(0)
            
            # Get text embedding
            text = f"a photo of a {sample['category_name']}"
            text_tokens = clip.tokenize([text]).to(self.device)
            
            with torch.no_grad():
                text_embedding = self.clip_model.encode_text(text_tokens).float().squeeze(0)
            
            return {
                'visual_embedding': visual_embedding,
                'text_embedding': text_embedding,
                'category_name': sample['category_name']
            }
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return self._get_dummy_sample()
    
    def _get_dummy_sample(self):
        """Return dummy sample for error cases"""
        return {
            'visual_embedding': torch.zeros(512).to(self.device),
            'text_embedding': torch.zeros(512).to(self.device),
            'category_name': 'unknown'
        }


class AlignmentTrainer:
    """
    Trainer for Alignment Module
    """
    def __init__(self, model_save_path='../saved/AlignmentModule'):
        
        self.device = device
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)
        
        print(f"Training on device: {self.device}")
        
        # Initialize CLIP
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        print("CLIP model loaded successfully")

        # Initialize alignment module
        self.alignment_model = AlignmentModule(
            input_dim=512,
            hidden_dim=256,
            output_dim=256
        ).to(self.device)
        
        # Loss function
        self.criterion = ContrastiveLoss(temperature=0.07)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def create_datasets(self, data_dir, max_train_samples, max_val_samples):
        """Create train and validation datasets"""
        print("Creating datasets...")
        
        # Training dataset - Fix: Pass clip_preprocess
        self.train_dataset = COCOAlignmentDataset(
            data_dir=data_dir,
            split='train',
            max_samples=max_train_samples,
            clip_model=self.clip_model,
            clip_preprocess=self.clip_preprocess
        )
        
        # Validation dataset - Fix: Pass clip_preprocess  
        self.val_dataset = COCOAlignmentDataset(
            data_dir=data_dir,
            split='val',
            max_samples=max_val_samples,
            clip_model=self.clip_model,
            clip_preprocess=self.clip_preprocess
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
    
    def create_dataloaders(self, batch_size=32, num_workers=4):
        """Create data loaders"""
        # Fix: Reduce num_workers for stability
        num_workers = min(num_workers, 2) if os.name == 'nt' else num_workers  # Windows compatibility
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
    
    def train_epoch(self, optimizer):
        """Train for one epoch"""
        self.alignment_model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            try:
                # Get batch data
                visual_emb = batch['visual_embedding'].to(self.device)
                text_emb = batch['text_embedding'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                aligned_visual, aligned_text = self.alignment_model(visual_emb, text_emb)
                
                # Compute loss
                loss, similarity = self.criterion(aligned_visual, aligned_text)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss/num_batches:.4f}'
                })
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def validate(self):
        """Validate the model"""
        self.alignment_model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                try:
                    # Get batch data
                    visual_emb = batch['visual_embedding'].to(self.device)
                    text_emb = batch['text_embedding'].to(self.device)
                    
                    # Forward pass
                    aligned_visual, aligned_text = self.alignment_model(visual_emb, text_emb)
                    
                    # Compute loss
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
              save_every=save_every,
              data_dir=data_dir,
              max_train_samples=max_train_samples,
              max_val_samples=max_val_samples):
        """Full training pipeline"""
        
        print("\nDataset initialization...")
        # Create datasets
        self.create_datasets(data_dir, max_train_samples, max_val_samples)
        
        # Create data loaders
        self.create_dataloaders(batch_size)
        
        # Setup optimizer
        optimizer = optim.Adam(self.alignment_model.parameters(), lr=learning_rate)
        
        print("\n\nTraining phase...")
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}:")
            
            # Train
            train_loss = self.train_epoch(optimizer)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(self.model_save_path, f'alignment_model_best_e{epoch+1}.pth')
                self.alignment_model.save_model(best_model_path)
                print(f"New best model saved! Val Loss: {val_loss:.4f}")
            
            # Save periodic checkpoints
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(self.model_save_path, f'alignment_model_e{epoch+1}.pth')
                self.alignment_model.save_model(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
        
        # Plot training curves
        print("\n\nTRAINING COMPLETED!")
        self.plot_training_curves()
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Models saved in: {self.model_save_path}")
    
    def plot_training_curves(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Alignment Module Training Curves')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.model_save_path, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()


# Main training execution
if __name__ == "__main__":
    print("COCO Alignment Module Training")
    print("=" * 50)
    
    try:
        trainer = AlignmentTrainer()   
        trainer.train()
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()