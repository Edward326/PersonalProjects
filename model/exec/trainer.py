import os,sys
parent_dir = os.path.abspath(os.path.join(".."))
if not parent_dir in sys.path:
    sys.path.append(parent_dir)
import platform
import torch
from main.modelmethods import download_coco_dataset, COCOCaptionDataset, Trainer
from main.basemodel import HybridLightCapYOLOv8
from model_saver import export_model


def install_dependencies():
    print("\n🔧 Verificare și instalare dependințe...")
    if platform.system() == 'Linux':
        os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    else:
        os.system("pip install torch torchvision torchaudio")
    os.system("pip install ultralytics transformers clip-by-openai pycocotools pycocoevalcap tqdm matplotlib nltk")

def main():
    install_dependencies()

    # Descărcare dataset
    data_dir = download_coco_dataset('./data', year='2017')
    train_img_dir = os.path.join(data_dir, 'train2017')
    val_img_dir = os.path.join(data_dir, 'val2017')
    ann_file = os.path.join(data_dir, 'annotations', 'captions_train2017.json')
    val_ann_file = os.path.join(data_dir, 'annotations', 'captions_val2017.json')
    train_dataset = COCOCaptionDataset(train_img_dir, ann_file)
    val_dataset = COCOCaptionDataset(val_img_dir, val_ann_file)

    # Inițializare model
    model = HybridLightCapYOLOv8(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Trainer
    trainer = Trainer(model, train_dataset, val_dataset)
    trained_model = trainer.train()

    # Salvare model
   
    export_model(trained_model)

if __name__ == '__main__':
    main()