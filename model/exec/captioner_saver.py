import os
import sys
from PIL import Image
# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from main.captioner import BLIPCaptionConverter
# Configuration
model_name = "Salesforce/blip-image-captioning-base"
finetuned_model = None  # Fine-tuned model from Hugging Face Hub
checkpoint = None  # Fine-tuned model checkpoint path
save_path = "../saved/captioner"

# --- Main Execution ---
if __name__ == "__main__":
    captioner = BLIPCaptionConverter(model_name=model_name, finetuned_model=finetuned_model, checkpoint=checkpoint)
    
    caption = captioner.generate_caption_pytorch('website.jpg')
    print(f"✅ Generated Caption: '{caption}'")
    
    captioner.export_to_onnx(save_path)