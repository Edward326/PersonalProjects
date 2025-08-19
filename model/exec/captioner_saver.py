import os,sys
from PIL import Image
parent_dir = os.path.abspath(os.path.join(".."))
if not parent_dir in sys.path:
    sys.path.append(parent_dir)
from main.captioner import BLIPCaptionConverter
model_name="Salesforce/blip-image-captioning-base"
checkpoint=None
save_path="../saved/captioner"

# --- Main Execution ---
if __name__ == "__main__":
    captioner = BLIPCaptionConverter(model_name=model_name,checkpoint=checkpoint)

    # Test caption generation with the PyTorch model
    print("\nGenerating a sample caption...")
    try:
        test_image = Image.new(mode="RGB", size=(256, 256),color='red')
        caption = captioner.generate_caption_pytorch(test_image)
        print(f"Caption:\n{caption}")
    except Exception as e:
        print(f"Could not generate caption: {e}")

    #Export the vision model to ONNX for mobile deployment
    print("\nExporting vision model to ONNX...")
    captioner.export_to_onnx(save_path)