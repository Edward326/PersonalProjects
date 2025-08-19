import torch
import torch.nn as nn
import os
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import onnx
import onnxruntime as ort

class BLIPCaptionConverter(nn.Module):
    """
    A lightweight, mobile-focused image captioner using the BLIP model from Hugging Face.
    This class is designed to load either a pre-trained model or a fine-tuned checkpoint,
    and export the vision encoder to ONNX for efficient mobile deployment.
    """

    def __init__(self, model_name, checkpoint=None):
        """
        Initializes the captioner.

        Args:
            model_name (str): The name of the pre-trained BLIP model from Hugging Face.
                              'blip-image-captioning-base' is smaller and faster.
                              'blip-image-captioning-large' is more accurate but larger.
            checkpoint (str, optional): Path to a local fine-tuned model checkpoint (.pth).
                                        If provided, these weights will be loaded. Defaults to None.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        print(f"Using device: {self.device}")
        print(f"Loading model: {model_name}")

        # Load BLIP model and processor from Hugging Face
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(self.device)

        # --- MODIFICATION: Load fine-tuned checkpoint if provided ---
        if checkpoint:
            if os.path.isfile(checkpoint):
                print(f"Loading fine-tuned weights from: {checkpoint}")
                try:
                    checkpt = torch.load(checkpoint, map_location=self.device)
                    # Handle checkpoints saved with a 'model' key
                    state_dict = checkpt.get("model", checkpt)
                    self.model.load_state_dict(state_dict, strict=False)
                    print("✅ Fine-tuned checkpoint loaded successfully.")
                except Exception as e:
                    print(f"❌ Error loading checkpoint: {e}")
            else:
                # Raise an error if the checkpoint file does not exist
                raise FileNotFoundError(f"Checkpoint path is invalid or file not found: {checkpoint}")

        self.model.eval()

        # Extract the vision model component for ONNX export
        self.vision_model = self.model.vision_model
        print(f"Model loaded")

    def generate_caption_pytorch(self, image_path_or_pil):
        """
        Generates a caption for an image using the full PyTorch model.

        Args:
            image_path_or_pil (str or PIL.Image.Image): The path to the image or a PIL Image object.

        Returns:
            str: The generated caption.
        """
        if isinstance(image_path_or_pil, str):
            raw_image = Image.open(image_path_or_pil).convert("RGB")
        else:
            raw_image = image_path_or_pil

        # Preprocess the image and send it to the device
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)

        # Generate caption
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=50, num_beams=4)

        # Decode the generated tokens to a string
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

    def export_to_onnx(self, export_dir):
        """
        Exports the vision encoder part of the model to ONNX format.
        This ONNX model can be used on mobile to extract image features,
        which can then be used with template-based captioning.

        Args:
            export_dir (str): The directory where the ONNX model and helper files will be saved.
        """
        os.makedirs(export_dir, exist_ok=True)

        # Create a sample input tensor for tracing the model
        dummy_image = torch.randn(1, 3, 224, 224, device=self.device)
        print(f"Exporting vision encoder with input shape: {dummy_image.shape}")

        vision_path = os.path.join(export_dir, "blip_vision_encoder.onnx")

        try:
            # Export the vision encoder
            torch.onnx.export(
                self.vision_model,
                dummy_image,
                vision_path,
                export_params=True,
                opset_version=14, # Using a modern opset
                do_constant_folding=True,
                input_names=['pixel_values'],
                output_names=['image_features'],
                dynamic_axes={
                    'pixel_values': {0: 'batch_size'},
                    'image_features': {0: 'batch_size'}
                }
            )
            print(f"✅ Vision encoder exported to: {vision_path}")

            # Verify the exported model
            onnx_model = onnx.load(vision_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model is valid.")

        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            return

        # Save model and preprocessing information for your mobile app
        self.save_deployment_info(export_dir)

        # Generate mobile-friendly templates for captioning
        self.generate_mobile_templates(export_dir)

        print(f"\n✅ Export complete!")
        print(f"Export directory: {export_dir}")

    def save_deployment_info(self, export_dir):
        """Saves a JSON file with model metadata for mobile deployment."""
        info = {
            'model_name': self.model_name,
            'task': 'image_captioning_vision_encoder',
            'input_size': [224, 224],
            'normalization': {
                'mean': self.processor.image_processor.image_mean,
                'std': self.processor.image_processor.image_std
            },
            'onnx_opset': 14,
            'deployment_notes': [
                "This is the vision encoder only. The output is a feature vector.",
                "Use these features with template-based captioning logic on the mobile client.",
                "Combine with YOLOv8 detections for more context-aware captions."
            ]
        }
        info_path = os.path.join(export_dir, "model_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        print(f"Saved deployment info to: {info_path}")

    def generate_mobile_templates(self, export_dir):
        """
        Generates a JSON file with caption templates and a Python script
        demonstrating how to use them with YOLOv8 object detections.
        """
        templates_path = os.path.join(export_dir, "mobile_caption_templates.json")
        templates = {
            "single_object": [
                "A photo of a {object}.",
                "There is a {object} in the image.",
                "I can see a {object}."
            ],
            "multiple_objects": [
                "A photo showing a {object1} and a {object2}.",
                "The image contains a {object1} near a {object2}."
            ],
            "general": [
                "The image contains {main_object} and {count} other things.",
                "A photo of various objects, including a {main_object}."
            ]
        }
        with open(templates_path, 'w') as f:
            json.dump(templates, f, indent=4)
        print(f"Saved mobile caption templates to: {templates_path}")

        # You can adapt this logic to Java for your app
        mobile_helper_code = '''
# This is a Python example. Adapt this logic to Java for your app.
import json
import random

class MobileCaptionGenerator:
    def __init__(self, templates_path):
        with open(templates_path, 'r') as f:
            self.templates = json.load(f)

    def generate_from_yolo(self, detections):
        """
        Generates a caption from a list of YOLOv8 detections.
        Args:
            detections (list): A list of tuples, e.g., [('person', 0.9), ('car', 0.85)].
        """
        if not detections:
            return "I can't identify specific objects in the image."

        # Sort by confidence score
        detections.sort(key=lambda x: x[1], reverse=True)
        
        object_names = [d[0] for d in detections]

        if len(object_names) == 1:
            template = random.choice(self.templates['single_object'])
            return template.format(object=object_names[0])
        elif len(object_names) >= 2:
            template = random.choice(self.templates['multiple_objects'])
            return template.format(object1=object_names[0], object2=object_names[1])
        else:
            template = random.choice(self.templates['general'])
            return template.format(main_object=object_names[0], count=len(object_names) - 1)

# Example Usage:
# yolo_detections = [('person', 0.95), ('dog', 0.88), ('bench', 0.7)]
# generator = MobileCaptionGenerator('mobile_caption_templates.json')
# caption = generator.generate_from_yolo(yolo_detections)
# print(caption) # Output: "A photo showing a person and a dog."
'''
        code_path = os.path.join(export_dir, "mobile_caption_generator_example.py")
        with open(code_path, 'w') as f:
            f.write(mobile_helper_code)
        print(f"Saved mobile helper script to: {code_path}")