import torch
import torch.nn as nn
import os
import json
from lavis.models import load_model_and_preprocess
from PIL import Image
import onnx
import onnxruntime as ort

class BLIPVQAWrapper(nn.Module):
    """Wrapper to make BLIP-VQA model ONNX-exportable"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Extract the vision encoder and text components
        self.vision_encoder = model.visual_encoder
        self.text_encoder = model.text_encoder
        self.text_decoder = model.text_decoder if hasattr(model, 'text_decoder') else None
        
    def forward(self, image, input_ids, attention_mask):
        # Process image through vision encoder
        image_embeds = self.vision_encoder(image)
        
        # For VQA, we need to process both image and text
        # This is a simplified version - you may need to adjust based on the exact model architecture
        if hasattr(self.model, 'forward_encoder'):
            # Use the model's forward method for encoding
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            
            # Forward through text encoder with image context
            text_output = self.text_encoder(
                input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True
            )
            
            return text_output.last_hidden_state
        else:
            # Fallback: simple concatenation approach
            return image_embeds

def export_blip_vqa_to_onnx():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load BLIP-VQA model
    print("Loading BLIP-VQA model...")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_vqa", 
        model_type="vqav2", 
        is_eval=True, 
        device=device
    )
    
    model.eval()
    print("Model loaded and set to eval mode")
    
    # Create export directory
    export_dir = "onnx_models/blip_vqa_mobile"
    os.makedirs(export_dir, exist_ok=True)
    
    # Prepare sample inputs for tracing
    print("Preparing sample inputs...")
    
    # Sample image (mobile-optimized size)
    sample_image = Image.new("RGB", (224, 224), color='white')
    processed_image = vis_processors["eval"](sample_image).unsqueeze(0).to(device)
    
    # Sample question/text
    sample_question = "What is in this image?"
    
    # For ONNX export, we need to work with the tokenizer directly
    if hasattr(model, 'tokenizer'):
        tokenizer = model.tokenizer
    else:
        # Try to access tokenizer from text processor
        tokenizer = txt_processors["eval"] if hasattr(txt_processors["eval"], 'tokenize') else None
    
    if tokenizer is None:
        print("Warning: Could not access tokenizer. Using dummy input_ids.")
        input_ids = torch.randint(0, 1000, (1, 32), dtype=torch.long).to(device)
        attention_mask = torch.ones((1, 32), dtype=torch.long).to(device)
    else:
        # Tokenize the sample question
        encoded = tokenizer(
            sample_question,
            padding='max_length',
            truncation=True,
            max_length=32,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
    
    print(f"Input shapes - Image: {processed_image.shape}, Input IDs: {input_ids.shape}")
    
    try:
        # Method 1: Export the full model (may be complex)
        print("Attempting to export full model...")
        
        # Wrap the model to make it more ONNX-friendly
        wrapped_model = BLIPVQAWrapper(model)
        
        # Export to ONNX
        onnx_path = os.path.join(export_dir, "blip_vqa_full.onnx")
        
        torch.onnx.export(
            wrapped_model,
            (processed_image, input_ids, attention_mask),
            onnx_path,
            export_params=True,
            opset_version=12,  # Use opset 12 for better mobile compatibility
            do_constant_folding=True,
            input_names=['image', 'input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Full model exported to: {onnx_path}")
        
    except Exception as e:
        print(f"Full model export failed: {e}")
        print("Trying component-wise export...")
        
        # Method 2: Export components separately (more reliable)
        try:
            # Export vision encoder separately
            vision_path = os.path.join(export_dir, "blip_vqa_vision.onnx")
            
            torch.onnx.export(
                model.visual_encoder,
                processed_image,
                vision_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['image'],
                output_names=['image_features'],
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'image_features': {0: 'batch_size'}
                }
            )
            print(f"Vision encoder exported to: {vision_path}")
            
            # Export text encoder separately (if available)
            if hasattr(model, 'text_encoder'):
                text_path = os.path.join(export_dir, "blip_vqa_text.onnx")
                
                torch.onnx.export(
                    model.text_encoder,
                    (input_ids, attention_mask),
                    text_path,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['text_features'],
                    dynamic_axes={
                        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                        'text_features': {0: 'batch_size'}
                    }
                )
                print(f"Text encoder exported to: {text_path}")
                
        except Exception as e2:
            print(f"Component export also failed: {e2}")
            print("The model architecture might be too complex for direct ONNX export.")
    
    # Save preprocessing information
    preprocess_info = {
        'image_mean': [0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
        'image_std': [0.26862954, 0.26130258, 0.27577711],
        'image_size': [224, 224],
        'text_max_length': 32,
        'model_info': {
            'name': 'blip_vqa',
            'type': 'base_vqav2'
        }
    }
    
    preprocess_path = os.path.join(export_dir, "preprocessing_info.json")
    with open(preprocess_path, 'w') as f:
        json.dump(preprocess_info, f, indent=2)
    print(f"Preprocessing info saved to: {preprocess_path}")
    
    # Test ONNX model if export was successful
    onnx_files = [f for f in os.listdir(export_dir) if f.endswith('.onnx')]
    
    for onnx_file in onnx_files:
        onnx_path = os.path.join(export_dir, onnx_file)
        try:
            print(f"\nTesting {onnx_file}...")
            
            # Load and verify ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"âœ“ {onnx_file} is valid")
            
            # Test with ONNX Runtime
            ort_session = ort.InferenceSession(onnx_path)
            
            # Get input/output info
            input_names = [inp.name for inp in ort_session.get_inputs()]
            output_names = [out.name for out in ort_session.get_outputs()]
            print(f"  Input names: {input_names}")
            print(f"  Output names: {output_names}")
            
            # Test inference (adjust inputs based on the specific model)
            if 'image' in input_names and len(input_names) == 1:
                # Vision-only model
                test_input = {input_names[0]: processed_image.cpu().numpy()}
                output = ort_session.run(None, test_input)
                print(f"  Output shape: {[o.shape for o in output]}")
                
        except Exception as e:
            print(f"âœ— Error testing {onnx_file}: {e}")
    
    print(f"\n=== Export Complete ===")
    print(f"ONNX models saved in: {export_dir}")
    print("\nFor mobile deployment:")
    print("1. Use ONNX Runtime Mobile for iOS/Android")
    print("2. Consider model quantization for smaller size:")
    print("   - Use onnxruntime quantization tools")
    print("   - Consider INT8 quantization for mobile devices")
    print("3. Preprocessing must be done on the mobile side using the saved preprocessing_info.json")

def quantize_onnx_model(onnx_path, quantized_path):
    """Optional: Quantize ONNX model for mobile deployment"""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QUInt8
        )
        print(f"Quantized model saved to: {quantized_path}")
        
    except ImportError:
        print("Install onnxruntime quantization tools for model quantization:")
        print("pip install onnxruntime[quantization]")
    except Exception as e:
        print(f"Quantization failed: {e}")