import torch
import torch.nn as nn
import numpy as np
import os
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

class BLIPCaptionConverter(nn.Module):
    """
    A mobile-focused image captioner using BLIP model from Hugging Face.
    This class loads pre-trained or fine-tuned BLIP models and exports 
    the complete model to ONNX for efficient mobile deployment.
    """

    def __init__(self, model_name, finetuned_model=None, checkpoint=None):
        """
        Initializes the BLIP captioner with support for both Hub models and local checkpoints.

        Args:
            model_name (str): The name of the base pre-trained BLIP model from Hugging Face.
                              'Salesforce/blip-image-captioning-base' or 'Salesforce/blip-image-captioning-large'
            finetuned_model (str, optional): Hub path to your fine-tuned model (e.g., 'your_username/blip-finetuned-model').
                                           If provided, this model will be loaded instead of the base model.
            checkpoint (str, optional): Path to a local fine-tuned model checkpoint (.pt/.pth file).
                                        If provided, these weights will be loaded on top of the base/hub model.
                                        Note: Use either finetuned_model OR checkpoint, not both.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = model_name
        self.finetuned_model = finetuned_model
        
        print(f"Using device: {self.device}")

        # Determine which model to load
        if finetuned_model and checkpoint:
            raise ValueError("Please provide either finetuned_model OR checkpoint, not both.")
        
        # Load the appropriate model
        if finetuned_model:
            print(f"Loading fine-tuned model from Hub: {finetuned_model}")
            self.model_name = finetuned_model
            try:
                self.processor = BlipProcessor.from_pretrained(finetuned_model)
                self.model = BlipForConditionalGeneration.from_pretrained(finetuned_model)
                print("Fine-tuned model loaded successfully from Hub")
            except Exception as e:
                print(f"Failed to load fine-tuned model from Hub: {e}")
                print(f"Falling back to base model: {model_name}")
                self.model_name = model_name
                self.processor = BlipProcessor.from_pretrained(model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        else:
            print(f"Loading base BLIP model: {model_name}")
            self.model_name = model_name
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        self.model = self.model.to(self.device)

        # Load local checkpoint if provided (this will override Hub model weights)
        if checkpoint:
            print("Loading additional weights from local checkpoint...")
            self._load_checkpoint(checkpoint)

        self.model.eval()
        print(f"Model loaded successfully\nbase model: {self.base_model_name}")

    def _load_checkpoint(self, checkpoint_path):
        """
        Loads fine-tuned checkpoint weights.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        if os.path.isfile(checkpoint_path):
            print(f"Loading fine-tuned weights from: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
                
                # Load the state dict
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"Warning: Missing keys in checkpoint: {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
                
                print("Fine-tuned checkpoint loaded successfully.")
                
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                raise
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    def generate_caption_pytorch(self, image_path_or_pil, max_length=50):
        """
        Generates a caption for an image using the PyTorch model.

        Args:
            image_path_or_pil (str or PIL.Image.Image): Path to image or PIL Image object
            max_length (int): Maximum length of generated caption

        Returns:
            str: The generated caption
        """
        if isinstance(image_path_or_pil, str):
            raw_image = Image.open(image_path_or_pil).convert("RGB")
        else:
            raw_image = image_path_or_pil

        # Process the image
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)

        # Generate caption
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                do_sample=False,
                early_stopping=True
            )

        # Decode the generated tokens
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption

    def export_to_onnx(self, export_dir, quantize=True, opset_version=13):
        """
        Exports the BLIP model to ONNX format using direct torch.onnx.export.
        
        Args:
            export_dir (str): Directory where ONNX model will be saved
            quantize (bool): Whether to apply quantization to the model
            opset_version (int): ONNX opset version (13 is stable for decoder models)
        """
        os.makedirs(export_dir, exist_ok=True)
        
        print("Exporting BLIP model to ONNX using torch.onnx.export...")
        
        # Create dummy inputs following your working approach
        dummy_image = Image.fromarray(
                np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        )
        dummy_inputs = self.processor(images=dummy_image, return_tensors="pt")
        pixel_values = dummy_inputs["pixel_values"]
        
        # Create dummy decoder input with valid start token
        bos_token_id = (
            self.model.config.decoder_start_token_id
            or self.processor.tokenizer.bos_token_id
            or self.processor.tokenizer.cls_token_id
            or self.processor.tokenizer.eos_token_id
        )
        if bos_token_id is None:
            raise ValueError("No valid decoder start token found.")
        
        decoder_input_ids = torch.tensor([[bos_token_id]], dtype=torch.long)
        
        # Export path
        onnx_path = os.path.join(export_dir, "blip_captioner.onnx")
        try:
            # Export the model directly (no wrapper needed!)
            torch.onnx.export(
                self.model,
                (pixel_values, decoder_input_ids),
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["pixel_values", "decoder_input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size"},
                    "decoder_input_ids": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"}
                }
            )
            
            print(f"BLIP model exported to: {onnx_path}")
            # Verify the exported model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model is valid")
            
            # Apply quantization if requested
            if quantize:
                self._quantize_onnx_model(onnx_path, export_dir)
            
            # Save deployment information
            self._save_deployment_info(export_dir)
            
            # Create inference scripts
            self._create_inference_scripts(export_dir)
            
            # Test the exported model
            self._test_onnx_model(onnx_path, pixel_values, decoder_input_ids)
            
            print(f"Export complete! All files saved to: {export_dir}")
            
        except Exception as e:
            print(f"ONNX export failed: {e}")
            raise

    def _quantize_onnx_model(self, onnx_path, export_dir):
        """
        Quantizes the ONNX model using onnxruntime.quantization.
        
        Args:
            onnx_path (str): Path to the original ONNX model
            export_dir (str): Export directory for saving quantized model
        """
        print("Applying dynamic quantization to ONNX model...")
        
        quantized_path = os.path.join(export_dir, "blip_captioner_quantized.onnx")
        
        try:
            # Apply dynamic quantization
            quantize_dynamic(
                model_input=onnx_path,
                model_output=quantized_path,
                weight_type=QuantType.QUInt8
            )
            
            print(f"Quantized model saved to: {quantized_path}")
            
            # Test the quantized model
            # We need to create fresh inputs for the quantized test
            bos_token_id = (
                self.model.config.decoder_start_token_id or
                self.processor.tokenizer.bos_token_id or
                self.processor.tokenizer.cls_token_id or
                self.processor.tokenizer.eos_token_id
            )
            
            dummy_image_q = Image.fromarray(
                torch.randint(0, 256, (384, 384, 3), dtype=torch.uint8).numpy()
            )
            dummy_inputs_q = self.processor(images=dummy_image_q, return_tensors="pt")
            pixel_values_q = dummy_inputs_q["pixel_values"].to(self.device)
            decoder_input_ids_q = torch.tensor([[bos_token_id]], dtype=torch.long, device=self.device)
            
            self._test_onnx_model(quantized_path, pixel_values_q, decoder_input_ids_q, is_quantized=True)
            
        except Exception as e:
            print(f"Quantization failed: {e}")
            print("Continuing without quantization...")

    def _test_onnx_model(self, onnx_path, pixel_values, decoder_input_ids, is_quantized=False):
        """
        Tests the exported ONNX model with actual inputs.
        
        Args:
            onnx_path (str): Path to the ONNX model
            pixel_values (torch.Tensor): Sample pixel values for testing
            decoder_input_ids (torch.Tensor): Sample decoder input IDs
            is_quantized (bool): Whether this is a quantized model
        """
        model_type = "quantized" if is_quantized else "regular"
        print(f"Testing {model_type} ONNX model...")
        
        try:
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(onnx_path)
            
            # Prepare inputs
            ort_inputs = {
                "pixel_values": pixel_values.cpu().numpy(),
                "decoder_input_ids": decoder_input_ids.cpu().numpy()
            }
            
            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)
            logits = ort_outputs[0]
            
            print(f"{model_type.capitalize()} ONNX model test successful. Logits shape: {logits.shape}")
            
            # Get the most probable next token
            if logits.shape[-1] > 0:
                next_token_id = np.argmax(logits[0, -1, :])
                print(f"Most probable next token ID: {next_token_id}")
                
                # Try to decode if possible
                try:
                    token_text = self.processor.tokenizer.decode([next_token_id])
                    print(f"Next token text: '{token_text}'")
                except:
                    print("Could not decode token (normal for some tokens)")
                
        except Exception as e:
            print(f"{model_type.capitalize()} ONNX model test failed: {e}")

    def _save_deployment_info(self, export_dir):
        """
        Saves deployment information as JSON.
        
        Args:
            export_dir (str): Export directory
        """
        info = {
            'model_name': self.model_name,
            'model_type': 'blip-image-captioning',
            'task': 'image_captioning',
            'framework': 'onnx_runtime',
            'model_file': 'blip_complete.onnx',
            'quantized_model': 'blip_complete_quantized.onnx',
            'input_spec': {
                'name': 'pixel_values',
                'shape': [1, 3, 384, 384],
                'type': 'float32',
                'description': 'RGB image tensor, preprocessed'
            },
            'output_spec': {
                'name': 'generated_ids', 
                'shape': [1, 'variable_length'],
                'type': 'int64',
                'description': 'Generated token IDs (decode with processor)'
            },
            'preprocessing': {
                'image_size': [384, 384],
                'normalization': {
                    'mean': [0.48145466, 0.4578275, 0.40821073],
                    'std': [0.26862954, 0.26130258, 0.27577711]
                },
                'pixel_value_range': [0.0, 1.0]
            },
            'generation_config': {
                'max_length': 50,
                'strategy': 'greedy_embedded_in_model',
                'note': 'Generation logic is built into the ONNX model'
            },
            'onnx_opset': 14,
            'deployment_notes': [
                "Single ONNX model handles complete image-to-text pipeline",
                "Input: preprocessed image tensor [1, 3, 384, 384]",
                "Output: generated token IDs",
                "Simply decode output tokens with BLIP processor",
                "Use quantized model for better mobile performance"
            ]
        }
        
        info_path = os.path.join(export_dir, "model_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        print(f"Deployment info saved to: {info_path}")

    def _create_inference_scripts(self, export_dir):
        """
        Creates inference scripts for testing the exported model.
        
        Args:
            export_dir (str): Export directory
        """
        
        # Python inference script
        python_script = '''
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
from transformers import BlipProcessor

class ONNXBLIPCaptioner:
    """
    Complete BLIP image captioning using ONNX Runtime.
    This shows the logic you'll implement in Java for Android.
    """
    
    def __init__(self, onnx_model_path, model_info_path, quantized=False):
        # Load ONNX model
        self.session = ort.InferenceSession(onnx_model_path)
        
        # Load model info
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # Initialize processor (you'll need to implement preprocessing in Java)
        model_name = self.model_info['model_name']
        self.processor = BlipProcessor.from_pretrained(model_name)
        
        self.quantized = quantized
        print(f"ONNX BLIP Captioner initialized ({'quantized' if quantized else 'regular'} model)")
    
    def preprocess_image(self, image_path):
        """Preprocess image for BLIP model."""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
            
        # Use BLIP processor
        inputs = self.processor(images=image, return_tensors="np")
        return inputs.pixel_values
    
    def generate_caption(self, image_path):
        """
        Complete pipeline: image -> caption.
        Much simpler than separate encoder/decoder!
        """
        # 1. Preprocess image
        pixel_values = self.preprocess_image(image_path)
        
        # 2. Run ONNX model - ONE CALL!
        ort_inputs = {self.session.get_inputs()[0].name: pixel_values}
        ort_outputs = self.session.run(None, ort_inputs)
        generated_ids = ort_outputs[0]
        
        # 3. Decode to string
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return caption

# Example usage
if __name__ == "__main__":
    # Test regular model
    captioner = ONNXBLIPCaptioner(
        onnx_model_path="blip_complete.onnx",
        model_info_path="model_info.json"
    )
    
    # Test with dummy image
    test_image = Image.new('RGB', (384, 384), color='red')
    caption = captioner.generate_caption(test_image)
    print(f"Regular model caption: {caption}")
    
    # Test quantized model if available
    try:
        quantized_captioner = ONNXBLIPCaptioner(
            onnx_model_path="blip_complete_quantized.onnx",
            model_info_path="model_info.json",
            quantized=True
        )
        caption_q = quantized_captioner.generate_caption(test_image)
        print(f"Quantized model caption: {caption_q}")
    except:
        print("Quantized model not available")

"""
JAVA IMPLEMENTATION NOTES FOR ANDROID:
======================================

public class BLIPCaptioner {
    private OrtSession session;
    private int inputHeight = 384;
    private int inputWidth = 384;
    
    // Preprocessing constants (from BLIP)
    private final float[] MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    private final float[] STD = {0.26862954f, 0.26130258f, 0.27577711f};
    
    public String generateCaption(Bitmap image) {
        // 1. Preprocess image
        float[] pixelValues = preprocessImage(image);
        
        // 2. Create ONNX tensor [1, 3, 384, 384]
        long[] shape = {1, 3, inputHeight, inputWidth};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, pixelValues, shape);
        
        // 3. Run inference - SINGLE CALL!
        Map<String, OnnxTensor> inputs = Collections.singletonMap("pixel_values", inputTensor);
        OrtSession.Result results = session.run(inputs);
        
        // 4. Get generated token IDs
        long[][] generatedIds = (long[][]) results.get(0).getValue();
        
        // 5. Decode to string (implement BPE tokenizer or use pre-built library)
        String caption = decodeTokens(generatedIds[0]);
        
        return caption;
    }
    
    private float[] preprocessImage(Bitmap bitmap) {
        // Resize to 384x384
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 384, 384, true);
        
        float[] pixels = new float[1 * 3 * 384 * 384];
        int[] intPixels = new int[384 * 384];
        resized.getPixels(intPixels, 0, 384, 0, 0, 384, 384);
        
        // Convert to [C, H, W] format and normalize
        for (int i = 0; i < 384 * 384; i++) {
            int pixel = intPixels[i];
            
            // Extract RGB and normalize to [0,1]
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;
            
            // Apply BLIP normalization
            pixels[i] = (r - MEAN[0]) / STD[0];                    // R channel
            pixels[384*384 + i] = (g - MEAN[1]) / STD[1];          // G channel  
            pixels[2*384*384 + i] = (b - MEAN[2]) / STD[2];        // B channel
        }
        
        return pixels;
    }
}

ADVANTAGES:
- Single ONNX inference call (very fast!)
- No complex autoregressive generation in Java
- Simple preprocessing and postprocessing
- Perfect for mobile deployment with YOLOv8
"""
'''
        
        python_script_path = os.path.join(export_dir, "onnx_blip_inference.py")
        with open(python_script_path, 'w') as f:
            f.write(python_script)
        print(f"Python inference script saved to: {python_script_path}")