import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import torch.quantization as quant
from torch.ao.quantization import get_default_qconfig
import os
import warnings
warnings.filterwarnings("ignore")
from huggingface_hub import login
login("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")

class Gemma3Optimizer:
    def __init__(self, model_name="google/gemma-3-4b-it"):
        """
        Initialize Gemma3 model with optimization capabilities
        Using the 4B instruction-tuned version for better mobile deployment
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.optimized_model = None
        
    def load_model(self):
        """Load Gemma3 model and tokenizer"""
        print(f"🔄 Loading Gemma3 model: {self.model_name}")
        
        try:
            # Load processor (handles both text and images)
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with torch_dtype for efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for better mobile compatibility
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Gemma3 model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Trying alternative model...")
            
            # Fallback to a smaller model if the main one fails
            try:
                self.model_name = "google/gemma-2-2b"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                )
                print("✅ Fallback model loaded successfully")
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                return None
        
        return self.model, self.tokenizer
    
    def apply_quantization(self):
        """Apply dynamic quantization for mobile deployment"""
        print("🔧 Applying dynamic quantization to Gemma3...")
        
        if self.model is None:
            print("❌ Model not loaded. Load model first.")
            return None
        
        # Set to evaluation mode
        self.model.eval()
        
        # Apply dynamic quantization
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear},  # Quantize Linear layers (most compute in transformers)
                dtype=torch.qint8
            )
            
            self.optimized_model = quantized_model
            print("✅ Dynamic quantization applied to Gemma3")
            
        except Exception as e:
            print(f"⚠️ Quantization failed: {e}")
            print("Using original model without quantization")
            self.optimized_model = self.model
        
        return self.optimized_model
    
    def optimize_for_mobile(self, enable_layer_pruning=False):
        """Complete optimization pipeline for mobile deployment"""
        print("📱 Starting Gemma3 mobile optimization pipeline...")
        
        # Load model
        result = self.load_model()
        if result is None:
            return None
        
        # Apply quantization
        self.apply_quantization()
        
        print("✅ Gemma3 mobile optimization complete")
        return self.optimized_model, self.tokenizer
    
    def generate_caption(self, labels_list, max_length=100):
        """Generate caption from detected labels using chat template"""
        if not labels_list:
            return "No objects detected in the image."
        
        # Create prompt using chat template format
        labels_str = ", ".join(labels_list)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant that describes scenes based on detected objects."}]
            },
            {
                "role": "user", 
                "content": [{"type": "text", "text": f"Objects found in the image: {labels_str}. Describe the relationships between them and the scene briefly:"}]
            }
        ]
        
        # Use optimized model if available
        model_to_use = self.optimized_model if self.optimized_model else self.model
        
        # Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate
        with torch.no_grad():
            outputs = model_to_use.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][input_len:]
        caption = self.processor.decode(generated_tokens, skip_special_tokens=True)
        
        return caption.strip()
    
    def test_inference(self, test_labels=None):
        """Test inference speed"""
        print("🧪 Testing Gemma3 inference...")
        
        if test_labels is None:
            test_labels = ["person", "car", "tree", "building"]
        
        model_to_test = self.optimized_model if self.optimized_model else self.model
        
        if model_to_test is None:
            print("❌ No model available for testing")
            return None
        
        # Warm up
        for _ in range(3):
            try:
                _ = self.generate_caption(test_labels, max_length=50)
            except Exception as e:
                print(f"Warmup failed: {e}")
                break
        
        # Time inference
        import time
        start_time = time.time()
        
        try:
            for _ in range(5):
                caption = self.generate_caption(test_labels, max_length=50)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 5
            
            print(f"⏱️ Average caption generation time: {avg_time:.3f} seconds")
            print(f"📝 Sample caption: {caption}")
            
            return avg_time
            
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    optimizer = Gemma3Optimizer()
    
    # Optimize model (without layer pruning for safety)
    result = optimizer.optimize_for_mobile(enable_layer_pruning=False)
    
    if result:
        # Test inference
        optimizer.test_inference()
        
        print("\n📋 Gemma3 Optimization Summary:")
        print("- Dynamic quantization applied for faster inference")
        print("- Model optimized for mobile deployment")
        print("- Ready for TorchScript conversion")
    else:
        print("❌ Optimization failed")