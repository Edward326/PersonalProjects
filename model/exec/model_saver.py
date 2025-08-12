import os,sys
parent_dir = os.path.abspath(os.path.join(".."))
if not parent_dir in sys.path:
    sys.path.append(parent_dir)
import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from main.basemodel import HybridLightCapYOLOv8


def export_model(model, export_dir='../savedPyTMobile'):
    print("\n📦 Exportare model pentru mobil...")

    os.makedirs(export_dir, exist_ok=True)

    # TorchScript conversion
    #dummy_input = torch.randn(1, 3, 640, 640).to(model.device)
    #traced_model = torch.jit.trace(model, dummy_input)

    # Salvare TorchScript
    torchscript_path = os.path.join(export_dir, 'lightcap_yolov8_mobile.pt')
    scripted_module = torch.jit.script(model)
    optimized_scripted_module = optimize_for_mobile(scripted_module)

    # using optimized lite interpreter model makes inference about 60% faster than the non-optimized lite interpreter model, which is about 6% faster than the non-optimized full jit model
    optimized_scripted_module._save_for_lite_interpreter(torchscript_path)

    print(f"✅ Model salvat ca TorchScript: {torchscript_path}")

    # Estimare FLOPs și parametri
    #total_params = sum(p.numel() for p in model.parameters())
    #print(f"📊 Parametri totali: {total_params:,}")

    # Notă: Pentru FLOPs exacte, se poate integra fvcore sau ptflops dacă dorești