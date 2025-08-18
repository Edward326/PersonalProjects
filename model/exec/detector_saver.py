import os,sys
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
implicit_path='../saved/detector/yolov8n_mobile.ptl'
model_to_export='yolov8n.pt'

os.system(f'yolo export model={model_to_export} format=torchscript optimize=True')
ts_model = torch.jit.load("yolov8n.torchscript")
ts_model.eval()
optimized = optimize_for_mobile(ts_model)
optimized._save_for_lite_interpreter(implicit_path)