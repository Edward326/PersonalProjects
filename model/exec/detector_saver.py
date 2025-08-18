import os,sys
parent_dir = os.path.abspath(os.path.join(".."))
if not parent_dir in sys.path:
    sys.path.append(parent_dir)
from main.detector import YOLOv8Optimizer
from torch.utils.mobile_optimizer import optimize_for_mobile
model_path='yolov8n.pt'
save_path='../saved/detector/yolov8n_mobile.ptl'
class_names_path = '../saved/detector/coco_class_names.txt'

if __name__ == "__main__":
    opt = YOLOv8Optimizer(model_path, save_path)
    opt.export_to_ptl()
    opt.save_class_names(class_names_path)