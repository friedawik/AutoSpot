from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
from IPython import embed

"""
Code to perform object detection with a finetuned YOLO11 model
"""

# Load the finetuned YOLO11n model
model = YOLO("runs/segment/train3/weights/best.pt")
metrics = model.val(data="../../data/patch_256_8bit/dataset.yaml",split="test", plots=True)

# Get segmentation performance
print(metrics.results_dict)



