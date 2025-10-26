from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
from IPython import embed

"""
Code to perform object detection with a finetuned YOLO11 model
"""

# Load the finetuned YOLO11n model
model = YOLO("runs/segment/train3/weights/best.pt")

# Run inference on image example
img = '../../data/patch_256_8bit/test/images/MF_MaxIP_3ch_2_000_230623_544_84_F_XY4_x0_y1.png'
results = model.predict(img, save=True, imgsz=640, conf=0.5,show_labels = False)

# Get detection performance on test dataset
metrics = model.val(data="../../data/patch_256_8bit/dataset.yaml",split="test", plots=True)
print(metrics.results_dict)



