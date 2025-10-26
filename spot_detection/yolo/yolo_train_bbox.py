from ultralytics import YOLO
from IPython import embed

"""
Code to train the YOLO11 model on custom data.
"""

# Load a pretrained model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(data="../../data/patch_256_bbox/dataset.yaml", 
                        batch = 4, 
                        epochs=5, 
                        lr0=0.05, 
                        imgsz=256,
                        mask_ratio=1,
                        cls=0.1
                        )

metrics = model.val()

# Print performance metrics
print(metrics.results_dict)
