from ultralytics import YOLO
# from IPython import embed

from ultralytics import settings
settings.update({"wandb": True})

"""
Code to train yolo segmentation model while tracking with wandb.
"""

# Load a pretrained model
model = YOLO("pretrained_models/yolo11x-seg.pt")

# Train the model
results = model.train(project="yolo_x_seg",
                      data="../../data/patch_256_seg/dataset.yaml", 
                        batch = 4, 
                        epochs=500, 
                        imgsz=640,
                        patience=20
                        )

metrics = model.val()

# Print performance metrics
print(metrics.results_dict)
