
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from IPython import embed
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from make_dataset import SpotDataset
from model import SpotModel
from functions import visualize_predictions

# Choose architecture
architecture =  'DeepLabV3plus'
encoder = "resnet34"
batch_size = 4

# Create test dataset and dataloader
test_dataset = SpotDataset(
    image_dir='../../../data/patch_256/test/images',
    mask_dir='../../../data/patch_256/test/masks',
    transform=None,
)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Test pretrained:
preprocess_input = get_preprocessing_fn(encoder, pretrained='imagenet')
model = SpotModel(architecture, encoder, in_channels=3, out_classes=1)
trainer = pl.Trainer()
model = model.to('cuda')
model.eval()

# Visualize some test images
visualize_predictions(model, test_dataloader, num_samples=batch_size, output_prefix=f'{architecture}_pretrained')

# Get performance metrics
test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
print(test_metrics)

# Test finetuned model
preprocess_input = get_preprocessing_fn(encoder, pretrained='imagenet')
best_model_path = f"../models/{architecture}/checkpoints/epoch=10-step=1122.ckpt"
model = SpotModel.load_from_checkpoint(
    best_model_path,
    arch=architecture,
    encoder_name=encoder,
    in_channels=3,
    out_classes=1
)
model = model.to('cuda')
model.eval()

trainer = pl.Trainer()

# Visualize some test images
visualize_predictions(model, test_dataloader, num_samples=batch_size, output_prefix=f'{architecture}_finetuned')

# Get performance metrics
test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
print(test_metrics)