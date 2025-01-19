import matplotlib.pyplot as plt
from IPython import embed
import tifffile 
import torch

# read tif file and convert to np array
def tiff_to_array(tiff_path):
    with tifffile.TiffFile(tiff_path) as tiff:
        image = tiff.asarray() 
    return image

def visualize_sample(train_dataset, sample_index=0, output_filename='test_1.png'):
    """
    Visualize and save a sample from a dataset.

    :param train_dataset: The dataset object from which to sample.
    :param sample_index: The index of the sample to visualize.
    :param output_filename: The filename to save the visualization.
    """
    
    # Extract the sample
    sample = train_dataset[sample_index]
    
    # Visualize the image
    plt.subplot(1, 2, 1)
    vmax = sample['image'].max()
    plt.imshow(sample['image'][1, :, :], vmax=vmax)
    
    # Visualize the mask
    plt.subplot(1, 2, 2)
    plt.imshow(sample['mask'].squeeze())
    
    # Save the figure
    plt.savefig(output_filename)
    plt.show()


def visualize_predictions(model, test_dataloader, num_samples=5, image_channel=1, output_prefix='test'):
    """
    Visualizes predictions of a model from a test dataloader.
    
    :param model: PyTorch model used for prediction.
    :param test_dataloader: A dataloader providing batches of data (images and masks).
    :param num_samples: The number of samples to visualize.
    :param image_channel: The specific channel of the image to visualize.
    :param output_prefix: Prefix for the output filenames.
    """
    # Get a batch of data
    batch = next(iter(test_dataloader))

    # Move all tensors in the dictionary to CUDA
    batch = {key: value.to('cuda') for key, value in batch.items()}
    # Perform inference
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    
    # Apply sigmoid to get predicted masks
    pr_masks = logits.sigmoid()
    pr_masks = pr_masks > 0.5
 
    # Iterate over the batch to visualize
    for idx, (image, gt_mask, pr_mask) in enumerate(zip(batch["image"], batch["mask"], pr_masks)):
        if idx < num_samples:
            image = image.cpu()
            gt_mask = gt_mask.cpu()
            pr_mask = pr_mask.cpu()
            # Create the figure for the current sample
            plt.figure(figsize=(10, 5))
            
            # Plot the image
            plt.subplot(1, 3, 1)
            plot_image = image.numpy()[image_channel, :, :]  # Plot only the specified channel
            plt.imshow(plot_image)
            plt.title("Original Atg8a channel")
            plt.axis("off")
            
            # Plot the ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask.numpy().squeeze())
            plt.title("Ground truth")
            plt.axis("off")
            
            # Plot the predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(pr_mask.numpy().squeeze())
            plt.title("Prediction")
            plt.axis("off")
            
            # Save the figure
            plt.savefig(f'../predictions/{output_prefix}_{idx}.png')
            plt.close()  # Close the plot to free up memory
        else:
            break

