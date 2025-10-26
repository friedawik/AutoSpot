import cv2
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# Unique image paths dictionary
path_dict = {
    'Prompts with prominence>1200': '../results/diff_prompts/prompts_1200.png',
    'Prompts with prominence>600': '../results/diff_prompts/prompts_600.png',
    'Prompts with prominence>180': '../results/diff_prompts/prompts_0.png',
}


# Initialize a 3x3 grid of subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

# Flatten the axes array for easy iteration
axes_flat = axes.flatten()

x_start=0
y_start = 0
y_end = y_start + 255
x_end = x_start + 255
# Loop over the image paths and plot each in a subplot
for ax, (key, path) in zip(axes_flat, path_dict.items()):
    try:
        # Load and display the image
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        ax.imshow(img)
        ax.set_title(key)
    except FileNotFoundError:
        ax.set_title(f"Missing: {key}")
    
    ax.axis('off')  # Hide axes

# Adjust layout and display the plot
plt.tight_layout(pad=0.5)  # Reduces padding between plots
plt.savefig('sam_prompts.png')