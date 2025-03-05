import matplotlib.pyplot as plt
import os
from PIL import Image

def display_image_pair(base_prompt, image_dir="generated_images"):
    """
    Display a pair of images (base and Hopper style) side by side.
    
    Args:
        base_prompt (str): The base prompt without style specification (e.g., "a lighthouse")
        image_dir (str): Directory containing the generated images
    """
    # Create the filenames
    base_filename = base_prompt.replace(" ", "_").lower() + ".png"
    hopper_filename = f"{base_prompt} in Edward Hopper style".replace(" ", "_").lower() + ".png"
    
    # Full paths
    base_path = os.path.join(image_dir, base_filename)
    hopper_path = os.path.join(image_dir, hopper_filename)
    
    # Check if files exist
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base image not found: {base_path}")
    if not os.path.exists(hopper_path):
        raise FileNotFoundError(f"Hopper style image not found: {hopper_path}")
    
    # Load images
    base_image = Image.open(base_path)
    hopper_image = Image.open(hopper_path)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Display images
    ax1.imshow(base_image)
    ax1.set_title("Base Image\n" + base_prompt, fontsize=12)
    ax1.axis('off')
    
    ax2.imshow(hopper_image)
    ax2.set_title("Hopper Style\n" + f"{base_prompt} in Edward Hopper style", fontsize=12)
    ax2.axis('off')
    
    # Add a title to the figure
    fig.suptitle("Image Comparison", fontsize=14, y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# Example usage in a Jupyter notebook:
"""
# Import the function
from compare_images import display_image_pair

# Display a pair of images
display_image_pair("a lighthouse")
""" 