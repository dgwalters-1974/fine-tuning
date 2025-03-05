import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# ==========================
# Configuration
# ==========================
model_id = "CompVis/stable-diffusion-v1-4"  # Base model identifier
fine_tuned_unet_dir = "stable-diffusion-lora-finetuned"  # Directory where your fine-tuned UNet is saved
prompt = "a serene urban landscape in the style of Edward Hopper"  # Example prompt
num_inference_steps = 50  # Number of denoising steps
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# Load Base Pipeline
# ==========================
# Load the full pipeline in half-precision.
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    variant="fp16",  # Use variant="fp16" for half-precision weights.
    torch_dtype=torch.float16
)
pipeline.to(device)

# ==========================
# Replace the UNet with the Fine-Tuned LoRA UNet
# ==========================
# This assumes that the fine-tuned UNet was saved using the PEFT/LoRA approach.
# It loads the LoRA-adapted UNet and replaces the base UNet in the pipeline.
pipeline.unet = pipeline.unet.from_pretrained(fine_tuned_unet_dir)
pipeline.unet.to(device)

# ==========================
# Inference
# ==========================
# Generate an image using the fine-tuned model.
with torch.autocast(device):
    output = pipeline(prompt, num_inference_steps=num_inference_steps)

# Extract the generated image.
generated_image = output.images[0]

# ==========================
# Display and Save the Image
# ==========================
plt.figure(figsize=(8, 8))
plt.imshow(generated_image)
plt.axis("off")
plt.title("Generated Image")
plt.show()

# Optionally, save the image to disk.
generated_image.save("hopper_finetuned_output.jpg")
print("Image saved as 'hopper_finetuned_output.jpg'.")
