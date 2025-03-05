import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from tqdm import tqdm
from peft import PeftModel, LoraConfig

def setup_pipeline(model_path="trained_model"):
    """Load the fine-tuned pipeline"""
    print(f"Loading model from {model_path}...")
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model directory {model_path} does not exist!")
    
    # List contents of model directory
    print("\nModel directory contents:")
    for item in os.listdir(model_path):
        print(f"- {item}")
    
    # Load the base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",  # Use the same base model as training
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    
    # Load and apply LoRA weights
    print("\nLoading LoRA weights...")
    pipeline.unet = PeftModel.from_pretrained(
        pipeline.unet,
        model_path,
        is_trainable=False,
        config=lora_config
    )
    
    print(f"\nModel loaded on {device}")
    
    # Print model information
    print("\nModel information:")
    print(f"- UNet device: {pipeline.unet.device}")
    print(f"- VAE device: {pipeline.vae.device}")
    print(f"- Text encoder device: {pipeline.text_encoder.device}")
    
    return pipeline

def generate_image(pipeline, prompt, negative_prompt=None, num_inference_steps=50, guidance_scale=7.5):
    """Generate a single image with the given prompt"""
    if negative_prompt is None:
        negative_prompt = "blurry, bad quality, distorted, ugly"
    
    print(f"\nGenerating with parameters:")
    print(f"- Steps: {num_inference_steps}")
    print(f"- Guidance scale: {guidance_scale}")
    print(f"- Prompt: {prompt}")
    print(f"- Negative prompt: {negative_prompt}")
    
    # Generate the image
    with torch.no_grad():
        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pil"
        )
    
    image = output.images[0]
    
    # Verify image is not blank
    if image.getextrema() == (0, 0):
        print("WARNING: Generated image is completely black!")
    elif image.getextrema() == (255, 255):
        print("WARNING: Generated image is completely white!")
    
    return image

def save_image(image, prompt, output_dir="generated_images"):
    """Save the generated image with a filename based on the prompt"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a filename from the prompt (first 50 characters)
    filename = prompt[:50].replace(" ", "_").lower()
    filepath = os.path.join(output_dir, f"{filename}.png")
    
    # Save the image
    image.save(filepath)
    print(f"Saved image to {filepath}")
    return filepath

def test_model():
    """Test the model with various prompts"""
    # Load the pipeline
    pipeline = setup_pipeline()
    
    # Base prompts without style specification
    base_prompts = [
        "a city street at night",
        "a lonely diner",
        "a woman sitting by a window",
        "a gas station at night",
        "a hotel room",
        "a train station",
        "a movie theater",
        "a lighthouse"
    ]
    
    # Generate images for each prompt
    for base_prompt in tqdm(base_prompts, desc="Generating images"):
        # Generate version without style
        print(f"\nGenerating image for base prompt: {base_prompt}")
        try:
            image = generate_image(pipeline, base_prompt)
            save_image(image, base_prompt)
        except Exception as e:
            print(f"Error generating image for base prompt '{base_prompt}': {str(e)}")
            print("Full error details:", e)
        
        # Generate version with Hopper style
        hopper_prompt = f"{base_prompt} in Edward Hopper style"
        print(f"\nGenerating image for Hopper style prompt: {hopper_prompt}")
        try:
            image = generate_image(pipeline, hopper_prompt)
            save_image(image, hopper_prompt)
        except Exception as e:
            print(f"Error generating image for Hopper style prompt '{hopper_prompt}': {str(e)}")
            print("Full error details:", e)

def main():
    """Main function to run the inference"""
    try:
        test_model()
        print("\nInference completed successfully!")
    except Exception as e:
        print(f"\nError during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main() 