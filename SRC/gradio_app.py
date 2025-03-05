import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from peft import PeftModel, LoraConfig

def setup_pipeline(model_path="trained_model"):
    """Load the fine-tuned pipeline"""
    print(f"Loading model from {model_path}...")
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model directory {model_path} does not exist!")
    
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
    return pipeline

def setup_base_pipeline():
    """Load the original Stable Diffusion pipeline"""
    print("Loading original Stable Diffusion model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    print(f"Original model loaded on {device}")
    return pipeline

def generate_images(prompt):
    """Generate three images: original SD, fine-tuned SD, and fine-tuned SD with Hopper style"""
    try:
        # Generate image with original Stable Diffusion
        base_output = base_pipeline(
            prompt=prompt,
            negative_prompt="blurry, bad quality, distorted, ugly",
            num_inference_steps=50,
            guidance_scale=7.5,
            output_type="pil"
        )
        original_image = base_output.images[0]
        
        # Generate image with fine-tuned model
        finetuned_output = pipeline(
            prompt=prompt,
            negative_prompt="blurry, bad quality, distorted, ugly",
            num_inference_steps=50,
            guidance_scale=7.5,
            output_type="pil"
        )
        finetuned_image = finetuned_output.images[0]
        
        # Generate image with fine-tuned model and Hopper style
        hopper_prompt = f"{prompt} in Edward Hopper style"
        hopper_output = pipeline(
            prompt=hopper_prompt,
            negative_prompt="blurry, bad quality, distorted, ugly",
            num_inference_steps=50,
            guidance_scale=7.5,
            output_type="pil"
        )
        hopper_image = hopper_output.images[0]
        
        return original_image, finetuned_image, hopper_image
        
    except Exception as e:
        print(f"Error generating images: {str(e)}")
        return None, None, None

# Initialize both pipelines
pipeline = setup_pipeline()
base_pipeline = setup_base_pipeline()

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Edward Hopper Style Generator") as interface:
        gr.Markdown("""
        # Edward Hopper Style Generator
        
        Enter a description below to generate three images:
        1. Original Stable Diffusion model
        2. Fine-tuned Stable Diffusion model
        3. Fine-tuned model with Edward Hopper style
        
        Example prompts:
        - "a city street at night"
        - "a lonely diner"
        - "a woman sitting by a window"
        - "a gas station at night"
        - "a hotel room"
        - "a train station"
        - "a movie theater"
        - "a lighthouse"
        """)
        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Enter your description",
                    placeholder="e.g., a lighthouse",
                    lines=2
                )
                generate_btn = gr.Button("Generate Images")
            
            with gr.Column():
                with gr.Row():
                    original_output = gr.Image(label="Original Stable Diffusion")
                    finetuned_output = gr.Image(label="Fine-tuned Model")
                    hopper_output = gr.Image(label="Hopper Style")
        
        generate_btn.click(
            fn=generate_images,
            inputs=prompt_input,
            outputs=[original_output, finetuned_output, hopper_output]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        share=True,  # Enable sharing via public URL
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860  # Default Gradio port
    ) 