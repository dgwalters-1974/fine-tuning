import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from peft import PeftModel, LoraConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_pipeline(model_path="trained_model"):
    """Load the fine-tuned pipeline"""
    logger.info(f"Loading model from {model_path}...")
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model directory {model_path} does not exist!")
    
    # Check for required files
    adapter_config = "adapter_config.json"
    adapter_weights_bin = "adapter_model.bin"
    adapter_weights_safe = "adapter_model.safetensors"
    
    if not os.path.exists(os.path.join(model_path, adapter_config)):
        raise ValueError(f"Missing adapter_config.json in {model_path}")
    
    # Check for either .bin or .safetensors format
    if not (os.path.exists(os.path.join(model_path, adapter_weights_bin)) or 
            os.path.exists(os.path.join(model_path, adapter_weights_safe))):
        raise ValueError(f"No model weights found in {model_path}. Need either {adapter_weights_bin} or {adapter_weights_safe}")
    
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
    logger.info(f"Base pipeline loaded on {device}")
    
    # Configure LoRA with improved parameters
    lora_config = LoraConfig(
        r=32,  # Increased from 16 to 32 for more expressiveness
        lora_alpha=64,  # Increased from 32 to 64 for stronger style influence
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "proj_in", "proj_out",  # Added projection layers
            "conv_in", "conv_out"   # Added convolution layers
        ],
        lora_dropout=0.1,  # Increased dropout slightly for better generalization
        bias="none"
    )
    
    # Load and apply LoRA weights to the UNet
    logger.info("Loading LoRA weights...")
    try:
        # First, ensure the UNet is in eval mode
        pipeline.unet.eval()
        
        # Log the contents of the model directory
        logger.info(f"Contents of {model_path}:")
        for file in os.listdir(model_path):
            logger.info(f"  - {file}")
        
        # Load the LoRA weights
        logger.info("Applying LoRA adapter to UNet...")
        pipeline.unet = PeftModel.from_pretrained(
            pipeline.unet,
            model_path,
            is_trainable=False,
            config=lora_config
        )
        
        # Verify the weights were loaded
        logger.info("Verifying LoRA weights...")
        lora_found = False
        for name, _ in pipeline.unet.named_parameters():
            if 'lora_' in name:
                lora_found = True
                logger.info(f"Found LoRA parameter: {name}")
                break
                
        if not lora_found:
            raise ValueError("No LoRA weights found after loading!")
        
        # Log which format was loaded
        if os.path.exists(os.path.join(model_path, adapter_weights_safe)):
            logger.info("Loaded weights from .safetensors format")
        else:
            logger.info("Loaded weights from .bin format")
        
        logger.info("LoRA weights loaded successfully")
        
        try:
            # Print model analysis
            logger.info("Computing parameter statistics...")
            total_params = sum(p.numel() for name, p in pipeline.unet.named_parameters())
            lora_params = sum(p.numel() for name, p in pipeline.unet.named_parameters() if 'lora_' in name)
            logger.info(f"Total UNet parameters: {total_params:,d}")
            logger.info(f"LoRA parameters: {lora_params:,d}")
            logger.info(f"LoRA parameter %: {(lora_params/total_params)*100:.4f}%")
        except Exception as stat_e:
            logger.warning(f"Could not compute parameter statistics: {str(stat_e)}")
        
    except Exception as e:
        logger.error(f"Error loading LoRA weights: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise
    
    logger.info(f"Model loaded on {device}")
    return pipeline

def setup_base_pipeline():
    """Load the original Stable Diffusion pipeline"""
    logger.info("Loading original Stable Diffusion model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    logger.info(f"Original model loaded on {device}")
    return pipeline

def generate_images(prompt):
    """Generate three images with improved parameters"""
    try:
        logger.info(f"Generating images for prompt: {prompt}")
        
        # Enhanced negative prompt
        negative_prompt = (
            "cartoon, anime, digital art, modern art, abstract, "
            "blurry, bad quality, distorted, ugly, oversaturated, "
            "unrealistic lighting, oversharpened, low resolution, "
            "watermark, signature, text, logo"
        )
        
        # Generate image with original Stable Diffusion
        logger.info("Generating original SD image...")
        base_output = base_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=75,  # Increased steps
            guidance_scale=8.5,      # Increased guidance
            output_type="pil"
        )
        original_image = base_output.images[0]
        
        # Generate image with fine-tuned model
        logger.info("Generating fine-tuned model image...")
        finetuned_output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=75,
            guidance_scale=8.5,
            output_type="pil"
        )
        finetuned_image = finetuned_output.images[0]
        
        # Enhanced Hopper style prompt
        hopper_prompt = (
            f"{prompt}, in Edward Hopper style, dramatic lighting, "
            "strong shadows, urban isolation, cinematic composition, "
            "realistic oil painting, high detail, professional photography"
        )
        
        # Generate image with fine-tuned model and enhanced Hopper style
        logger.info("Generating Hopper style image...")
        hopper_output = pipeline(
            prompt=hopper_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=75,
            guidance_scale=8.5,
            output_type="pil"
        )
        hopper_image = hopper_output.images[0]
        
        # Verify images are not blank
        for img, name in [(original_image, "original"), 
                         (finetuned_image, "fine-tuned"), 
                         (hopper_image, "hopper")]:
            if img.getextrema() == (0, 0):
                logger.error(f"Blank image detected in {name} output!")
                return None, None, None
        
        logger.info("All images generated successfully")
        return original_image, finetuned_image, hopper_image
        
    except Exception as e:
        logger.error(f"Error generating images: {str(e)}")
        return None, None, None

# Initialize both pipelines
try:
    pipeline = setup_pipeline()
    base_pipeline = setup_base_pipeline()
    logger.info("Both pipelines initialized successfully")
except Exception as e:
    logger.error(f"Error initializing pipelines: {str(e)}")
    raise

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Edward Hopper Style Generator") as interface:
        gr.Markdown("""
        # Edward Hopper Style Generator
        
        Enter a description below to generate three images:
        1. Original Stable Diffusion model
        2. Fine-tuned Stable Diffusion model
        3. Fine-tuned model with Edward Hopper style
        
        The fine-tuning process uses LoRA (Low-Rank Adaptation) to modify the UNet's attention layers,
        which are responsible for understanding and applying the style during image generation.
        
        Tips for better results:
        - Be specific about the scene and lighting
        - Include architectural or urban elements
        - Mention time of day or weather conditions
        - Describe the mood or atmosphere
        
        Example prompts:
        - "a city street at night with neon signs and rain-slicked pavement"
        - "a lonely diner at dawn with warm light streaming through windows"
        - "a woman sitting by a window in an empty apartment, afternoon light"
        - "a gas station at night with dramatic shadows and empty highway"
        - "a hotel room with morning light and empty chair by the window"
        - "a train station platform at sunset with long shadows"
        - "a movie theater facade at dusk with marquee lights"
        - "a lighthouse on a rocky coast at twilight"
        """)
        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Enter your description",
                    placeholder="e.g., a lighthouse on a rocky coast at twilight",
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