import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, get_scheduler
from accelerate import Accelerator
import numpy as np
from tqdm.auto import tqdm
from contextlib import nullcontext
import time
import signal
import sys
import os
import gc
from torchvision import transforms
import itertools
from torchvision.transforms import functional as TF  # Use torchvision's functional instead of F

# Import PEFT/LoRA utilities
from peft import LoraConfig, get_peft_model

# Add these utility functions
def print_trainable_parameters(model):
    """Simplified parameter counter"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params:,d} | Total: {all_param:,d} | Trainable%: {100 * trainable_params / all_param:.2f}%")

def analyze_model(model, title="Model Analysis"):
    """
    Analyze and print detailed information about the model architecture,
    parameters, and LoRA modifications.
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}\n")

    # 1. Basic Model Information
    print("1. Model Class:", model.__class__.__name__)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Format large numbers
    def format_number(num):
        if num >= 1e9:
            return f"{num/1e9:.2f}B"
        if num >= 1e6:
            return f"{num/1e6:.2f}M"
        if num >= 1e3:
            return f"{num/1e3:.2f}K"
        return str(num)
    
    print(f"Total Parameters: {format_number(total_params)}")
    print(f"Trainable Parameters: {format_number(trainable_params)}")
    print(f"Percentage Trainable: {(trainable_params/total_params)*100:.2f}%\n")

    # 2. LoRA Analysis
    print("2. LoRA Modules Analysis:")
    lora_params = 0
    found_lora = False
    
    for name, module in model.named_modules():
        if 'lora_' in name:
            found_lora = True
            print(f"\nFound LoRA in: {name}")
            if hasattr(module, 'weight'):
                print(f"Shape: {module.weight.shape}")
                lora_params += module.weight.numel()

    if found_lora:
        print(f"\nTotal LoRA Parameters: {format_number(lora_params)}")
        print(f"LoRA Parameter %: {(lora_params/total_params)*100:.4f}%")
    else:
        print("\nNo LoRA modules found!")

    # 3. Device Information
    print("\n3. Device Information:")
    print(f"Model is on: {next(model.parameters()).device}")
    
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Running on Apple Silicon (MPS)")

def inspect_lora_weights(model):
    """
    Print the actual values of LoRA weights
    """
    for name, param in model.named_parameters():
        if 'lora_' in name:
            print(f"\nLoRA Weight: {name}")
            print(f"Shape: {param.shape}")
            print(f"Mean: {param.mean().item():.6f}")
            print(f"Std: {param.std().item():.6f}")
            print(f"Min: {param.min().item():.6f}")
            print(f"Max: {param.max().item():.6f}")

# Model Configuration
MODEL_CONFIG = {
    "name": "runwayml/stable-diffusion-v1-5",
    "version": "v1.5",
    "description": "Improved SD version with better image quality and prompt understanding",
    "vae_path": None,  # Set to a specific VAE if needed
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ],
        "lora_dropout": 0.05,
        "bias": "none",
    }
}

def setup_mps_device():
    """Setup MPS device with proper memory management"""
    if torch.backends.mps.is_available():
        # Set a conservative memory limit (70% of available memory)
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'
        device = "mps"
    else:
        device = "cpu"
    
    return device, torch.float32

def setup_model_and_pipeline():
    """Setup model pipeline with proper error handling"""
    try:
        # Setup device first
        device, torch_dtype = setup_mps_device()
        print(f"Using device: {device}, dtype: {torch_dtype}")
        
        # Load pipeline components
        pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_CONFIG["name"],
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Move to device safely
        try:
            pipeline = pipeline.to(device)
        except RuntimeError as e:
            print(f"Error moving pipeline to {device}, falling back to CPU: {str(e)}")
            device = "cpu"
            pipeline = pipeline.to(device)
        
        print(f"Model loaded on: {device}")
        
        return (
            pipeline,
            pipeline.unet,
            pipeline.vae,
            pipeline.text_encoder,
            pipeline.tokenizer,
            pipeline.scheduler,
            device,
            torch_dtype
        )
        
    except Exception as e:
        print(f"Error in setup_model_and_pipeline: {str(e)}")
        raise

def load_and_prepare_dataset():
    """Load the Hopper dataset"""
    print("Loading Hopper fine-tuning dataset...")
    
    try:
        dataset = load_dataset(
            "kooshkashkai/hopper-finetuning-dataset",
            cache_dir="dataset_cache"
        )
        print(f"Dataset loaded with {len(dataset['train'])} examples")
        return dataset["train"]
    except Exception as e:
        print(f"Dataset loading failed: {str(e)}")
        raise

def collate_fn(examples):
    """Process batch of examples with shape verification"""
    pixel_values = []
    captions = []
    
    for example in examples:
        pv = example["pixel_values"]
        if not isinstance(pv, torch.Tensor):
            pv = torch.tensor(pv)
        
        pv = pv.float()
        
        # Ensure correct shape [C, H, W]
        if len(pv.shape) == 3:
            if pv.shape[2] == 3:  # If [H, W, C]
                pv = pv.permute(2, 0, 1)  # Convert to [C, H, W]
        
        # Resize if needed
        if pv.shape[-2:] != (512, 512):
            pv = TF.resize(pv, [512, 512], antialias=True)
        
        # Normalize
        pv = (pv / 127.5) - 1.0
        
        pixel_values.append(pv)
        captions.append(example["caption"])
    
    return {
        "pixel_values": torch.stack(pixel_values),
        "caption": captions
    }

class ProgressDataLoader:
    """Wrapper for DataLoader with progress bar"""
    def __init__(self, dataloader, desc="Processing batches"):
        self.dataloader = dataloader
        self.desc = desc
        self.total = len(dataloader)
    
    def __iter__(self):
        return iter(tqdm(
            self.dataloader,
            desc=self.desc,
            total=self.total,
            unit="batch"
        ))
    
    def __len__(self):
        return len(self.dataloader)

def validate_batch(batch, batch_idx=0):
    """Validate batch contents with timing"""
    start_time = time.time()
    
    print(f"\nValidating batch {batch_idx}:")
    print(f"├─ Pixel values:")
    print(f"│  ├─ Shape: {batch['pixel_values'].shape}")
    print(f"│  ├─ Type: {batch['pixel_values'].dtype}")
    print(f"│  ├─ Range: [{batch['pixel_values'].min():.2f}, {batch['pixel_values'].max():.2f}]")
    print(f"│  └─ Memory: {batch['pixel_values'].element_size() * batch['pixel_values'].nelement() / 1024:.2f} KB")
    print(f"├─ Captions:")
    print(f"│  ├─ Count: {len(batch['caption'])}")
    print(f"│  └─ First caption length: {len(batch['caption'][0])}")
    
    end_time = time.time()
    print(f"└─ Validation time: {(end_time - start_time)*1000:.2f}ms")

def train_unet(unet, vae, text_encoder, train_dataloader, tokenizer, noise_scheduler, device):
    """Training function with memory management"""
    print("Starting training...")
    
    # Setup models
    unet.train()
    vae.eval()
    text_encoder.eval()
    
    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )
    
    num_epochs = 3
    progress_bar = tqdm(range(num_epochs), desc="Epochs")
    
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            try:
                if device == "mps":
                    torch.mps.empty_cache()
                
                with torch.no_grad():
                    tokens = tokenizer(
                        batch["caption"],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids.to(device)
                    
                    encoder_hidden_states = text_encoder(tokens)[0]
                
                pixel_values = batch["pixel_values"].to(device)
                
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                       (latents.shape[0],), device=device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                if step % 10 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
                
                del noise_pred, latents, noisy_latents
                if device == "mps":
                    torch.mps.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at epoch {epoch}, step {step}. Clearing cache...")
                    if device == "mps":
                        torch.mps.empty_cache()
                    continue
                raise e
        
        progress_bar.update(1)
    
    return unet

def main():
    # Initialize device variable at the start
    device = "cpu"  # Default fallback
    
    try:
        # Setup
        (pipeline, unet, vae, text_encoder, tokenizer, 
         noise_scheduler, device, torch_dtype) = setup_model_and_pipeline()
        
        # Load dataset
        print("\n=== Loading Dataset ===")
        dataset = load_and_prepare_dataset()
        
        # Create DataLoader
        train_dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # Train
        trained_unet = train_unet(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            train_dataloader=train_dataloader,
            tokenizer=tokenizer,
            noise_scheduler=noise_scheduler,
            device=device
        )
        
        # Save model
        save_dir = "trained_model"
        os.makedirs(save_dir, exist_ok=True)
        pipeline.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nDebug information:")
        print(f"Device: {device}")
        if device == "mps":
            try:
                print(f"Memory allocated: {torch.mps.current_allocated_memory() / 1e9:.2f}GB")
            except Exception as mem_e:
                print(f"Could not get memory info: {str(mem_e)}")
        raise

if __name__ == "__main__":
    main()

# ==========================
# Configuration & Hyperparameters
# ==========================
model_id = "CompVis/stable-diffusion-v1-4"  # Base model identifier
batch_size = 4
learning_rate = 1e-4
num_epochs = 3

# Device and dtype setup
def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        return "mps", torch.float32  # MPS needs float32
    return "cpu", torch.float32

device, dtype = get_device_and_dtype()
print(f"Using device: {device}, dtype: {dtype}")

# ==========================
# Load HF Dataset
# ==========================
# Our dataset: Hopper fine-tuning dataset
dataset = load_dataset("kooshkashkai/hopper-finetuning-dataset", split="train")

def collate_fn(examples):
    # Convert each pixel_values (stored as a NumPy array) into a torch.Tensor.
    images = [torch.tensor(example["pixel_values"]) for example in examples]
    captions = [example["caption"] for example in examples]
    return {"pixel_values": torch.stack(images), "captions": captions}

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# ==========================
# Load Stable Diffusion Components
# ==========================
# Load the full Stable Diffusion pipeline.
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,  # Use the determined dtype
)
pipeline.to(device)

# Extract individual components.
vae = pipeline.vae                # VAE: encodes images into latent space.
text_encoder = pipeline.text_encoder  # CLIP text encoder.
tokenizer = pipeline.tokenizer     # CLIP tokenizer.
unet = pipeline.unet               # UNet: to be fine-tuned.
noise_scheduler = pipeline.scheduler  # Diffusion noise scheduler.

# Freeze VAE and text encoder.
for param in vae.parameters():
    param.requires_grad = False
for param in text_encoder.parameters():
    param.requires_grad = False

# ==========================
# Integrate LoRA into UNet
# ==========================
# List of valid target modules for Stable Diffusion U-Net
TARGET_MODULES = [
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
]

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
)

# Apply LoRA to U-Net
unet = pipeline.unet
unet = get_peft_model(unet, lora_config)
unet.to(device).to(dtype)

# Verify the setup
print("Model device:", next(unet.parameters()).device)
print("LoRA applied successfully!")

# Optional: Print trainable parameters
print_trainable_parameters(unet)

# ==========================
# Set Up Optimizer
# ==========================
optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

# ==========================
# Initialize Accelerator (for multi-GPU/distributed training)
# ==========================
accelerator = Accelerator(
    mixed_precision="no",  # Disable mixed precision to ensure dtype consistency
    gradient_accumulation_steps=4
)

print(f"Using device: {accelerator.device}")
print(f"Mixed precision: {accelerator.mixed_precision}")

# ==========================
# Training Loop
# ==========================
accelerator.print("Starting LoRA fine-tuning for Hopper style...")

for epoch in range(num_epochs):
    accelerator.print(f"Epoch {epoch+1}/{num_epochs}")
    for batch in dataloader:
        optimizer.zero_grad()
        
        # --- Prepare Inputs ---
        # Move images to the accelerator's device and cast to float16.
        images = batch["pixel_values"].to(accelerator.device).to(dtype)  # [B, C, H, W]
        captions = batch["captions"]
        
        # Tokenize captions.
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.to(accelerator.device)
        
        # Encode text into hidden states.
        encoder_hidden_states = text_encoder(input_ids)[0]
        
        # --- Encode Images ---
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        
        # --- Add Noise ---
        noise = torch.randn(latents.shape, device=accelerator.device, dtype=latents.dtype)
        bs = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=accelerator.device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # --- Forward Pass through LoRA-adapted UNet ---
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # --- Compute Loss ---
        loss = F.mse_loss(noise_pred, noise)
        
        accelerator.backward(loss)
        optimizer.step()
        
        accelerator.print(f"Loss: {loss.item():.4f}")
    
    # Optionally save a checkpoint after each epoch.
    torch.save(unet.state_dict(), f"unet_epoch_{epoch+1}_lora.pt")

# ==========================
# Save the Fine-Tuned Model
# ==========================
unet.save_pretrained("stable-diffusion-lora-finetuned")
accelerator.print("LoRA fine-tuning complete. Model saved as 'stable-diffusion-lora-finetuned'.")

# Function to print model structure
def print_model_structure(model, depth=0, max_depth=2):
    """Print the structure of the model's named modules."""
    for name, module in model.named_modules():
        if depth <= max_depth:
            print("  " * depth + f"{name}: {module.__class__.__name__}")

# Inspect U-Net structure
print("U-Net Structure:")
print_model_structure(pipeline.unet)

# Training function with device handling
def process_batch(batch, vae, dtype):
    """Process a batch of images through VAE with correct dtype"""
    with torch.no_grad():
        # Ensure images are in the correct dtype
        images = batch["pixel_values"].to(device).to(dtype)
        latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
    return latents

# Global flag for graceful interruption
should_stop = False

def signal_handler(signum, frame):
    """Handle interrupt signal"""
    global should_stop
    if not should_stop:
        print("\nInterrupt received. Will stop after current epoch. Press Ctrl+C again to force quit.")
        should_stop = True
    else:
        print("\nForce quitting...")
        sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

# Memory management settings for MPS
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'  # Adjust memory limit
torch.mps.empty_cache()

def optimize_memory():
    torch.mps.empty_cache()
    gc.collect()
    print("Memory cleared")

# Start training with memory optimization
print("Initializing training...")
optimize_memory()

try:
    trained_unet = train_unet(
        unet,
        vae,
        text_encoder,
        dataset,
        tokenizer,
        noise_scheduler,
        device=device
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed: {e}")
    raise

# Save the final model
pipeline.save_pretrained("fine_tuned_model")

# Test model loading
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
)
print("Model loaded successfully!")

# Verify U-Net structure
print("\nU-Net target modules:")
print_model_structure(pipeline.unet)
