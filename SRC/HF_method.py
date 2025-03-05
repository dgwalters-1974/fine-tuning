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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float32  # Use float32 for better stability
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
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    
    # Use a lower learning rate for LoRA
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=1e-5,  # Lower learning rate for LoRA
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )
    
    # Add gradient clipping
    max_grad_norm = 1.0
    
    # Initialize accelerator with no mixed precision for now
    accelerator = Accelerator(
        mixed_precision="no",  # Disable mixed precision to avoid FP16 gradient issues
        gradient_accumulation_steps=4,
        device_placement=True
    )
    
    # Prepare models and optimizer
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )
    
    num_epochs = 3
    progress_bar = tqdm(range(num_epochs), desc="Epochs")
    
    # Initialize loss tracking
    running_loss = 0.0
    num_batches = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            try:
                # Convert pixel values to float32
                pixel_values = batch["pixel_values"].to(device).float()
                
                with torch.no_grad():
                    tokens = tokenizer(
                        batch["caption"],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids.to(device)
                    
                    encoder_hidden_states = text_encoder(tokens)[0]
                
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Scale down the latents to prevent numerical instability
                latents = latents * 0.18215  # Additional scaling factor
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                       (latents.shape[0],), device=device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Forward pass
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch+1}, step {step}. Skipping batch...")
                    optimizer.zero_grad()
                    continue
                
                accelerator.backward(loss)
                
                # Clip gradients using standard PyTorch method
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # Update loss tracking
                epoch_loss += loss.item()
                running_loss += loss.item()
                num_batches += 1
                
                if step % 10 == 0:
                    avg_loss = running_loss / num_batches
                    print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
                    running_loss = 0.0
                    num_batches = 0
                
                del noise_pred, latents, noisy_latents
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at epoch {epoch}, step {step}. Clearing cache...")
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    continue
                raise e
        
        # Print epoch statistics
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint after each epoch
        save_dir = f"checkpoints/epoch_{epoch+1}"
        os.makedirs(save_dir, exist_ok=True)
        unet.save_pretrained(save_dir)
        print(f"Saved checkpoint to {save_dir}")
        
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
        
        # Create DataLoader with larger batch size
        train_dataloader = DataLoader(
            dataset,
            batch_size=4,  # Increased batch size
            shuffle=True,  # Enable shuffling
            num_workers=2,  # Add workers for faster data loading
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
        
        # Save LoRA weights
        save_dir = "trained_model"
        os.makedirs(save_dir, exist_ok=True)
        trained_unet.save_pretrained(save_dir)
        print(f"LoRA weights saved to {save_dir}")
        
        # Print model analysis
        analyze_model(trained_unet, "Final Trained Model Analysis")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nDebug information:")
        print(f"Device: {device}")
        if device == "cuda":
            try:
                print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            except Exception as mem_e:
                print(f"Could not get memory info: {str(mem_e)}")
        raise

if __name__ == "__main__":
    main()

# Remove everything after this point
