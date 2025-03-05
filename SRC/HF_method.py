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

# Import PEFT/LoRA utilities
from peft import LoraConfig, get_peft_model

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
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )

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

def train_unet(
    unet,
    vae,
    text_encoder,
    dataset,
    num_epochs=50,
    batch_size=1,  # Reduced batch size for MPS
    learning_rate=1e-4,
    gradient_accumulation_steps=4,
    save_dir="checkpoints"
):
    global should_stop
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup accelerator with memory optimization
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[{"no_sync_in_gradient_accumulation": True}]
    )

    # Create dataloader with smaller batch size
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    # Optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    # Prepare everything
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    # Training metrics
    best_loss = float('inf')
    running_loss = []
    
    # Setup progress bars
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)
    batch_pbar = tqdm(total=len(train_dataloader), desc="Batches", position=1, leave=False)
    metrics_pbar = tqdm(total=0, bar_format='{desc}', position=2)

    try:
        start_time = time.time()
        
        for epoch in range(num_epochs):
            if should_stop:
                print("\nGracefully stopping training...")
                break
                
            unet.train()
            epoch_loss = 0
            batch_pbar.reset()
            
            for step, batch in enumerate(train_dataloader):
                # Memory management
                if step % 10 == 0:
                    torch.mps.empty_cache()
                    gc.collect()

                with accelerator.accumulate(unet):
                    try:
                        # Process in smaller chunks if needed
                        with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                            # Forward pass with memory optimization
                            with torch.no_grad():
                                encoder_hidden_states = text_encoder(
                                    batch["input_ids"].to(accelerator.device)
                                )[0]
                                
                                latents = vae.encode(
                                    batch["pixel_values"].to(accelerator.device)
                                ).latent_dist.sample()
                                latents = latents * vae.config.scaling_factor

                            # Free up memory
                            del batch["pixel_values"]
                            torch.mps.empty_cache()

                            # Forward pass through U-Net
                            noise_pred = unet(
                                latents,
                                encoder_hidden_states=encoder_hidden_states
                            ).sample

                            # Calculate loss
                            loss = torch.nn.functional.mse_loss(
                                noise_pred.float(),
                                batch["target"].to(accelerator.device).float()
                            )

                        # Backward pass
                        accelerator.backward(loss)

                        # Gradient clipping
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), max_norm=1.0)

                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)  # Memory optimization

                        # Update metrics
                        epoch_loss += loss.item()
                        running_loss.append(loss.detach().item())
                        if len(running_loss) > 50:  # Keep shorter running average
                            running_loss.pop(0)

                        # Update progress
                        avg_loss = sum(running_loss) / len(running_loss)
                        metrics_pbar.set_description_str(
                            f"Loss: {avg_loss:.4f} | Batch: {step}/{len(train_dataloader)}"
                        )
                        batch_pbar.update(1)

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.mps.empty_cache()
                            gc.collect()
                            print(f"\nOOM error in batch {step}. Skipping...")
                            continue
                        raise e

                # Save checkpoint periodically
                if step % 100 == 0:
                    accelerator.save_state(f"{save_dir}/checkpoint-epoch{epoch}-step{step}")
            
            # Update epoch progress
            epoch_pbar.update(1)
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                accelerator.save_state(f"{save_dir}/checkpoint-{epoch+1}")
                
            # Save latest state
            accelerator.save_state(f"{save_dir}/latest")

    except Exception as e:
        print(f"\nError during training: {e}")
        # Save emergency checkpoint
        accelerator.save_state(f"{save_dir}/emergency_save")
        raise
        
    finally:
        # Clean up progress bars
        batch_pbar.close()
        epoch_pbar.close()
        metrics_pbar.close()
        
        # Save final model
        if not should_stop:  # Only if training completed normally
            accelerator.save_state(f"{save_dir}/final_model")
        
        # Print training summary
        print("\nTraining Summary:")
        print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Final loss: {avg_loss:.4f}")

    return unet

# Verify setup
print("\nVerifying setup:")
print(f"VAE dtype: {next(vae.parameters()).dtype}")
print(f"UNet dtype: {next(unet.parameters()).dtype}")
print(f"Text Encoder dtype: {next(text_encoder.parameters()).dtype}")

# Memory optimization settings
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
        num_epochs=50,
        batch_size=1,  # Keep batch size small
        learning_rate=1e-4,
        gradient_accumulation_steps=4
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
