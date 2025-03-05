import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
import numpy as np

# ==========================
# Configuration & Hyperparameters
# ==========================
model_id = "CompVis/stable-diffusion-v1-4"  # or your preferred base model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
learning_rate = 1e-4
num_epochs = 3

# ==========================
# Load HF Dataset
# ==========================
# Replace with your dataset repo or local path
dataset = load_dataset("your-username/my-finetuning-dataset", split="train")

# Create a PyTorch DataLoader from the HF dataset.
def collate_fn(examples):
    # Convert each pixel_values (a NumPy array) to a torch.Tensor.
    images = [torch.tensor(example["pixel_values"]) for example in examples]
    captions = [example["caption"] for example in examples]
    return {"pixel_values": torch.stack(images), "captions": captions}

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# ==========================
# Load Stable Diffusion Components
# ==========================
# We load the complete pipeline but then extract the individual components.
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    revision="fp16",               # use fp16 for memory efficiency, if available
    torch_dtype=torch.float16
)
pipeline.to(device)

# Extract components
vae = pipeline.vae  # Variational Autoencoder
text_encoder = pipeline.text_encoder  # CLIP text encoder
tokenizer = pipeline.tokenizer  # CLIP tokenizer
unet = pipeline.unet  # UNet that we will fine-tune
noise_scheduler = pipeline.scheduler  # noise scheduler for diffusion

# Freeze VAE and text encoder to save memory and training time.
for param in vae.parameters():
    param.requires_grad = False
for param in text_encoder.parameters():
    param.requires_grad = False

# (Optionally freeze parts of UNet or add LoRA adapters here.)
for param in unet.parameters():
    param.requires_grad = True

# ==========================
# Set Up Optimizer
# ==========================
optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

# ==========================
# Training Loop
# ==========================
print("Starting fine-tuning...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for batch in dataloader:
        optimizer.zero_grad()
        
        # --- Prepare inputs ---
        # Convert images to device and ensure they are float16 (as expected)
        images = batch["pixel_values"].to(device).to(torch.float16)  # shape: [B, C, H, W]
        captions = batch["captions"]
        
        # Tokenize captions
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.to(device)
        
        # Encode text into hidden states
        encoder_hidden_states = text_encoder(input_ids)[0]
        
        # Encode images to latent space using the VAE (without gradients)
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        
        # Sample random noise to add to the latents
        noise = torch.randn(latents.shape, device=device, dtype=latents.dtype)
        bs = latents.shape[0]
        # Sample random timesteps for each image in the batch
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
        
        # Add noise to the latents according to the diffusion schedule
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # --- Forward Pass through UNet ---
        # The UNet should predict the noise residual given the noisy latents, timesteps, and text conditioning.
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # Compute mean-squared error loss between the predicted noise and the actual noise.
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")
    
    # Optionally, save checkpoints each epoch
    torch.save(unet.state_dict(), f"unet_epoch_{epoch+1}.pt")

# ==========================
# Save Fine-Tuned Model
# ==========================
# You can save just the UNet or rebuild the entire pipeline.
unet.save_pretrained("stable-diffusion-finetuned")
print("Fine-tuning complete. Model saved as 'stable-diffusion-finetuned'.")
