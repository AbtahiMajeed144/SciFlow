"""
curate_imagenet_vae.py
──────────────────────
Downloads ImageNet-1K via HuggingFace `datasets` in streaming mode,
processes them through a frozen Stable Diffusion VAE encoder,
and saves the compressed latents as raw `.npy` files in an ImageFolder structure:

    output_dir/
        class_0000/
            000000.npy
            000001.npy
            ...
        class_0001/
            ...

Scaling factor of 0.18215 is applied to the latents before saving.

**Optimization**: Uses a multi-threaded Producer-Consumer model. 
A background thread continuously streams and queues images while the 
main thread processes the GPU encoding and saves to disk, eliminating I/O bottlenecks.

Usage (Kaggle):
    pip install datasets diffusers accelerate torchvision
    python curate_imagenet_vae.py --split train --output_dir /kaggle/working/imagenet_latents/train --batch_size 128
"""

import os
import argparse
import numpy as np
import torch
import threading
import queue
import time
from tqdm import tqdm
from torchvision import transforms
from datasets import load_dataset
from diffusers import AutoencoderKL

def class_folder(label: int) -> str:
    return f"class_{label:04d}"

def download_producer(ds, batch_size, data_queue, max_samples):
    """
    Background thread that streams images from HuggingFace and pushes batches into the queue.
    """
    batch_images = []
    batch_labels = []
    processed_count = 0
    
    try:
        for sample in ds:
            batch_images.append(sample["image"])
            batch_labels.append(sample["label"])
            
            if len(batch_images) >= batch_size:
                # This will block if the queue is full (maxsize reached)
                data_queue.put((batch_images, batch_labels))
                processed_count += len(batch_images)
                
                batch_images = []
                batch_labels = []
                
                if max_samples and processed_count >= max_samples:
                    break
                    
        # Push any remaining images as a final partial batch
        if len(batch_images) > 0 and (max_samples is None or processed_count < max_samples):
            data_queue.put((batch_images, batch_labels))
            
    except Exception as e:
        print(f"\n[Producer Error] Download interrupted: {e}")
        
    finally:
        # Sentinel value to tell the consumer that downloading is finished
        data_queue.put(None)

def main():
    parser = argparse.ArgumentParser(description="Curate ImageNet-1K latents using SD VAE")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test"],
                        help="Which split to process")
    parser.add_argument("--output_dir", type=str, default="./imagenet_latents/train",
                        help="Root directory for the .npy output")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for VAE encoding")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on total latents (useful for quick tests)")
    parser.add_argument("--queue_size", type=int, default=5,
                        help="Number of pre-fetched batches to hold in RAM")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face API token for gated datasets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load VAE and Setup Device ──────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading VAE on {device}...")
    
    # Cast to float16 if on GPU to save VRAM and speed up inference
    weight_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=weight_dtype)
    vae = vae.to(device)
    vae.eval()
    
    # Freeze VAE gradients
    for param in vae.parameters():
        param.requires_grad = False

    # ── 2. Data Transform Pipeline ────────────────────────────────────────────
    # Resize to 256x256, convert to tensor, and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # ── 3. Start the Producer Thread ──────────────────────────────────────────
    print(f"Streaming ImageNet-1K split='{args.split}' in the background...")
    
    # Try argument, then environment variable
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    # If still not found, check Kaggle Secrets natively
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    
    if not hf_token:
        print("\n[ERROR] Hugging Face token is missing!")
        print("Because ImageNet-1k is a gated dataset, you MUST provide a token.")
        print("Usage: python curate_imagenet_vae.py ... --hf_token hf_YOUR_TOKEN")
        return
        
    ds = load_dataset("ILSVRC/imagenet-1k", split=args.split, streaming=True, token=hf_token)
    
    # We use a Queue to pass batches between the download thread and the GPU thread.
    # Limiting maxsize prevents RAM OOM if the internet is faster than the GPU.
    data_queue = queue.Queue(maxsize=args.queue_size)
    
    producer = threading.Thread(
        target=download_producer,
        args=(ds, args.batch_size, data_queue, args.max_samples),
        daemon=True
    )
    producer.start()

    counters = {}
    saved_count = 0
    start_time = time.time()
    
    # ImageNet train has 1,281,167 images. Validation has 50,000.
    total_images = args.max_samples if args.max_samples else (1281167 if args.split == "train" else 50000)
    pbar = tqdm(total=total_images, desc="Compressing Latents", unit="img")

    def process_and_save_batch(images, labels):
        nonlocal saved_count
        
        # Preprocess images
        tensors = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensors.append(transform(img))
            
        batch_tensor = torch.stack(tensors).to(device, dtype=weight_dtype)
        
        # Encode with VAE
        with torch.no_grad():
            latents = vae.encode(batch_tensor).latent_dist.sample()
            # CRITICAL: Apply the exact scaling factor 0.18215
            latents = latents * 0.18215
            
        latents_np = latents.cpu().to(torch.float32).numpy()
        
        # Save each latent
        for latent, label in zip(latents_np, labels):
            folder = os.path.join(args.output_dir, class_folder(label))
            os.makedirs(folder, exist_ok=True)
            
            idx = counters.get(label, 0)
            counters[label] = idx + 1
            
            filepath = os.path.join(folder, f"{idx:06d}.npy")
            np.save(filepath, latent)
            
            saved_count += 1
            
        pbar.update(len(images))
        pbar.set_postfix({"Classes": len(counters)})

    # ── 4. Main Iteration Loop (Consumer) ─────────────────────────────────────
    print("GPU consumer is waiting for the first batch...")
    while True:
        # This will block until the producer puts a batch in the queue
        batch_data = data_queue.get()
        
        if batch_data is None:
            # Sentinel value received, meaning download is complete
            break
            
        batch_images, batch_labels = batch_data
        
        # GPU processes the batch while the producer thread concurrently downloads the next
        process_and_save_batch(batch_images, batch_labels)
        
        data_queue.task_done()

    pbar.close()
    producer.join()

    total_time = time.time() - start_time
    print(f"\nDone! {saved_count} latents saved to {args.output_dir}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Classes with data: {len(counters)}")

if __name__ == "__main__":
    main()
