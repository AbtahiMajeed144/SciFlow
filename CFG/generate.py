import os
import argparse
import torch
from torchvision.utils import save_image
from diffusers import AutoencoderKL

from models import KARTFlowModel
from utils import load_config


def main():
    parser = argparse.ArgumentParser(description="1-Step Inference Script for KART-Flow")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset-specific config")
    parser.add_argument("--global_config", type=str, default="configs/global.yaml", help="Path to global config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to EMA checkpoint .pt file")
    parser.add_argument("--output", type=str, default="outputs/generated.png", help="Output image path")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="CFG Extrapolation scale")
    args = parser.parse_args()

    config = load_config(args.global_config, args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float16 if device.type == 'cuda' else torch.float32

    # 1. Model Initialization
    print("Loading TimeAgnosticDiT and FourierKARTLayer from checkpoint...")
    model = KARTFlowModel(config).to(device)
    
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print("Loading VAE (frozen)...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=weight_dtype).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    num_classes = config.get('cfg', {}).get('num_classes', 10)
    labels = torch.randint(0, num_classes, (args.num_samples,), device=device)

    with torch.no_grad():
        # 2. The 1-Step Generative Jump
        print(f"Sampling latent trajectory for {args.num_samples} images...")
        z_0 = torch.randn(args.num_samples, 4, 32, 32, device=device)
        
        # Extrapolate blueprint and integrate
        delta_x = model.generate_with_cfg(z_0, labels, args.guidance_scale, num_classes)
        
        latent_pred = z_0 + delta_x

        # 3. The Decoding Phase
        print("Un-scaling and decoding latents to RGB...")
        latent_pred = latent_pred / 0.18215
        
        # Pass to VAE decoder
        latent_pred = latent_pred.to(weight_dtype)
        image_tensor = vae.decode(latent_pred).sample
        
        # 4. Post-Processing
        image_tensor = (image_tensor + 1) / 2
        image_tensor = torch.clamp(image_tensor, 0, 1)
        
        dirpath = os.path.dirname(args.output)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        save_image(image_tensor, args.output, nrow=4)
        print(f"Success! Saved generated images to {args.output}")

if __name__ == '__main__':
    main()
