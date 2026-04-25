import torch
from utils import save_image_grid

def generate_1step(model, device, num_samples=64, filename="outputs/sample.png", labels=None, guidance_scale=3.0, num_classes=10):
    model.eval()
    with torch.no_grad():
        x_0 = torch.randn(num_samples, 3, 32, 32, device=device)
        if labels is None:
            labels = torch.randint(0, num_classes, (num_samples,), device=device)
            
        # 1. Conditional vector
        h_cond = model.integrate_1step(x_0, labels)
        
        # 2. Unconditional vector
        null_labels = torch.full_like(labels, num_classes)
        h_uncond = model.integrate_1step(x_0, null_labels)
        
        # 3. Extrapolate
        h_guided = h_uncond + guidance_scale * (h_cond - h_uncond)
        
        # 4. Final 1-Step Jump
        x_final = x_0 + h_guided
        
        save_image_grid(x_final, filename)
        print(f"Saved {filename}")
