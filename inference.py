import torch
from utils import save_image_grid

def generate_1step(model, device, num_samples=64, filename="outputs/sample.png"):
    model.eval()
    with torch.no_grad():
        x0 = torch.randn(num_samples, 3, 32, 32, device=device)
        # Analytical 1-step integration
        h_1 = model.integrate_1step(x0)
        # Kinematic addition
        x_final = x0 + h_1
        
        save_image_grid(x_final, filename)
        print(f"Saved {filename}")
