import torch
from utils import save_image_grid

def generate_1step(model, device, num_samples=64, filename="outputs/sample.png"):
    model.eval()
    with torch.no_grad():
        x_0 = torch.randn(num_samples, 3, 32, 32, device=device)
        h_1 = model.integrate_1step(x_0)
        x_final = x_0 + h_1
        save_image_grid(x_final, filename)
        print(f"Saved {filename}")
