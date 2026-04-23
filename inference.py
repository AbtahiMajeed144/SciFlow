import torch
from utils import save_image_grid

def generate_euler(model, device, num_samples=64, num_steps=50, filename="outputs/sample.png"):
    model.eval()
    with torch.no_grad():
        dt = 1.0 / num_steps
        x_t = torch.randn(num_samples, 3, 32, 32, device=device)
        
        for step in range(num_steps):
            t = step / num_steps
            t_tensor = torch.full((num_samples, 1), t, device=device)
            v_pred = model(x_t, t_tensor)
            x_t = x_t + v_pred * dt
            
        save_image_grid(x_t, filename)
        print(f"Saved {filename}")
