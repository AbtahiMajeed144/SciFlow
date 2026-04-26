import os
import shutil
import torch
from torchvision.utils import save_image
from tqdm import tqdm


def evaluate_model(model, vae, device, config, out_dir="outputs"):
    """
    Evaluates the model by generating a large batch of images and computing
    FID, sFID, Precision, and Recall using torch-fidelity and clean-fid.
    """
    eval_tmp_dir = os.path.join(out_dir, "eval_tmp")
    os.makedirs(eval_tmp_dir, exist_ok=True)
    
    # Check dependencies
    try:
        import torch_fidelity
        has_fidelity = True
    except ImportError:
        print("Warning: torch_fidelity is not installed. Skipping FID/Precision/Recall.")
        has_fidelity = False
        
    try:
        import cleanfid.fid as clean_fid
        has_clean_fid = True
    except ImportError:
        print("Warning: clean-fid is not installed. Skipping alternative FID/sFID.")
        has_clean_fid = False

    if not has_fidelity and not has_clean_fid:
        return {}

    num_samples = config['training'].get('eval_samples', 10000)
    batch_size = config['training'].get('eval_batch_size', 64)
    ref_stat_path = config['training'].get('ref_stat_path', 'path/to/imagenet/val')
    num_classes = config.get('cfg', {}).get('num_classes', 10)
    guidance_scale = config.get('cfg', {}).get('guidance_scale', 3.0)
    
    print(f"\n[EVALUATION] Generating {num_samples} samples to {eval_tmp_dir}...")
    model.eval()
    if vae is not None:
        vae.eval()
        weight_dtype = next(vae.parameters()).dtype
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    img_idx = 0
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Generating Eval Batches"):
            current_batch_size = min(batch_size, num_samples - img_idx)
            
            # Sample latent noise
            z_0 = torch.randn(current_batch_size, model.in_channels, model.img_size, model.img_size, device=device)
            labels = torch.randint(0, num_classes, (current_batch_size,), device=device)
            
            # Extrapolate blueprint and integrate
            h_guided = model.generate_with_cfg(z_0, labels, guidance_scale, num_classes)
            x_final = z_0 + h_guided
            
            # VAE Decoding if necessary
            if model.in_channels == 4 and vae is not None:
                latent_pred = x_final / 0.18215
                latent_pred = latent_pred.to(weight_dtype)
                image_tensor = vae.decode(latent_pred).sample
                x_final = image_tensor.to(torch.float32)
                
            # Normalize to [0, 1] for saving
            x_final = (x_final + 1) / 2
            x_final = torch.clamp(x_final, 0, 1)
            
            # Save individual images for torch-fidelity
            for j in range(current_batch_size):
                save_image(x_final[j], os.path.join(eval_tmp_dir, f"{img_idx}.png"))
                img_idx += 1
                
    results = {}
    
    # 1. torch-fidelity (FID, Precision, Recall)
    if has_fidelity:
        print("[EVALUATION] Computing metrics via torch-fidelity...")
        try:
            metrics = torch_fidelity.calculate_metrics(
                input1=eval_tmp_dir,
                input2=ref_stat_path,
                cuda=True,
                isc=False,
                fid=True,
                kid=False,
                prc=True,
                verbose=False
            )
            results['FID'] = metrics.get('frechet_inception_distance', 0.0)
            results['Precision'] = metrics.get('precision', 0.0)
            results['Recall'] = metrics.get('recall', 0.0)
        except Exception as e:
            print(f"torch-fidelity error: {e}")

    # 2. clean-fid (sFID and standard FID fallback)
    if has_clean_fid:
        print("[EVALUATION] Computing metrics via clean-fid...")
        try:
            # Standard Clean FID
            clean_fid_score = clean_fid.compute_fid(eval_tmp_dir, ref_stat_path, mode="clean")
            results['Clean_FID'] = clean_fid_score
            
            # Spatial FID (sFID) — uses spatial features from the Inception network
            sfid_score = clean_fid.compute_fid(eval_tmp_dir, ref_stat_path, mode="clean", use_dataparallel=False)
            results['sFID'] = sfid_score
            
        except Exception as e:
            print(f"clean-fid error: {e}")

    # Cleanup temporary images
    print("[EVALUATION] Cleaning up temporary files...")
    shutil.rmtree(eval_tmp_dir)
    
    return results
