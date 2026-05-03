## Pairing

We need to create an offline data preparation script (prepare_latents.py) to generate a static, paired dataset for SciFlow using an Asymmetric 1D Sliced Optimal Transport algorithm.

We are mapping a densely sampled noise distribution to a sparser data distribution to create 'basins of attraction'. To avoid memory explosion, we will use index-mapping instead of duplicating data tensors.

Please implement the following specifications precisely:

1. Configuration Updates:

Add a noise_multiplier integer to the config file (default: 4).

2. The prepare_latents.py Script:

Load Data: Load the full target dataset of latents into a single tensor Z_data of shape [N, C, H, W].

Generate Noise: Generate standard Gaussian noise Z_noise of shape [M, C, H, W], where M = N * config.noise_multiplier.

3. The Asymmetric 1D Sliced OT Algorithm (Vectorized):

Flatten both tensors to 2D: flat_data [N, D] and flat_noise [M, D] (where D = C*H*W).

Generate a single random projection vector v of shape [D, 1]. Normalize it: v = v / torch.norm(v).

Project both distributions onto the 1D line:
P_data = flat_data @ v (shape [N, 1])
P_noise = flat_noise @ v (shape [M, 1])

The Duplication Trick (In-Memory): >   Create an index array for the data: data_idx = torch.arange(N).
Repeat both the data projections and the indices by the multiplier:
P_data_expanded = P_data.repeat_interleave(config.noise_multiplier) (shape [M, 1])
data_idx_expanded = data_idx.repeat_interleave(config.noise_multiplier) (shape [M])

The Sort:
Sort P_noise to get sort_indices_noise.
Sort P_data_expanded to get sort_indices_data.

The Mapping:
Create an empty integer tensor pairing_map = torch.zeros(M, dtype=torch.long).
Map the indices:
pairing_map[sort_indices_noise] = data_idx_expanded[sort_indices_data]

4. Data Storage Strategy:

Do NOT save expanded latents. Save three distinct PyTorch .pt files to disk:

latents.pt: The original Z_data tensor [N, C, H, W].

noise.pt: The full Z_noise tensor [M, C, H, W].

pairing_map.pt: The 1D integer mapping array [M].

5. Dataloader Update (dataset.py):

Update the Dataset class to load these three files.

__len__ should return M (the size of the noise).

__getitem__(self, idx) should return:
z_0 = self.noise[idx]
target_idx = self.pairing_map[idx]
x_1 = self.latents[target_idx]
return z_0, x_1

Please write prepare_latents.py and update dataset.py according to this strict logic.


## VAE

We are preparing the final, static Lagrangian dataset for the SciFlow ImageNet $256 \times 256$ experiment. We need a standalone data preparation script (prepare_latents.py) that strictly follows an Offline Asymmetric 1D Sliced Optimal Transport pipeline.Please implement prepare_latents.py with the following strict specifications:1. The VAE Compression Phase:Initialize the stabilityai/sd-vae-ft-mse model from the diffusers library. Put it in eval() mode and freeze all gradients.Load the ImageNet $256 \times 256$ dataset.Iterate through the dataset in safe batches (e.g., batch_size=128) to prevent VRAM OOM errors.For each batch, pass the images through the VAE encoder. CRITICAL: Multiply the resulting latents by the exact scaling factor 0.18215.Collect all 1.2 million encoded latents into a single CPU CPU tensor Z_data of shape [N, 4, 32, 32] (where $N \approx 1.2M$).2. The Asymmetric Noise Generation Phase:Define a noise_multiplier = 4.Let $M = N \times 4$. Generate a single standard Gaussian noise tensor Z_noise of shape [M, 4, 32, 32] using torch.randn.3. The 1D Sliced OT Mapping Phase:Flatten Z_data to [N, 4096] and Z_noise to [M, 4096].Generate a random, normalized 1D projection vector v of shape [4096, 1].Project the data: P_data = Z_data_flat @ v.Project the noise: P_noise = Z_noise_flat @ v.Create data indices: data_idx = torch.arange(N).Expand the data projections and indices to match the noise size to create 'basins of attraction':P_data_expanded = P_data.repeat_interleave(noise_multiplier)data_idx_expanded = data_idx.repeat_interleave(noise_multiplier)Sort both 1D projections:sort_indices_noise = torch.argsort(P_noise.squeeze())sort_indices_data = torch.argsort(P_data_expanded.squeeze())Create the 1D mapping array:pairing_map = torch.zeros(M, dtype=torch.long)pairing_map[sort_indices_noise] = data_idx_expanded[sort_indices_data]4. Data Storage and Dataloader Update:Save three files to disk: latents.pt (Z_data), noise.pt (Z_noise), and pairing_map.pt (pairing_map).Update dataset.py to load these three files into system RAM.The __len__ of the dataset is now $M$.The __getitem__(self, idx) must return the exact pair: (self.noise[idx], self.latents[self.pairing_map[idx]])

## Encoder Setup

We are building Module M (the 'Blueprint Generator') for our SciFlow ImageNet experiment. We need to implement a TimeAgnosticDiT that automatically scales its architecture based on a simple string in the configuration file.Please create or update model.py with the following strict specifications:1. Configuration Handling:The model initialization should accept a string argument model_size (expecting 'small', 'base', 'large', or 'xl').Implement an internal dictionary to automatically set the hyperparameters based on this string:'small': hidden_size=384, depth=12, heads=6'base': hidden_size=768, depth=12, heads=12'large': hidden_size=1024, depth=24, heads=16'xl': hidden_size=1152, depth=28, heads=162. The Time-Agnostic Architecture (TimeAgnosticDiT):Input Patchification: The network must accept a latent space of in_channels=4 and a spatial size of 32x32. Use a patch_size=2. This results in a sequence of 256 tokens.Time Severance (CRITICAL): Remove all concepts of timestep $t$ from this transformer. There should be no timestep embedding layer.Class Embedding (CFG Support): The class embedding layer must support the null token for Classifier-Free Guidance. Initialize it as self.class_emb = nn.Embedding(num_classes + 1, hidden_size).AdaLN Conditioning: The AdaLN (Adaptive Layer Normalization) mechanism in every transformer block must modulate the scale and shift parameters using only the class_emb.3. The Output State:In a standard DiT, the final block is followed by a linear projection back to the pixel/latent dimension to predict velocity. Do not do this.The output of TimeAgnosticDiT should simply be the final normalized hidden states of the transformer tokens.The output shape must strictly be [Batch, 256, hidden_size]. This tensor acts as the spatial blueprint $X$ that will be passed into the FourierKARTLayer.Please write the TimeAgnosticDiT class implementing this exact routing and scaling logic.



## Time modulator setup

We need to update Module K (FourierKARTLayer) in model.py and the optimizer in train.py to handle the dense VAE latent space for the ImageNet scaling run.Please implement the following specific upgrades:1. Module K Initialization (model.py):Update the default parameters for the latent space: Q=128, K=6.The input to the KART layer (the Blueprint from Module M) is now exactly the hidden size of the DiT (e.g., 768 for Base).The Damped Basis friction parameter must now be initialized slightly positive to act as an early training safety net:self.gamma = nn.Parameter(torch.ones(Q) * 0.1)2. Optimizer Parameter Grouping (train.py):When initializing the AdamW optimizer, you must explicitly separate the parameters into two groups to handle weight decay correctly.Group 1 (Standard Weights): Apply the default weight_decay (e.g., 0.01) to all weights in the DiT and the amplitude/phase parameters ($A$, $B$, $C_q$) of the KART layer.Group 2 (Physics Parameters): Apply strictly 0.0 weight decay to the frequency (w_q) and friction (gamma) parameters inside the FourierKARTLayer. This prevents the optimizer from artificially killing the fluid physics.Please update the codebase with these parameter scaling and optimization rules.

## Objective setup

We are upgrading the SciFlow training loop in train.py to utilize a 'Dual KART Loss' that leverages our analytical integral for trajectory consistency without introducing MeanFlow optimization conflicts.Please implement the following loss calculation inside the training step:1. Data Setup:The dataloader yields z_0 (noise) and x_1 (target data latent).target_velocity = x_1 - z_0.2. The Forward Pass:Sample a random timestep t uniformly between 0 and 1 for the batch.Pass z_0 through Module M (TimeAgnosticDiT) to get the spatial blueprint.3. Term 1: Instantaneous Velocity ($L_{vel}$)Pass blueprint and t into Module K to get the instantaneous velocity:v_pred = kart_layer(blueprint, t)loss_vel = F.mse_loss(v_pred, target_velocity)4. Term 2: Analytical Endpoint ($L_{end}$)Call the analytical integral method on Module K from t=0 to t=1:delta_x = kart_layer.integrate_1step(blueprint)Calculate the predicted final state: x_pred = z_0 + delta_xloss_end = F.mse_loss(x_pred, x_1)5. The Total Objective:Combine the losses. We want to heavily weight the endpoint to force the soft landing.total_loss = loss_vel + (2.0 * loss_end)total_loss.backward()Please update the training loop with this mathematically aligned dual objective. Instead of fixed 2.0 weight, the weight should be controlled from config.

## Decoder Setup

We are building the 1-step inference script (generate.py) for the SciFlow architecture. We need to generate a latent trajectory using our Dual KART-DiT, un-scale it, and decode it back to a $256 \times 256$ RGB image using a frozen VAE.Please implement generate.py with the following specifications:1. Model Initialization:Load the pre-trained TimeAgnosticDiT and FourierKARTLayer from the saved checkpoint. Put them in eval() mode.Load the AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse") from the diffusers library. Put it in eval() mode, freeze parameters, and cast to the appropriate dtype (e.g., float16 if necessary for VRAM).2. The 1-Step Generative Jump:Sample a random standard Gaussian noise tensor z_0 of shape [Batch_Size, 4, 32, 32].Pass z_0 through the DiT to get the blueprint.CRITICAL: Do NOT use an ODE solver. Call the analytical integral method on the KART layer to calculate the exact displacement from $t=0$ to $t=1$:delta_x = kart_layer.integrate_1step(blueprint)Calculate the final predicted latent: latent_pred = z_0 + delta_x3. The Decoding Phase:Un-Scale the Latent: Divide the prediction by the strict scaling factor before decoding:latent_pred = latent_pred / 0.18215Pass the un-scaled latent through the VAE decoder:image_tensor = vae.decode(latent_pred).sample4. Post-Processing:The VAE outputs values roughly in the range [-1, 1].Clamp the output, normalize it to [0, 1], and convert it to a standard PIL Image (or save directly via torchvision.utils.save_image).Please ensure the math for the 1-step jump and the un-scaling is strictly followed.


## Scores

We are upgrading the evaluation pipeline in train.py to match the rigorous standards of modern 1-step generative models.

Please implement the following evaluation metrics using the torch-fidelity library (or similar standard metric suite):

1. FID and sFID Integration:

During the evaluation loop (e.g., every 50 epochs), generate a batch of 10,000 images using our 1-step analytical integral.

Compute both standard FID and sFID against the pre-calculated statistics of the ImageNet validation set.

2. Precision and Recall:

Along with FID, compute and log the Precision and Recall metrics (often provided by the same library calculating FID via Manifold metrics).

3. Logging:

Ensure all metrics (Train Loss_vel, Train Loss_end, FID, sFID, Precision, Recall) are explicitly logged to logs.txt under specific experiment.

## Exponential Moving Average (EMA):
You must implement a PyTorch EMA hook (typically with a decay rate of 0.9999). During training, you update the EMA weights silently in the background. When you run generate.py or calculate your FID scores, you strictly use the EMA checkpoint, not the active training weights.

## CFG training probability
We built the TimeAgnosticDiT to accept a null token for Classifier-Free Guidance (CFG), but how exactly do we condition the fluid during training?The SOTA Standard: * Training: You must randomly drop the ImageNet class label and replace it with the null_class exactly 10% of the time (p_uncond = 0.1). This forces the KART oscillators to learn how to route the macroscopic fluid without any semantic hints, relying purely on the geometry of $z_0$.Inference: For your final paper results, you will generate the images using the extrapolation formula: $Blueprint = \emptyset + w \cdot (Label - \emptyset)$. For 1-step models, a CFG scale ($w$) of 1.5 to 3.0 is the sweet spot.