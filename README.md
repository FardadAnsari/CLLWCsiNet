Data Flow Summary of CLLWCsiNet
This document explains the input-to-output tensor transformations in the CLLWCsiNet model, designed for CSI feedback compression in wireless communication systems.

Input Tensor
Shape: (batch_size, 2, 32, 32)

Description:

Represents a 2-channel (real + imaginary) CSI matrix of size 32×32.

Step-by-Step Data Flow
Stage	Tensor Shape	Key Operations
1. Input	(B, 2, 32, 32)	Raw CSI matrix (real + imaginary parts).
2. Encoder Paths	(B, 2, 32, 32) ×3	Three parallel branches with [1×7], [1×5], and [1×3] asymmetric kernels.
3. Feature Fusion	(B, 6, 32, 32) → (B, 2, 32, 32)	Concatenate → Conv1x1 reduces channels back to 2.
4. Reshape for Compression	(B, 64, 1, 32)	Flatten spatial dimensions for latent encoding.
5. Latent Compression	(B, 4, 1, 32)	Encoder reduces 64 → 4 channels (compressed latent space).
6. Noisy Feedback Simulation	(B, 4, 1, 32)	Adds Gaussian noise (SNR=40dB) to simulate real-world channel conditions.
7. Decoder Expansion	(B, 64, 1, 32)	Expands latent 4 → 64 channels for reconstruction.
8. Residual Denoising	(B, 64, 1, 32)	Subtracts noise estimate (x = x - y) for cleaner reconstruction.
9. Reshape Back	(B, 2, 32, 32)	Restores original spatial dimensions.
10. Refinement	(B, 2, 32, 32) ×2	Two RefineNet blocks enhance features via multi-scale fusion & residual learning.
11. Final Output	(B, 2, 32, 32)	Sigmoid() activation ensures output in [0, 1] range.
Key Features
✅ Multi-Scale Processing: Parallel encoder paths capture diverse CSI features.
✅ Noise Robustness: Simulates real-world noisy feedback with adjustable SNR.
✅ Residual Learning: Skip connections in RefineNet stabilize training.
✅ Efficient Compression: Latent space (4 channels) minimizes feedback overhead.