import torch
from xformers.components.attention import ScaledDotProduct

attention = ScaledDotProduct().cuda()

# FW a random bunch of data
inputs = torch.rand((16, 1024, 1024), device=torch.device("cuda"))

# Not a very sparse mask to begin with
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

mask = (torch.rand((1024, 1024)) < 0.9).cuda()
att = attention(q=inputs, k=inputs, v=inputs, mask=mask)

torch.cuda.synchronize()
max_memory = torch.cuda.max_memory_allocated() // 2 ** 20
print(f"Dense - Peak memory use: {max_memory}MB")

# Now use a very sparse mask and observe that memory use changes
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

import xformers.components.attention.attention_patterns as AP

# Couple of examples of attention patterns useful for vision
H, W = 20, 30  # assuming a 20 x 30 sequence length

# - axial attention
axial_pattern = AP.axial_2d_pattern(H, W)

# - local attention
loc_2d_dist = AP.local_2d_pattern(H, W, distance=5, p=2.0)  # distance and thresholds are user defined

# - random attention
gaus_2d_dist = AP.local_2d_gausian_distribution(H, W, sigma=2)
sparsity = 0.95
num_non_zeros = int((H * W) ** 2 * (1 - sparsity))
random_gaus_2d_pattern = AP.random_pattern_from_probability_matrix(gaus_2d_dist, num_non_zeros)

# - combining different patterns
combined_mask = axial_pattern | loc_2d_dist | random_gaus_2d_pattern

#mask = (torch.rand((1024, 1024)) < 0.01).cuda()
att = attention(q=inputs, k=inputs, v=inputs, mask=combined_mask)

torch.cuda.synchronize()
max_memory = torch.cuda.max_memory_allocated() // 2 ** 20
print(f"Sparse - Peak memory use: {max_memory}MB")
