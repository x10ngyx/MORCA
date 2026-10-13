# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Adapted for MORCA; see licenses/NOTICE.txt.
import numpy as np
import torch
from wan.modules.model import sinusoidal_embedding_1d

ratios = [
    1.00124,
    1.00155,
    0.99822,
    0.99851,
    0.99696,
    0.99687,
    0.99703,
    0.99732,
    0.9966,
    0.99679,
    0.99602,
    0.99658,
    0.99578,
    0.99664,
    0.99484,
    0.9949,
    0.99633,
    0.996,
    0.99659,
    0.99683,
    0.99534,
    0.99549,
    0.99584,
    0.99577,
    0.99681,
    0.99694,
    0.99563,
    0.99554,
    0.9944,
    0.99473,
    0.99594,
    0.9964,
    0.99466,
    0.99461,
    0.99453,
    0.99481,
    0.99389,
    0.99365,
    0.99391,
    0.99406,
    0.99354,
    0.99361,
    0.99283,
    0.99278,
    0.99268,
    0.99263,
    0.99057,
    0.99091,
    0.99125,
    0.99126,
    0.65523,
    0.65252,
    0.98808,
    0.98852,
    0.98765,
    0.98736,
    0.9851,
    0.98535,
    0.98311,
    0.98339,
    0.9805,
    0.9806,
    0.97776,
    0.97771,
    0.97278,
    0.97286,
    0.96731,
    0.96728,
    0.95857,
    0.95855,
    0.94385,
    0.94385,
    0.92118,
    0.921,
    0.88108,
    0.88076,
    0.80263,
    0.80181,
]


def nearest_interp(src_array, target_length):
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])
    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]


def magcache_forward(self, x, t, context, seq_len, clip_fea=None, y=None):
    """
    Forward pass through the diffusion model
    Args:
        x (List[Tensor]):
            List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x
    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    if self.model_type == "i2v":
        assert y is not None
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)
    if y is not None:
        x = [torch.cat([u, v], dim=0) for (u, v) in zip(x, y)]
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat(
        [torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x]
    )
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).float()
        )
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32
    context_lens = None
    context = self.text_embedding(
        torch.stack(
            [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
        )
    )
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
    )
    use_magcache = True
    skip_forward = False
    ori_x = x
    if self.split_step is not None:
        if self.mode == "i2v":
            if self.cnt < int(
                self.split_step + (self.num_steps - self.split_step) * self.retention_ratio
            ):
                use_magcache = False
        elif self.cnt < int(self.split_step * self.retention_ratio) or (
            self.cnt <= (self.num_steps - self.split_step) * self.retention_ratio + self.split_step
            and self.cnt >= self.split_step
        ):
            use_magcache = False
    elif self.cnt < int(self.num_steps * self.retention_ratio):
        use_magcache = False
    if use_magcache:
        cur_mag_ratio = self.mag_ratios[self.cnt]
        self.accumulated_ratio[self.cnt % 2] = self.accumulated_ratio[self.cnt % 2] * cur_mag_ratio
        self.accumulated_steps[self.cnt % 2] += 1
        cur_skip_err = np.abs(1 - self.accumulated_ratio[self.cnt % 2])
        self.accumulated_err[self.cnt % 2] += cur_skip_err
        if (
            self.accumulated_err[self.cnt % 2] < self.magcache_thresh
            and self.accumulated_steps[self.cnt % 2] <= self.K
        ):
            skip_forward = True
            residual_x = self.residual_cache[self.cnt % 2]
        else:
            self.accumulated_err[self.cnt % 2] = 0
            self.accumulated_steps[self.cnt % 2] = 0
            self.accumulated_ratio[self.cnt % 2] = 1.0
    if skip_forward:
        x = x + residual_x
    else:
        for ind, block in enumerate(self.blocks):
            x = block(x, **kwargs)
        residual_x = x - ori_x
    self.residual_cache[self.cnt % 2] = residual_x
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0
        self.accumulated_ratio = [1.0, 1.0]
        self.accumulated_err = [0.0, 0.0]
        self.accumulated_steps = [0, 0]
    return [u.float() for u in x]


def init_magcache(model, mag_ratios, args, split_steps=None, mode="t2v"):
    model.__class__.forward = magcache_forward
    model.__class__.cnt = torch.tensor(0)
    model.__class__.num_steps = args.sample_steps * 2
    model.__class__.split_step = split_steps * 2
    model.__class__.mode = mode
    model.__class__.magcache_thresh = args.magcache_thresh
    model.__class__.K = args.magcache_K
    model.__class__.accumulated_err = [0.0, 0.0]
    model.__class__.accumulated_steps = [0, 0]
    model.__class__.accumulated_ratio = [1.0, 1.0]
    model.__class__.retention_ratio = args.retention_ratio
    model.__class__.residual_cache = [None, None]
    model.__class__.mag_ratios = np.array([1.0] * 2 + mag_ratios)
    if len(model.__class__.mag_ratios) != args.sample_steps * 2:
        mag_ratio_con = nearest_interp(model.__class__.mag_ratios[0::2], args.sample_steps)
        mag_ratio_ucon = nearest_interp(model.__class__.mag_ratios[1::2], args.sample_steps)
        interpolated_mag_ratios = np.concatenate(
            [mag_ratio_con.reshape(-1, 1), mag_ratio_ucon.reshape(-1, 1)], axis=1
        ).reshape(-1)
        model.__class__.mag_ratios = interpolated_mag_ratios
