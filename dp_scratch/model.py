"""Diffusion Policy: ConditionalUnet1D + ResNet-18 encoder for LIBERO."""

import math
import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import T5EncoderModel, T5Tokenizer


# ==================== Building Blocks ====================


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResBlock1D(nn.Module):
    """Residual block with FiLM conditioning (scale + bias)."""

    def __init__(self, in_ch, out_ch, cond_dim, kernel_size=5, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_ch, out_ch, kernel_size, n_groups),
            Conv1dBlock(out_ch, out_ch, kernel_size, n_groups),
        ])
        self.cond_proj = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch * 2),
        )
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.blocks[0](x)
        embed = self.cond_proj(cond).unsqueeze(-1)
        scale, bias = embed.chunk(2, dim=1)
        h = scale * h + bias
        h = self.blocks[1](h)
        return h + self.residual(x)


# ==================== ConditionalUnet1D ====================


class ConditionalUnet1D(nn.Module):
    """1D U-Net for denoising action trajectories, conditioned on observations via FiLM.

    Architecture:
        - Encoder: n_levels of (2x ConditionalResBlock + Downsample)
        - Mid: 2x ConditionalResBlock at bottleneck
        - Decoder: n_levels of (concat skip + 2x ConditionalResBlock + Upsample)
        - Final: Conv1dBlock + Conv1d projection
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()

        all_dims = [input_dim] + list(down_dims)
        n_levels = len(down_dims)
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        # Timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # Encoder
        self.down_blocks = nn.ModuleList()
        for i in range(n_levels):
            is_last = i == n_levels - 1
            self.down_blocks.append(nn.ModuleList([
                ConditionalResBlock1D(all_dims[i], all_dims[i + 1], cond_dim, kernel_size, n_groups),
                ConditionalResBlock1D(all_dims[i + 1], all_dims[i + 1], cond_dim, kernel_size, n_groups),
                Downsample1d(all_dims[i + 1]) if not is_last else nn.Identity(),
            ]))

        # Mid
        mid_dim = down_dims[-1]
        self.mid_blocks = nn.ModuleList([
            ConditionalResBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ConditionalResBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
        ])

        # Decoder (symmetric: concat skip → resblocks → upsample)
        self.up_blocks = nn.ModuleList()
        prev_dim = mid_dim
        for i in reversed(range(n_levels)):
            skip_dim = all_dims[i + 1]
            out_dim = all_dims[i + 1]
            has_upsample = i > 0
            self.up_blocks.append(nn.ModuleList([
                ConditionalResBlock1D(prev_dim + skip_dim, out_dim, cond_dim, kernel_size, n_groups),
                ConditionalResBlock1D(out_dim, out_dim, cond_dim, kernel_size, n_groups),
                Upsample1d(out_dim) if has_upsample else nn.Identity(),
            ]))
            prev_dim = out_dim

        # Final projection
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(self, x, timestep, global_cond):
        """
        Args:
            x: (B, T, input_dim) noisy action trajectory
            timestep: (B,) diffusion timestep
            global_cond: (B, global_cond_dim) observation features
        Returns:
            (B, T, input_dim) predicted noise
        """
        x = x.permute(0, 2, 1)  # (B, input_dim, T)
        cond = torch.cat([self.time_emb(timestep), global_cond], dim=-1)

        # Encoder
        skips = []
        h = x
        for res1, res2, down in self.down_blocks:
            h = res1(h, cond)
            h = res2(h, cond)
            skips.append(h)
            h = down(h)

        # Mid
        for mid in self.mid_blocks:
            h = mid(h, cond)

        # Decoder
        for res1, res2, up in self.up_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = res1(h, cond)
            h = res2(h, cond)
            h = up(h)

        h = self.final_conv(h)
        return h.permute(0, 2, 1)


# ==================== Vision Encoder ====================


def _replace_bn_with_gn(module):
    """Replace all BatchNorm2d with GroupNorm (needed for EMA compatibility)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            n_ch = child.num_features
            setattr(module, name, nn.GroupNorm(max(1, n_ch // 16), n_ch))
        else:
            _replace_bn_with_gn(child)


class ResNet18Encoder(nn.Module):
    """ResNet-18 visual encoder outputting 512-dim feature per image."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Identity()
        _replace_bn_with_gn(net)
        self.net = net
        self.feature_dim = 512

    def forward(self, x):
        """x: (B, 3, H, W) ImageNet-normalized. Returns: (B, 512)."""
        return self.net(x)


# ==================== Cosine Noise Schedule ====================


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


# ==================== Full Diffusion Policy ====================


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy for LIBERO.

    Observation: multi-camera images + proprioceptive state + task embedding.
    Action: DDPM training on action chunks, DDIM inference.
    """

    def __init__(
        self,
        task_descs: List[str],
        action_dim: int = 7,
        state_dim: int = 8,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_cameras: int = 2,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        down_dims: tuple = (256, 512, 1024),
        diffusion_step_embed_dim: int = 128,
        task_embed_dim: int = 32,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_cameras = n_cameras
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.task_descs = list(task_descs)

        # Vision encoder (shared across cameras and timesteps)
        self.vision_enc = ResNet18Encoder(pretrained=True)
        img_feat_dim = self.vision_enc.feature_dim

        # Text conditioning: T5 encodes task descriptions
        self._init_text_encoder(task_descs, task_embed_dim)

        # Global conditioning:
        #   image features:  n_obs_steps * n_cameras * 512
        #   state features:  n_obs_steps * state_dim
        #   text embedding:  task_embed_dim
        global_cond_dim = n_obs_steps * n_cameras * img_feat_dim + n_obs_steps * state_dim + task_embed_dim

        # Noise prediction network
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
        )

        # Noise schedule
        betas = cosine_beta_schedule(num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Normalization stats (set via set_norm_stats)
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))

    def _init_text_encoder(self, task_descs: List[str], task_embed_dim: int):
        """Encode task descriptions with T5, precompute and cache."""
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        t5 = T5EncoderModel.from_pretrained("t5-small")
        t5.eval()
        embs = []
        with torch.no_grad():
            for desc in task_descs:
                inp = tokenizer(desc, return_tensors="pt", padding=True, truncation=True, max_length=64)
                out = t5(**inp).last_hidden_state
                emb = out.mean(dim=1).squeeze(0)
                embs.append(emb)
        text_feat_dim = embs[0].shape[0]
        self.text_embeddings = nn.Parameter(torch.stack(embs).float(), requires_grad=True)
        self.text_proj = nn.Linear(text_feat_dim, task_embed_dim)
        del tokenizer, t5

    # ---------- normalization helpers ----------

    def set_norm_stats(self, action_mean, action_std, state_mean, state_std):
        self.action_mean.copy_(torch.as_tensor(action_mean, dtype=torch.float32))
        self.action_std.copy_(torch.as_tensor(action_std, dtype=torch.float32))
        self.state_mean.copy_(torch.as_tensor(state_mean, dtype=torch.float32))
        self.state_std.copy_(torch.as_tensor(state_std, dtype=torch.float32))

    def normalize_action(self, action):
        return (action - self.action_mean) / self.action_std.clamp(min=1e-6)

    def unnormalize_action(self, action):
        return action * self.action_std + self.action_mean

    def normalize_state(self, state):
        return (state - self.state_mean) / self.state_std.clamp(min=1e-6)

    # ---------- observation encoding ----------

    def encode_obs(self, images, state, task_idx):
        """
        Args:
            images: (B, n_obs_steps, n_cameras, 3, H, W) float [0,1]
            state:  (B, n_obs_steps, state_dim)
            task_idx: (B,) long
        Returns:
            (B, global_cond_dim)
        """
        B = images.shape[0]

        # Flatten all images → batch through ResNet
        imgs = images.reshape(B * self.n_obs_steps * self.n_cameras, 3, images.shape[-2], images.shape[-1])
        mean = IMAGENET_MEAN.to(imgs.device)
        std = IMAGENET_STD.to(imgs.device)
        imgs = (imgs - mean) / std
        img_features = self.vision_enc(imgs)                  # (B*T*C, 512)
        img_features = img_features.reshape(B, -1)            # (B, T*C*512)

        state_norm = self.normalize_state(state).reshape(B, -1)  # (B, T*state_dim)
        text_emb = self.text_proj(self.text_embeddings[task_idx])  # (B, task_embed_dim)

        return torch.cat([img_features, state_norm, text_emb], dim=-1)

    # ---------- training ----------

    def compute_loss(self, images, state, task_idx, actions):
        """
        Standard DDPM ε-prediction loss.

        Args:
            images:   (B, n_obs_steps, n_cameras, 3, H, W)
            state:    (B, n_obs_steps, state_dim)
            task_idx: (B,) long
            actions:  (B, horizon, action_dim)
        Returns:
            scalar MSE loss
        """
        B = actions.shape[0]
        actions_norm = self.normalize_action(actions)
        global_cond = self.encode_obs(images, state, task_idx)

        noise = torch.randn_like(actions_norm)
        timesteps = torch.randint(0, self.num_train_timesteps, (B,), device=actions.device)

        sqrt_a = self.sqrt_alphas_cumprod[timesteps][:, None, None]
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None]
        noisy_actions = sqrt_a * actions_norm + sqrt_1ma * noise

        noise_pred = self.noise_pred_net(noisy_actions, timesteps.float(), global_cond)
        return F.mse_loss(noise_pred, noise)

    # ---------- inference ----------

    @torch.no_grad()
    def predict_action(self, images, state, task_idx):
        """
        DDIM sampling (deterministic, eta=0).

        Returns: (B, horizon, action_dim) un-normalized actions.
        """
        B = images.shape[0]
        device = images.device
        global_cond = self.encode_obs(images, state, task_idx)

        # Start from pure noise
        x = torch.randn(B, self.horizon, self.action_dim, device=device)

        # DDIM timestep schedule (evenly spaced, high → low)
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, self.num_inference_steps) * step_ratio).round().astype(np.int64)
        timesteps = np.flip(timesteps).copy()

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.noise_pred_net(x, t_batch.float(), global_cond)

            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)

            # Predict x_0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x0_pred = x0_pred.clamp(-10, 10)

            # DDIM deterministic update
            x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * noise_pred

        return self.unnormalize_action(x)

    # ---------- eval-compatible interface ----------

    @torch.no_grad()
    def infer(self, obs_dict):
        """
        Interface compatible with LIBERO evaluation.

        Args:
            obs_dict: {
                "observation/image":       (H, W, 3) uint8,
                "observation/wrist_image": (H, W, 3) uint8,
                "observation/state":       (8,) float32,
                "prompt":                  str,
            }
        Returns:
            {"actions": (horizon, action_dim) ndarray}
        """
        device = next(self.parameters()).device

        img = _to_tensor_img(obs_dict["observation/image"]).to(device)
        wrist = _to_tensor_img(obs_dict["observation/wrist_image"]).to(device)
        imgs_single = torch.stack([img, wrist], dim=0)                                # (2, 3, H, W)
        images = imgs_single.unsqueeze(0).unsqueeze(0).expand(1, self.n_obs_steps, -1, -1, -1, -1)  # (1, T, 2, 3, H, W)

        state = torch.as_tensor(obs_dict["observation/state"], dtype=torch.float32, device=device)
        state = state.unsqueeze(0).unsqueeze(0).expand(1, self.n_obs_steps, -1)      # (1, T, 8)

        task_idx = self._prompt_to_idx(obs_dict.get("prompt", ""))
        task_idx = torch.tensor([task_idx], dtype=torch.long, device=device)

        actions = self.predict_action(images, state, task_idx)
        return {"actions": actions[0].cpu().numpy()}

    def _prompt_to_idx(self, prompt: str) -> int:
        prompt_l = prompt.strip().lower()
        for i, d in enumerate(self.task_descs):
            if d.strip().lower() == prompt_l:
                return i
        for i, d in enumerate(self.task_descs):
            if prompt_l in d.lower() or d.lower() in prompt_l:
                return i
        return 0


# ---------- helpers ----------


def _to_tensor_img(img):
    """Convert uint8 (H, W, 3) → float (3, 224, 224) in [0,1]."""
    import cv2
    img = np.asarray(img)
    if img.shape[:2] != (224, 224):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 255.0


def create_ema(model: nn.Module) -> nn.Module:
    """Create an EMA copy of the model."""
    return copy.deepcopy(model)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.995):
    """Exponential moving average update for both parameters and buffers."""
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)
