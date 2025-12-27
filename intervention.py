#!/usr/bin/env python3
"""
SAE intervention experiments for action prediction.

Supports:
- Feature ablation: set SAE feature i to 0
- Feature clamp: set SAE feature i to some value (e.g., mean)
- Feature addition: add alpha to SAE feature i
- Reconstruction ablation: replace h with SAE decode(encode(h)) (removes residual)

Measures:
- delta in action predictions (continuous) OR delta in logits (discrete), depending on your model API.

You MUST provide a ModelAdapter for your VLA that exposes:
- how to run a forward pass
- which module corresponds to "layer 11 residual stream" to hook
- where the action output lives

This file is intentionally model-agnostic.
"""

import os, json, glob, re, argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn

from overcomplete.sae import TopKSAE  # same SAE class you used

ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


# =========================
# Dataset indexing + actions loader (copied from your script, trimmed)
# =========================
libero_task_map = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_cream_cheese_in_the_bowl",
        "turn_on_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_wine_bottle_on_the_rack",
    ],
    "libero_10": [
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    ],
}

def parse_episode_id_from_actions_json(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def parse_episode_id_from_video(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def parse_episode_id_from_activation_npy(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def prompt_for_group_and_episode(group_name: str, episode_id: str) -> str:
    m = re.search(r"task(\d+)", episode_id)
    if m:
        idx = int(m.group(1))
        key = f"libero_{group_name}" if not group_name.startswith("libero_") else group_name
        if key in libero_task_map and 0 <= idx < len(libero_task_map[key]):
            return libero_task_map[key][idx]
    return f"{group_name}:unknown_prompt"

@dataclass
class Episode:
    group: str
    episode_id: str
    actions_path: str
    video_path: Optional[str]
    prompt: str
    act_path: Optional[str]  # activation npy path

def index_libero_dataset(data_root: str, activations_root: str, groups=("10","goal","object","spatial")):
    actions_map, video_map = {}, {}
    for g in groups:
        actions_dir = os.path.join(data_root, g, "actions")
        videos_dir  = os.path.join(data_root, g, "videos")
        for p in sorted(glob.glob(os.path.join(actions_dir, "*.json"))):
            actions_map[(g, parse_episode_id_from_actions_json(p))] = p
        for p in sorted(glob.glob(os.path.join(videos_dir, "*.mp4"))):
            video_map[(g, parse_episode_id_from_video(p))] = p

    act_map = {}
    for p in sorted(glob.glob(os.path.join(activations_root, "*.npy"))):
        act_map[parse_episode_id_from_activation_npy(p)] = p

    episodes = []
    for (g, eid_raw), a_path in actions_map.items():
        v_path = video_map.get((g, eid_raw.replace("actions", "rollout")), None)

        mnum = re.search(r"\d+", eid_raw)
        if mnum is None:
            continue
        num = int(mnum.group())

        eid = eid_raw.replace("actions_", "").split("_trial")[0]
        task = next((i for i, s in enumerate(libero_task_map[f"libero_{g}"]) if eid in s), -1)

        # your activation naming convention:
        act_path = act_map.get(f"task{task}_ep{num}_post_ffn_last_step", None)
        prompt = prompt_for_group_and_episode(g, eid)

        episodes.append(Episode(g, eid, a_path, v_path, prompt, act_path))

    print(f"Indexed {len(episodes)} episodes.")
    return episodes

def _is_num(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)

def _as_float_vec(x):
    if isinstance(x, np.ndarray) and x.ndim == 1 and np.issubdtype(x.dtype, np.number):
        return x.astype(np.float32)
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(_is_num(v) for v in x):
        return np.asarray(x, dtype=np.float32)
    return None

def _find_action_in_dict(d):
    candidate_keys = ["action","actions","robot_action","robot_actions","ctrl","control","command","ee_action","ee_delta","delta"]
    for k in candidate_keys:
        if k in d:
            v = d[k]
            vec = _as_float_vec(v)
            if vec is not None:
                return vec
            if isinstance(v, dict):
                for vv in v.values():
                    vec2 = _as_float_vec(vv)
                    if vec2 is not None:
                        return vec2
    for v in d.values():
        vec = _as_float_vec(v)
        if vec is not None:
            return vec
        if isinstance(v, dict):
            for vv in v.values():
                vec2 = _as_float_vec(vv)
                if vec2 is not None:
                    return vec2
    return None

def load_actions(actions_json_path: str) -> np.ndarray:
    with open(actions_json_path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        if "actions" in obj:
            obj = obj["actions"]
        else:
            raise ValueError(f"Dict JSON without 'actions' in {actions_json_path}")

    if isinstance(obj, list) and len(obj) > 0:
        if isinstance(obj[0], (list, tuple, np.ndarray)):
            acts = np.asarray(obj, dtype=np.float32)
            assert acts.ndim == 2
            return acts
        if isinstance(obj[0], dict):
            rows = []
            for step in obj:
                vec = _find_action_in_dict(step)
                if vec is None:
                    raise ValueError(f"Could not parse action in {actions_json_path}")
                rows.append(vec)
            dim0 = rows[0].shape[0]
            for v in rows:
                if v.shape[0] != dim0:
                    raise ValueError(f"Inconsistent action_dim in {actions_json_path}")
            return np.stack(rows, axis=0).astype(np.float32)
    return np.zeros((0, 0), dtype=np.float32)


# =========================
# SAE interventions
# =========================
class SAEIntervention:
    """
    Given a hidden vector h (B,d), performs:
      z = SAEEnc(h)
      z' = modify(z)
      h' = SAEDec(z')
    and returns h' (or optionally h + delta).
    """
    def __init__(
        self,
        sae: TopKSAE,
        mode: str,                 # "ablate_feature", "add_feature", "clamp_feature", "reconstruct"
        feature_idx: Optional[int] = None,
        alpha: float = 1.0,
        clamp_value: float = 0.0,
        add_to_residual: bool = False,  # if True, return h + (h' - h) else return h'
    ):
        self.sae = sae
        self.mode = mode
        self.feature_idx = feature_idx
        self.alpha = alpha
        self.clamp_value = clamp_value
        self.add_to_residual = add_to_residual

        if mode in ("ablate_feature","add_feature","clamp_feature"):
            if feature_idx is None:
                raise ValueError(f"mode={mode} requires feature_idx")

    @torch.no_grad()
    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, d)
        _, z = self.sae.encode(h)          # (B, m)
        z = z.float()

        if self.mode == "reconstruct":
            z_mod = z

        elif self.mode == "ablate_feature":
            z_mod = z.clone()
            z_mod[:, self.feature_idx] = 0.0

        elif self.mode == "clamp_feature":
            z_mod = z.clone()
            z_mod[:, self.feature_idx] = self.clamp_value

        elif self.mode == "add_feature":
            z_mod = z.clone()
            z_mod[:, self.feature_idx] = z_mod[:, self.feature_idx] + self.alpha

        else:
            raise ValueError(f"Unknown intervention mode: {self.mode}")

        # decode -> reconstructed hidden
        h_rec = self.sae.decode(z_mod)     # (B, d)

        if self.add_to_residual:
            return h + (h_rec - h)
        return h_rec


# =========================
# Model adapter: YOU MUST IMPLEMENT THIS FOR YOUR VLA
# =========================
class ModelAdapter:
    """
    Minimal interface your VLA must provide for true interventions.

    You need to implement:
      - get_layer_module_to_hook(layer_idx): returns the nn.Module whose output is the residual stream you want to patch
      - forward_episode_step(ep, t): returns (pred, extra) for a given episode and timestep
        where pred is either:
          * continuous actions (7,) torch
          * OR logits over discrete actions
    """
    def __init__(self, device: str):
        self.device = device

    def get_layer_module_to_hook(self, layer_idx: int) -> nn.Module:
        raise NotImplementedError

    def forward_episode_step(self, ep: Episode, t: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError


# =========================
# Hook-based intervention runner
# =========================
class HookRunner:
    def __init__(self, adapter: ModelAdapter):
        self.adapter = adapter
        self._handle = None

    def install_intervention(
        self,
        layer_module: nn.Module,
        patch_fn: Callable[[torch.Tensor], torch.Tensor],
        token_index: Optional[int] = None,
    ):
        """
        Expects module output shaped like:
          - (B, T, d) or (T, d) or (B, d)
        If token_index is provided and output is (B,T,d), patches only that token.
        """
        def hook(_module, _inp, out):
            # normalize output to tensor
            if not torch.is_tensor(out):
                return out

            if out.dim() == 3 and token_index is not None:
                # (B,T,d) patch only token_index
                out2 = out.clone()
                h = out2[:, token_index, :]
                out2[:, token_index, :] = patch_fn(h)
                return out2

            if out.dim() == 2:
                # (B,d) patch all rows
                return patch_fn(out)

            if out.dim() == 3 and token_index is None:
                # patch all tokens
                B,T,D = out.shape
                flat = out.reshape(B*T, D)
                flat2 = patch_fn(flat)
                return flat2.reshape(B, T, D)

            return out  # fallback

        self._handle = layer_module.register_forward_hook(hook)

    def remove_intervention(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @torch.no_grad()
    def run(
        self,
        episodes: List[Episode],
        timesteps_per_episode: int,
        layer_idx: int,
        intervention: Optional[SAEIntervention],
        token_index: Optional[int],
        out_path: str,
    ):
        layer_module = self.adapter.get_layer_module_to_hook(layer_idx)

        def identity(x): return x

        patch_fn = identity if intervention is None else intervention

        # install hook
        self.install_intervention(layer_module, patch_fn, token_index=token_index)

        records = []
        for ep in episodes:
            acts = load_actions(ep.actions_path)
            T = min(timesteps_per_episode, acts.shape[0])

            for t in range(T):
                pred, extra = self.adapter.forward_episode_step(ep, t)  # pred shape: (7,) or (V,)
                records.append({
                    "group": ep.group,
                    "episode_id": ep.episode_id,
                    "t": int(t),
                    "prompt": ep.prompt,
                    "pred": pred.detach().cpu().tolist(),
                    "extra": {k: (v.detach().cpu().tolist() if torch.is_tensor(v) else v) for k,v in extra.items()},
                    "gt_action": acts[t].tolist() if acts.shape[1] == 7 else None,
                })

        self.remove_intervention()

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Wrote {len(records)} records -> {out_path}")


# =========================
# Metrics for continuous actions
# =========================
def summarize_continuous_deltas(
    base_records: List[Dict[str, Any]],
    intv_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    assert len(base_records) == len(intv_records)

    base = np.asarray([r["pred"] for r in base_records], dtype=np.float32)
    itv  = np.asarray([r["pred"] for r in intv_records], dtype=np.float32)
    if base.ndim != 2:
        raise ValueError("pred must be a 2D array [N, action_dim] for continuous summaries")

    delta = itv - base
    l2 = np.linalg.norm(delta, axis=1)

    out = {
        "n": int(base.shape[0]),
        "delta_l2_mean": float(l2.mean()),
        "delta_l2_median": float(np.median(l2)),
        "delta_l2_p95": float(np.percentile(l2, 95)),
        "delta_mean_per_dim": {ACTION_NAMES[i]: float(delta[:, i].mean()) for i in range(min(7, delta.shape[1]))},
        "delta_abs_mean_per_dim": {ACTION_NAMES[i]: float(np.abs(delta[:, i]).mean()) for i in range(min(7, delta.shape[1]))},
    }
    return out


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--activations_root", type=str, required=True)

    ap.add_argument("--layer_idx", type=int, default=11)
    ap.add_argument("--token_index", type=int, default=-1,
                    help="Which token to patch if hooked tensor is (B,T,d). Use -1 for last token.")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--mode", type=str, default="ablate_feature",
                    choices=["ablate_feature", "add_feature", "clamp_feature", "reconstruct", "none"])
    ap.add_argument("--feature_idx", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--clamp_value", type=float, default=0.0)
    ap.add_argument("--add_to_residual", action="store_true")

    ap.add_argument("--num_episodes", type=int, default=5)
    ap.add_argument("--timesteps_per_episode", type=int, default=50)

    ap.add_argument("--out_dir", type=str, default="./sae_interventions_out")
    args = ap.parse_args()

    # Load SAE
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["d"], nb_concepts=ckpt["nb_concepts"], top_k=ckpt["top_k"], device="cpu")
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval().to(args.device)

    # Episodes
    eps = index_libero_dataset(args.data_root, args.activations_root)
    usable = [ep for ep in eps if ep.act_path and os.path.exists(ep.act_path) and os.path.exists(ep.actions_path)]
    usable = usable[: args.num_episodes]
    if len(usable) == 0:
        raise RuntimeError("No usable episodes found.")

    # Intervention object
    intervention = None
    if args.mode != "none":
        intervention = SAEIntervention(
            sae=sae,
            mode=args.mode,
            feature_idx=args.feature_idx,
            alpha=args.alpha,
            clamp_value=args.clamp_value,
            add_to_residual=args.add_to_residual,
        )

    # -----------------------------
    # IMPORTANT: Provide your adapter
    # -----------------------------
    # You need to implement this for your VLA. See below for a template.
    adapter = build_your_model_adapter(device=args.device)  # <-- YOU implement this

    runner = HookRunner(adapter)

    os.makedirs(args.out_dir, exist_ok=True)
    base_path = os.path.join(args.out_dir, "baseline.json")
    intv_path = os.path.join(args.out_dir, f"intv_{args.mode}_feat{args.feature_idx}.json")

    # Run baseline
    runner.run(
        episodes=usable,
        timesteps_per_episode=args.timesteps_per_episode,
        layer_idx=args.layer_idx,
        intervention=None,
        token_index=(args.token_index if args.token_index >= 0 else args.token_index),
        out_path=base_path,
    )

    # Run intervention
    runner.run(
        episodes=usable,
        timesteps_per_episode=args.timesteps_per_episode,
        layer_idx=args.layer_idx,
        intervention=intervention,
        token_index=(args.token_index if args.token_index >= 0 else args.token_index),
        out_path=intv_path,
    )

    # Summarize if predictions are continuous action vectors
    with open(base_path, "r") as f:
        base_records = json.load(f)
    with open(intv_path, "r") as f:
        intv_records = json.load(f)

    try:
        summary = summarize_continuous_deltas(base_records, intv_records)
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("Summary (continuous deltas):")
        print(json.dumps(summary, indent=2))
    except Exception as e:
        print(f"Could not summarize as continuous actions: {e}")
        print("If your pred is logits, compute KL / top-1 flip rate instead.")



def build_your_model_adapter(device: str) -> ModelAdapter:
    """
    TODO: Replace with your actual model loading + inference.

    You must return a ModelAdapter with:
      - self.model
      - get_layer_module_to_hook(layer_idx)
      - forward_episode_step(ep, t)

    forward_episode_step should produce a tensor prediction. For continuous actions, shape (7,).
    """
    class YourAdapter(ModelAdapter):
        def __init__(self, device: str):
            super().__init__(device=device)
            # TODO: load your VLA (and action expert if needed) here
            # self.model = ...
            # self.model.eval().to(device)
            raise NotImplementedError("Implement model loading here.")

        def get_layer_module_to_hook(self, layer_idx: int) -> nn.Module:
            # TODO: return the module that outputs the residual stream for that layer.
            # Common patterns:
            #   - self.model.transformer.h[layer_idx]
            #   - self.model.backbone.layers[layer_idx]
            #   - self.model.language_model.model.layers[layer_idx]
            raise NotImplementedError

        def forward_episode_step(self, ep: Episode, t: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
            # TODO:
            #  1) Load observation at time t (image) + prompt ep.prompt (and maybe history)
            #  2) Run model forward to get action prediction or logits
            # Return:
            #   pred: torch.Tensor (7,) OR logits (V,)
            #   extra: dict for anything useful (e.g., chosen action, logits, subtask)
            raise NotImplementedError

    return YourAdapter(device=device)

if __name__ == "__main__":
    main()