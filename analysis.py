import os, glob, json, math, subprocess
from collections import defaultdict, Counter

import numpy as np
import torch

# Optional: use decord (fast) or opencv for frames
try:
    import decord
    from decord import VideoReader, cpu
    HAVE_DECORD = True
except Exception:
    HAVE_DECORD = False

from overcomplete.sae import TopKSAE

# -----------------------------
# Config
# -----------------------------
device = "cuda"
ckpt_path = "/path/to/sae_checkpoint.pt"  # TODO
npy_dir   = "/n/netscratch/sham_lab/Lab/chloe00/pi0_activations"  # your activations
data_root = "/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/data/libero"

# Choose which layer’s activations to analyze (if you have per-layer activations)
LAYER_IDX = 0

# How many top examples per concept to keep (for frames/clips + summaries)
TOP_M = 200

# Define "active" threshold if your SAE can output small nonzeros.
# For TopK SAEs, codes are typically sparse with exact zeros; threshold=0 is fine.
ACT_THRESH = 0.0

# Clip settings
SAVE_DIR = "concept_viz"
os.makedirs(SAVE_DIR, exist_ok=True)
CLIP_HALF_SECONDS = 1.5  # save ~3s clip around peak timestep
FPS_FALLBACK = 30.0

# -----------------------------
# Helpers
# -----------------------------
def stem(p):
    return os.path.splitext(os.path.basename(p))[0]

def safe_read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def list_split_names(root):
    # returns ["10","goal","object","spatial"] etc (subdirs only)
    return [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]

def build_episode_index(data_root):
    """
    Build mapping: episode_stem -> dict(split, json_path, mp4_path)
    """
    idx = {}
    for split in list_split_names(data_root):
        actions_dir = os.path.join(data_root, split, "actions")
        videos_dir  = os.path.join(data_root, split, "videos")
        jsons = glob.glob(os.path.join(actions_dir, "*.json"))
        for jp in jsons:
            s = stem(jp)
            mp4 = os.path.join(videos_dir, s + ".mp4")
            if os.path.exists(mp4):
                idx[s] = {"split": split, "json": jp, "mp4": mp4}
    return idx

def load_sae(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sae = TopKSAE(
        ckpt["d"],
        nb_concepts=ckpt["nb_concepts"],
        top_k=ckpt["top_k"],
        device="cpu",
    )
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval().to(device)
    return sae, ckpt

def sae_encode(sae, x):
    """
    Returns concept codes / activations for x.
    This depends on your TopKSAE API. Common patterns:
      - sae.encode(x) -> codes
      - sae(x) -> (recon, codes) or dict
    Adjust this function to your implementation.
    """
    with torch.no_grad():
        out = sae(x)
    # ---- ADJUST HERE if your SAE returns a tuple/dict ----
    # Try common conventions:
    if isinstance(out, tuple) and len(out) == 2:
        recon, codes = out
        return codes
    if isinstance(out, dict) and "codes" in out:
        return out["codes"]
    # If out is already codes:
    return out

def discretize_action(a, decimals=2):
    """
    For continuous actions: bucket by rounding so "most common action" makes sense.
    You can replace this with clustering later.
    """
    a = np.asarray(a)
    return tuple(np.round(a, decimals=decimals).tolist())

def extract_prompt(meta):
    """
    Try common keys for language instruction. Adjust if your JSON schema differs.
    """
    for k in ["language", "instruction", "goal", "prompt", "task_description", "natural_language_instruction"]:
        if k in meta and isinstance(meta[k], str):
            return meta[k]
    # Sometimes nested
    if "metadata" in meta and isinstance(meta["metadata"], dict):
        for k in ["language", "instruction", "goal", "prompt"]:
            if k in meta["metadata"] and isinstance(meta["metadata"][k], str):
                return meta["metadata"][k]
    return None

def extract_actions_sequence(meta):
    """
    Try to extract actions as a list/array of shape (T, act_dim).
    Adjust depending on your JSON format.
    """
    for k in ["actions", "action", "robot_actions"]:
        if k in meta:
            return meta[k]
    if "trajectory" in meta and isinstance(meta["trajectory"], dict) and "actions" in meta["trajectory"]:
        return meta["trajectory"]["actions"]
    return None

def get_video_fps_and_len(mp4_path):
    """
    If you have decord, use it. Otherwise fallback to FPS_FALLBACK and unknown length.
    """
    if HAVE_DECORD:
        vr = VideoReader(mp4_path, ctx=cpu(0))
        # decord doesn't always expose fps cleanly; approximate via metadata if available
        # We'll use fallback fps but get number of frames.
        nframes = len(vr)
        return FPS_FALLBACK, nframes
    return FPS_FALLBACK, None

def timestep_to_seconds(t, T, mp4_path):
    fps, nframes = get_video_fps_and_len(mp4_path)
    if nframes is not None and T is not None and T > 0:
        # map timestep to approximate frame index
        frame_idx = int(round((t / (T - 1)) * max(nframes - 1, 0)))
        return frame_idx / fps
    # fallback: treat timestep as frame index
    return t / fps

def save_clip_ffmpeg(mp4_in, t_center, out_path, half_seconds=1.5):
    t0 = max(t_center - half_seconds, 0.0)
    dur = 2 * half_seconds
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t0:.3f}",
        "-i", mp4_in,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-an",
        out_path
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def save_frame(mp4_path, t_seconds, out_png):
    if HAVE_DECORD:
        vr = VideoReader(mp4_path, ctx=cpu(0))
        fps = FPS_FALLBACK
        frame_idx = int(round(t_seconds * fps))
        frame_idx = max(0, min(frame_idx, len(vr) - 1))
        frame = vr[frame_idx].asnumpy()  # HWC RGB
        import imageio
        imageio.imwrite(out_png, frame)
        return True
    else:
        # ffmpeg one-frame extraction
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{t_seconds:.3f}",
            "-i", mp4_path,
            "-frames:v", "1",
            out_png
        ]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(out_png)

# -----------------------------
# Main: load metadata and activations
# -----------------------------
episode_idx = build_episode_index(data_root)
print(f"Indexed {len(episode_idx)} episodes with both json+mp4.")

npy_paths = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
print(f"Found {len(npy_paths)} activation files.")

sae, ckpt = load_sae(ckpt_path, device=device)
nb_concepts = ckpt["nb_concepts"]
print(f"Loaded SAE: d={ckpt['d']} concepts={nb_concepts} top_k={ckpt['top_k']}")

# For each concept: keep top examples by activation value
# Store: (score, episode_stem, t, split, json_path, mp4_path, prompt, action_token)
top_examples = {c: [] for c in range(nb_concepts)}

# Also aggregate counts
concept_action_counts = [Counter() for _ in range(nb_concepts)]
concept_prompt_counts = [Counter() for _ in range(nb_concepts)]

def push_topk(lst, item, k):
    # keep largest k by score
    lst.append(item)
    lst.sort(key=lambda x: x[0], reverse=True)
    if len(lst) > k:
        lst.pop()

for npy_path in npy_paths:
    ep = stem(npy_path)
    if ep not in episode_idx:
        # If this happens a lot, your naming isn’t aligned.
        # You can print and later implement a mapping strategy.
        continue

    meta_paths = episode_idx[ep]
    meta = safe_read_json(meta_paths["json"])
    prompt = extract_prompt(meta)
    actions_seq = extract_actions_sequence(meta)

    A = np.load(npy_path).astype(np.float32)

    # ---- SHAPE HANDLING ----
    # Common possibilities:
    #   (T, d)                          -> single layer already
    #   (T, num_layers, d)              -> per timestep per layer
    #   (T, num_layers, 1, d) or similar
    #
    # Adapt below to your true shape.
    if A.ndim == 2:
        # (T, d)
        X = A
        T = X.shape[0]
    elif A.ndim == 3:
        # (T, num_layers, d)
        T = A.shape[0]
        X = A[:, LAYER_IDX, :]
    elif A.ndim == 4:
        # e.g. (T, num_layers, ?, d) -> squeeze middle dims
        T = A.shape[0]
        X = A[:, LAYER_IDX, ...]
        X = X.reshape(T, -1)
    else:
        raise ValueError(f"Unexpected activation shape {A.shape} in {npy_path}")

    # actions_seq alignment: expect len(actions_seq)==T or close
    # If actions are stored as dicts, or one action per timestep, adjust discretize.
    def get_action_token(t):
        if actions_seq is None:
            return None
        if isinstance(actions_seq, list) and len(actions_seq) > 0:
            tt = min(t, len(actions_seq) - 1)
            a = actions_seq[tt]
            # a can be list[float] or dict; handle both
            if isinstance(a, dict) and "action" in a:
                a = a["action"]
            if isinstance(a, (list, tuple, np.ndarray)):
                return discretize_action(a, decimals=2)
            if isinstance(a, str):
                return a
        return None

    # SAE encode in chunks to avoid OOM
    bs = 4096
    for start in range(0, T, bs):
        end = min(T, start + bs)
        xb = torch.from_numpy(X[start:end]).to(device)
        codes = sae_encode(sae, xb)  # (chunk, nb_concepts) expected
        codes = codes.detach().float().cpu().numpy()

        # For each timestep, find active concepts (sparse => fast)
        # If TopK, many zeros; use nonzero indices.
        for i in range(codes.shape[0]):
            t = start + i
            row = codes[i]
            active = np.where(row > ACT_THRESH)[0]
            if active.size == 0:
                continue
            act_tok = get_action_token(t)

            for c in active:
                score = float(row[c])
                # Aggregate "most common"
                if act_tok is not None:
                    concept_action_counts[c][act_tok] += 1
                if prompt is not None:
                    concept_prompt_counts[c][prompt] += 1

                push_topk(
                    top_examples[c],
                    (
                        score, ep, t,
                        meta_paths["split"],
                        meta_paths["json"],
                        meta_paths["mp4"],
                        prompt,
                        act_tok
                    ),
                    TOP_M
                )

print("Done scanning activations.")

# -----------------------------
# Emit summaries + save frames/clips for top examples
# -----------------------------
def summarize_top(counter, k=10):
    return counter.most_common(k)

REPORT_PATH = os.path.join(SAVE_DIR, "concept_report.txt")
with open(REPORT_PATH, "w") as f:
    for c in range(nb_concepts):
        f.write(f"\n=== Concept {c} ===\n")

        f.write("Top actions:\n")
        for tok, cnt in summarize_top(concept_action_counts[c], k=10):
            f.write(f"  {tok}: {cnt}\n")

        f.write("Top prompts:\n")
        for p, cnt in summarize_top(concept_prompt_counts[c], k=10):
            # avoid enormous lines
            p_short = (p[:160] + "...") if p and len(p) > 160 else p
            f.write(f"  ({cnt}) {p_short}\n")

        # Save a few visualizations
        exs = top_examples[c][:12]  # save first 12 frames/clips
        f.write("Top examples (saved frames/clips):\n")
        for j, (score, ep, t, split, jp, mp4, prompt, act_tok) in enumerate(exs):
            # Map timestep->seconds
            T_guess = None  # if you can store T per episode, pass it here
            t_sec = timestep_to_seconds(t, T_guess, mp4)

            frame_path = os.path.join(SAVE_DIR, f"c{c:04d}_ex{j:02d}_{ep}_t{t}.png")
            clip_path  = os.path.join(SAVE_DIR, f"c{c:04d}_ex{j:02d}_{ep}_t{t}.mp4")

            save_frame(mp4, t_sec, frame_path)
            save_clip_ffmpeg(mp4, t_sec, clip_path, half_seconds=CLIP_HALF_SECONDS)

            f.write(f"  score={score:.4f} split={split} ep={ep} t={t} "
                    f"frame={os.path.basename(frame_path)} clip={os.path.basename(clip_path)}\n")

print(f"Wrote report: {REPORT_PATH}")
print(f"Saved frames/clips under: {SAVE_DIR}/")