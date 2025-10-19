from glob import glob
import os
import re
import zlib
import hashlib
from typing import Optional, List

import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np


def _natural_key(s: str):
    """Natural sort key so 'flow_2_3.png' < 'flow_10_11.png'."""
    base = os.path.basename(s)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', base)]


def _uniform_indices(n_src: int, n_dst: int) -> List[int]:
    """Uniformly sample n_dst indices from [0..n_src-1] (deterministic)."""
    if n_dst <= 1:
        return [0]
    if n_src <= 1:
        return [0] * n_dst
    return [int(round(i * (n_src - 1) / (n_dst - 1))) for i in range(n_dst)]


def _stable_seed_from_path(base_seed: int, path: str) -> int:
    """
    Combine a user-provided base_seed with a stable hash of a path into a uint32 seed.
    Ensures per-sample determinism across workers and epochs.
    """
    h1 = zlib.adler32(path.encode("utf-8")) & 0xFFFFFFFF
    # Mix in a strong hash to reduce collisions across large datasets
    h2 = int(hashlib.sha1(path.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF
    return (int(base_seed) ^ h1 ^ h2) % (2**32)


class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir='data/ufc10', split='train', transform=None, base_seed: Optional[int] = None):
        """
        base_seed is accepted for API symmetry, but unused here since we don't
        do random sampling inside this dataset. Randomness (if any) in transforms
        is controlled by the DataLoader worker seeds.
        """
        self.root_dir = os.path.abspath(root_dir)
        self.split = split
        self.transform = transform
        self.base_seed = base_seed

        # Accept multiple extensions for frames
        candidates = []
        for ext in ("jpg", "jpeg", "png"):
            candidates.extend(glob(os.path.join(self.root_dir, "frames", split, "*", "*", f"*.{ext}")))
        self.frame_paths = sorted(set(candidates), key=_natural_key)

        meta_csv = os.path.join(self.root_dir, "metadata", f"{split}.csv")
        if not os.path.isfile(meta_csv):
            raise FileNotFoundError(
                f"Metadata CSV not found: {meta_csv}\nLooked under: {self.root_dir}/metadata"
            )
        self.df = pd.read_csv(meta_csv)

        if len(self.frame_paths) == 0:
            raise FileNotFoundError(
                f"No frames found under {self.root_dir}/frames/{split}/*/*/*.(jpg|jpeg|png)"
            )

    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        rows = self.df.loc[self.df[attr] == value]
        if rows.empty:
            raise KeyError(f"No metadata row where {attr} == {value}")
        return rows

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        # expects .../<class>/<video_name>/<frame_file>
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = int(video_meta['label'].item())

        try:
            frame = Image.open(frame_path).convert("RGB")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing frame file: {frame_path}") from e

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir='data/ufc10',
                 split='train',
                 transform=None,
                 stack_frames=True,
                 load_optical_flow=False,
                 n_sampled_frames: int = 10,
                 sample_strategy: str = "uniform",   # "uniform" (deterministic) or "random"
                 base_seed: Optional[int] = None):

        """
        If sample_strategy == "random", frames are sampled deterministically per video
        using a seed derived from (base_seed, video path). This remains reproducible
        across runs, workers, and epochs as long as base_seed is fixed.
        """
        self.root_dir = os.path.abspath(root_dir)
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        self.load_optical_flow = load_optical_flow

        self.n_sampled_frames = int(n_sampled_frames)
        self.sample_strategy = sample_strategy.lower()
        if self.sample_strategy not in {"uniform", "random"}:
            raise ValueError(f"sample_strategy must be 'uniform' or 'random', got {sample_strategy}")
        self.base_seed = 0 if base_seed is None else int(base_seed)

        # Index videos (if you have .avi files); required for mapping class/video_name
        videos_glob = os.path.join(self.root_dir, "videos", split, "*", "*.avi")
        self.video_paths = sorted(glob(videos_glob), key=_natural_key)

        meta_csv = os.path.join(self.root_dir, "metadata", f"{split}.csv")
        if not os.path.isfile(meta_csv):
            raise FileNotFoundError(
                f"Metadata CSV not found: {meta_csv}\nLooked under: {self.root_dir}/metadata"
            )
        self.df = pd.read_csv(meta_csv)

        if len(self.video_paths) == 0:
            # Helpful message if videos are not present
            maybe_frames = glob(os.path.join(self.root_dir, "frames", split, "*", "*"))
            maybe_flows1 = glob(os.path.join(self.root_dir, "flows-png", split, "*", "*"))
            maybe_flows2 = glob(os.path.join(self.root_dir, "flows_png", split, "*", "*"))
            raise FileNotFoundError(
                "No videos found under "
                f"{self.root_dir}/videos/{split}/*/*.avi\n"
                f"- If you intended to use frames-only, adapt indexing to scan frames folders.\n"
                f"- Quick check: frames present? {'YES' if len(maybe_frames)>0 else 'NO'}; "
                f"flows (flows-png)? {'YES' if len(maybe_flows1)>0 else 'NO'}; "
                f"flows (flows_png)? {'YES' if len(maybe_flows2)>0 else 'NO'}"
            )

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        rows = self.df.loc[self.df[attr] == value]
        if rows.empty:
            raise KeyError(f"No metadata row where {attr} == {value}")
        return rows

    def __getitem__(self, idx):
        """
        Expect video path layout:
          <root>/videos/<split>/<class>/<video_name>.avi
        We derive:
          frames: <root>/frames/<split>/<class>/<video_name>/
          flows : <root>/<flows_dir>/<split>/<class>/<video_name>/   where flows_dir in {'flows-png','flows_png'}
        """
        video_path = self.video_paths[idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # class name is the directory under .../videos/<split>/...
        split_root = os.path.join(self.root_dir, "videos", self.split)
        rel = os.path.relpath(video_path, split_root)   # '<class>/<video>.avi'
        class_name = rel.split(os.sep)[0]

        # ---- RGB frames dir (robust join; no string replace)
        frames_dir = os.path.join(self.root_dir, "frames", self.split, class_name, video_name)
        video_frames = self.load_frames(frames_dir, is_flow=False, video_key=video_path)

        # label from metadata
        video_meta = self._get_meta('video_name', video_name)
        label = int(video_meta['label'].item())

        # transform RGB
        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]

        if self.stack_frames:
            # [T, C, H, W] -> [C, T, H, W]
            frames = torch.stack(frames, dim=0).permute(1, 0, 2, 3)

        # ---- Optional: optical flow
        if self.load_optical_flow:
            # try both flows-png and flows_png
            flow_base_candidates = ["flows-png", "flows_png"]
            flow_dir = None
            tried = []
            for base in flow_base_candidates:
                candidate = os.path.join(self.root_dir, base, self.split, class_name, video_name)
                tried.append(candidate)
                if os.path.isdir(candidate):
                    flow_dir = candidate
                    break
            if flow_dir is None:
                raise FileNotFoundError(
                    "Flow directory does not exist for sample.\n"
                    f"Tried:\n" + "\n".join(tried)
                )

            flow_frames = self.load_frames(flow_dir, is_flow=True, video_key=video_path)

            if self.transform:
                flow_frames = [self.transform(frame) for frame in flow_frames]
            else:
                flow_frames = [T.ToTensor()(frame) for frame in flow_frames]

            if self.stack_frames:
                flow_frames = torch.stack(flow_frames, dim=0).permute(1, 0, 2, 3)

            return {'frames': frames, 'flow': flow_frames, 'label': label}

        return frames, label

    def load_frames(self, frames_dir, is_flow=False, video_key: Optional[str] = None):
        """
        Robust frame loader:
          - RGB: accept frame_*.jpg/png or any *.jpg/jpeg/png
          - Flow: accept flow_*.png (covers flow_1.png, flow_0001.png, flow_1_2.png, ...)
          - Natural sorting
          - Sampling to exactly self.n_sampled_frames using either:
              * 'uniform' (deterministic, old behavior)
              * 'random'  (reproducible given base_seed)
        """
        if not os.path.isdir(frames_dir):
            raise FileNotFoundError(f"Frames directory does not exist: {frames_dir}")

        files = []

        if is_flow:
            files.extend(glob(os.path.join(frames_dir, "flow_*.png")))
        else:
            # Prefer canonical "frame_*"
            for ext in ("jpg", "jpeg", "png"):
                files.extend(glob(os.path.join(frames_dir, f"frame_*.{ext}")))
            if not files:
                # fallback to any images
                for ext in ("jpg", "jpeg", "png"):
                    files.extend(glob(os.path.join(frames_dir, f"*.{ext}")))

        files = sorted(set(files), key=_natural_key)

        if len(files) == 0:
            kind = "flow" if is_flow else "rgb"
            raise FileNotFoundError(
                f"No {kind} frames found in {frames_dir}.\n"
                f"Expected patterns:\n"
                f"  - flow: flow_*.png\n"
                f"  - rgb : frame_*.jpg|jpeg|png OR *.jpg|jpeg|png"
            )

        # sample to exactly N frames
        N = self.n_sampled_frames
        if self.sample_strategy == "uniform":
            if len(files) >= N:
                idxs = _uniform_indices(len(files), N)
                files = [files[i] for i in idxs]
            else:
                files = files + [files[-1]] * (N - len(files))
        else:
            # Deterministic per-video random sampling using a stable seed
            key = video_key or frames_dir
            seed = _stable_seed_from_path(self.base_seed, key)
            rng = np.random.default_rng(seed)

            if len(files) >= N:
                idxs = rng.choice(len(files), size=N, replace=False)
            else:
                idxs = rng.choice(len(files), size=N, replace=True)

            # Sort to keep temporal order after random selection (optional but common)
            idxs = np.sort(idxs)
            files = [files[i] for i in idxs.tolist()]

        frames = []
        for fp in files:
            try:
                frames.append(Image.open(fp).convert("RGB"))
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Missing frame file: {fp}") from e

        return frames


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = 'data/ufc10'

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    # FrameImageDataset does not use base_seed internally (kept for API symmetry)
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform, base_seed=42)

    # Uniform sampling (original behavior, deterministic)
    framevideostack_dataset = FrameVideoDataset(
        root_dir=root_dir, split='val', transform=transform, stack_frames=True,
        n_sampled_frames=10, sample_strategy="uniform", base_seed=42
    )

    # Random-but-reproducible sampling
    framevideolist_dataset = FrameVideoDataset(
        root_dir=root_dir, split='val', transform=transform, stack_frames=False,
        n_sampled_frames=10, sample_strategy="random", base_seed=42
    )

    frameimage_loader = DataLoader(frameimage_dataset, batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset, batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset, batch_size=8, shuffle=False)

    # Example debug prints:
    # for frames, labels in frameimage_loader:
    #     print(frames.shape, labels.shape)

    # for video_frames, labels in framevideolist_loader:
    #     print(45*'-')
    #     for frame in video_frames:
    #         print(frame.shape, labels.shape)

    for video_frames, labels in framevideostack_loader:
        print(video_frames.shape, labels.shape)  # [batch, channels, number of frames, height, width]
