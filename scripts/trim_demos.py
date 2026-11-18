"""Keep the part of each demo from just before the arm leaves rest to the press.

The demos wait at rest for ~3 s, reach, press, and go back to rest. The wait makes
"at rest" mean both "stay" and "go", and the return makes every half-raised pose
mean both "up" and "down"; a policy trained on them hesitates and goes back and
forth. This writes a copy with, per episode:

- start: ``lead`` frames before any joint first moves ``move_deg`` from where it began;
- end: ``tail`` frames after the last frame still at least ``reach_fraction`` of the
  episode's furthest distance from the start pose (the press, before the return).

There is no button signal in the dataset; the press is taken to be where the arm
is furthest out, and ``--preview`` prints the cuts to check before writing.

    uv run python scripts/trim_demos.py --preview
    uv run python scripts/trim_demos.py --out outputs/datasets/corl-press-green-clean
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def cut(states: np.ndarray, *, move_deg: float, reach_fraction: float, lead: int, tail: int) -> tuple[int, int]:
    """[start, end) frames of one episode's states."""
    distance = np.abs(states - states[0]).max(axis=1)
    moved = np.flatnonzero(distance > move_deg)
    if len(moved) == 0:
        return 0, len(states)
    start = max(0, moved[0] - lead)
    out = np.flatnonzero(distance >= reach_fraction * distance.max())
    end = min(len(states), out[-1] + 1 + tail)
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="pravsels/corl-press-green")
    parser.add_argument("--out", default="outputs/datasets/corl-press-green-clean")
    parser.add_argument("--repo-id", default="local/corl-press-green-clean")
    parser.add_argument("--move-deg", type=float, default=3.0)
    parser.add_argument("--reach-fraction", type=float, default=0.8)
    parser.add_argument("--lead", type=int, default=6, help="frames kept before the arm moves (0.2 s)")
    parser.add_argument("--tail", type=int, default=3, help="frames kept after the press (0.1 s)")
    parser.add_argument("--preview", action="store_true", help="print the cuts and stop")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = LeRobotDataset(args.dataset, video_backend="pyav")
    table = source.hf_dataset.with_format("numpy")
    states = np.stack(table["observation.state"])
    episodes = np.asarray(table["episode_index"])
    fps = source.fps
    cuts = {}
    for episode in np.unique(episodes):
        index = np.flatnonzero(episodes == episode)
        start, end = cut(states[index], move_deg=args.move_deg, reach_fraction=args.reach_fraction,
                         lead=args.lead, tail=args.tail)
        cuts[int(episode)] = (index[start:end], start, end, len(index))
    kept = sum(len(c[0]) for c in cuts.values())
    for episode, (_, start, end, total) in cuts.items():
        print(f"episode {episode:2d}: keep {start / fps:4.1f}-{end / fps:4.1f} s of {total / fps:4.1f} s")
    print(f"{kept} of {len(states)} frames kept ({kept / fps:.0f} s)")
    if args.preview:
        return

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")
    skip = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    features = {
        key: {k: v for k, v in value.items() if k != "info"}
        for key, value in source.meta.features.items() if key not in skip
    }
    target = LeRobotDataset.create(args.repo_id, fps=fps, features=features, root=out,
                                   robot_type=source.meta.robot_type, use_videos=True)
    image_keys = [k for k, v in features.items() if v["dtype"] == "video"]
    task = source.meta.tasks.index[0] if hasattr(source.meta.tasks, "index") else "Press the green button"
    for episode, (frames, *_rest) in cuts.items():
        for i in frames:
            item = source[int(i)]
            frame = {"task": task}
            for key in features:
                value = item[key]
                if key in image_keys:
                    value = (value.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
                else:
                    value = value.numpy()
                frame[key] = value
            target.add_frame(frame)
        target.save_episode()
        print(f"wrote episode {episode}", flush=True)
    target.finalize()
    print(f"wrote {out} ({target.num_episodes} episodes, {target.num_frames} frames)")


if __name__ == "__main__":
    main()
