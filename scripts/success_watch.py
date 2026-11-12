"""Success rate of the latest rl-agent run, redrawn in the terminal every 2 s.

Leave it open in a second terminal (Cursor: split the terminal panel). It follows
the newest run in the run folder's episodes.jsonl: when `rl-agent run` starts
again, the chart starts over. Ctrl+C stops.

    .venv/bin/python scripts/success_watch.py outputs/rl/dsrl-clean
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

GREEN, RED, YELLOW, BLUE, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[34m", "\033[2m", "\033[1m", "\033[0m")
MARK = {"success": GREEN + "●", "safety": RED + "■", "timeout": YELLOW + "▲", "wrong_button": YELLOW + "◆"}
BLOCKS = " ▁▂▃▄▅▆▇█"


def latest_run(log: Path) -> list[dict]:
    rows = []
    for line in log.read_text().splitlines() if log.exists() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rows and row.get("episode", 0) <= rows[-1].get("episode", 0):
            rows = []  # episode numbers restart: a new run
        rows.append(row)
    return rows


def render(rows: list[dict], log: Path, window: int, height: int) -> str:
    width = shutil.get_terminal_size((80, 24)).columns
    out = [f"{BOLD}Success rate, latest run{RESET}  {DIM}{log} · refreshes every 2 s · Ctrl+C stops{RESET}"]
    if not rows:
        return "\n".join(out + ["", "No episodes yet."])
    ok = [r["outcome"] == "success" for r in rows]
    train = [o for o, r in zip(ok, rows) if not r.get("eval")]
    last = train[-window:]
    safety = sum(r["outcome"] == "safety" for r in rows)
    out.append(f"started {datetime.fromisoformat(rows[0]['time']).astimezone():%H:%M:%S} · {len(rows)} episodes · "
               f"{BOLD}{sum(ok)}/{len(rows)} ({sum(ok) / len(rows):.0%}){RESET} succeeded · "
               f"last {len(last)}: {BOLD}{sum(last)}/{len(last)} ({sum(last) / max(len(last), 1):.0%}){RESET} · "
               f"safety stops {safety}")
    rates, seen = [], []
    for o, r in zip(ok, rows):
        if not r.get("eval"):
            seen.append(o)
        recent = seen[-window:]
        rates.append(sum(recent) / len(recent) if recent else 0.0)
    columns = max(1, (width - 7) // 2)
    first = max(0, len(rows) - columns)
    shown = range(first, len(rows))
    out.append("")
    out.append(f"{DIM}moving average over the last {window} episodes (eval episodes excluded){RESET}")
    for level in range(height, 0, -1):
        label = f"{round(100 * level / height):>4}%" if level in (height, height // 2) else "     "
        cells = []
        for i in shown:
            fill = rates[i] * height - (level - 1)
            cells.append(BLUE + BLOCKS[max(0, min(8, int(round(fill * 8))))] + RESET + " ")
        out.append(f"{DIM}{label}{RESET} │" + "".join(cells))
    out.append("   0% └" + "─" * (2 * len(shown)))
    out.append("  out  " + "".join(MARK.get(rows[i]["outcome"], "?") + RESET + " " for i in shown))
    out.append("  eval " + "".join(("○ " if rows[i].get("eval") else "  ") for i in shown))
    axis = [" "] * (2 * len(shown))
    for i in shown:
        if (i + 1) % 5 == 0 or i == first:
            for k, ch in enumerate(str(i + 1)):
                if 2 * (i - first) + k < len(axis):
                    axis[2 * (i - first) + k] = ch
    out.append("  ep   " + "".join(axis))
    out.append(f"       {GREEN}●{RESET} success  {RED}■{RESET} safety stop  {YELLOW}▲{RESET} timeout  "
               f"{YELLOW}◆{RESET} wrong button  ○ eval")
    r = rows[-1]
    detail = r.get("reason") or (f"pressed in {r['time_to_press_s']} s" if r.get("time_to_press_s") else "")
    out.append(f"\nlast: episode {r['episode']} {r['outcome']}{' · ' + detail if detail else ''} · "
               f"moved up to {r.get('moved_max_deg')}° · {r.get('learner_updates')} SAC updates")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="?", default="outputs/rl/dsrl-clean")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--once", action="store_true", help="print once and exit")
    args = parser.parse_args()
    log = Path(args.run_dir) / "episodes.jsonl"
    if args.once:
        print(render(latest_run(log), log, args.window, args.height))
        return
    try:
        sys.stdout.write("\033[?25l")
        while True:
            frame = render(latest_run(log), log, args.window, args.height)
            sys.stdout.write("\033[H\033[2J" + frame + "\n")
            sys.stdout.flush()
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")


if __name__ == "__main__":
    main()
