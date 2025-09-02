#!/usr/bin/env python3
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt

def to_rgba(c):
    c = list(c)
    if len(c) == 3:
        c = c + [1.0]  # default alpha = 1.0
    # clip to [0,1] just in case
    return tuple(float(np.clip(x, 0.0, 1.0)) for x in c)

def main():
    ap = argparse.ArgumentParser(description="Plot DiffVG JSON strokes")
    ap.add_argument("json_path", help="Path to *_diffvg.json")
    ap.add_argument("--bg", type=str, default=None, help="Optional background image")
    ap.add_argument("--width", type=int, default=None, help="Canvas width (if JSON points are normalized)")
    ap.add_argument("--height", type=int, default=None, help="Canvas height (if JSON points are normalized)")
    ap.add_argument("--figsize", type=float, nargs=2, default=(8, 8), help="Matplotlib figure size")
    ap.add_argument("--out", type=str, default=None, help="Save to this file instead of showing")
    args = ap.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)
    strokes = data.get("strokes", [])
    if not strokes:
        print("No strokes found in JSON.")
        return

    # Determine canvas size / normalization
    # If bg provided, we’ll use its size as the canvas.
    bg_img = None
    W = H = None
    if args.bg:
        try:
            bg_img = plt.imread(args.bg)
            H, W = bg_img.shape[0], bg_img.shape[1]
        except Exception as e:
            print(f"Warning: could not read background image: {e}")

    # If no bg, infer from JSON points if they look like pixels
    # else require width/height or default to 1024x1024
    all_pts = np.concatenate([np.array(s["points"], dtype=float) for s in strokes], axis=0)
    looks_normalized = np.max(all_pts) <= 1.01
    if looks_normalized:
        if W is None or H is None:
            W = args.width if args.width is not None else 1024
            H = args.height if args.height is not None else 1024
    else:
        if W is None or H is None:
            # infer from max coords
            W = int(np.ceil(np.max(all_pts[:, 0]) + 1))
            H = int(np.ceil(np.max(all_pts[:, 1]) + 1))

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    if bg_img is not None:
        ax.imshow(bg_img, origin="upper")
    # Set a top-left origin coordinate system
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw strokes
    count = 0
    for s in strokes:
        pts = np.array(s["points"], dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue  # skip degenerate strokes
        if looks_normalized:
            pts = pts.copy()
            pts[:, 0] *= W
            pts[:, 1] *= H
        rgba = to_rgba(s.get("color", [0, 0, 0, 1]))
        lw = float(s.get("stroke_width", 1.0))
        ax.plot(pts[:, 0], pts[:, 1], "-", linewidth=lw, color=rgba, solid_capstyle="round")
        count += 1

    print(f"Plotted {count} strokes "
          f"(normalized={looks_normalized}, canvas={W}x{H})")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        plt.savefig(args.out, bbox_inches="tight", pad_inches=0, dpi=200)
        print(f"Saved to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
