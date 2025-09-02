#!/usr/bin/env python3
import json, argparse, os

def clamp01(x): 
    return max(0.0, min(1.0, float(x)))

def brightness_from_rgb(rgb, mode="luma"):
    r, g, b = (clamp01(rgb[0]), clamp01(rgb[1]), clamp01(rgb[2]))
    if mode == "max":
        return max(r, g, b)                # matches your init-on-color idea
    if mode == "mean":
        return (r + g + b) / 3.0
    # default: ITU-R BT.709 luma (perceptual-ish)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def main():
    ap = argparse.ArgumentParser(
        description="Remove strokes from a diffvg JSON if their *stroke color* is too dark."
    )
    ap.add_argument("--json_in",  required=True, help="Input strokes JSON")
    ap.add_argument("--json_out", default=None,  help="Output JSON (default: *_darkpruned.json)")
    ap.add_argument("--thresh",   type=float, default=0.08,
                    help="Brightness threshold in [0..1]. Remove if brightness <= thresh. Default 0.08")
    ap.add_argument("--mode",     choices=["luma","max","mean"], default="luma",
                    help="Brightness metric. 'max' matches your brightness mask style.")
    ap.add_argument("--mul_alpha", action="store_true",
                    help="Multiply brightness by alpha if present (off by default).")
    args = ap.parse_args()

    with open(args.json_in, "r") as f:
        data = json.load(f)

    strokes = list(data.get("strokes", []))
    kept, removed = [], 0

    for s in strokes:
        col = s.get("color", None) or s.get("stroke_color", None)
        if not col or len(col) < 3:
            # If no color, keep it (or change to remove if you prefer)
            kept.append(s)
            continue

        br = brightness_from_rgb(col, mode=args.mode)
        if args.mul_alpha and len(col) >= 4:
            br *= clamp01(col[3])

        if br <= args.thresh:
            removed += 1
            continue
        kept.append(s)

    out = dict(data)
    out["strokes"] = kept

    if args.json_out is None:
        root, ext = os.path.splitext(args.json_in)
        args.json_out = f"{root}_darkpruned{ext}"

    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=4)

    total = len(strokes)
    print(f"[dark-prune] total={total} kept={len(kept)} removed={removed} "
          f"(thresh={args.thresh}, mode={args.mode}, mul_alpha={args.mul_alpha})")
    print(f"[dark-prune] wrote: {args.json_out}")

if __name__ == "__main__":
    main()
