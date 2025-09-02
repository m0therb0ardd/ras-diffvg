# #!/usr/bin/env python3
# """
# Filter out strokes with stroke_width > THRESH from a DiffVG JSON file.
# Writes a new JSON file with only the kept strokes.
# """

# import json
# import argparse

# def main():
#     ap = argparse.ArgumentParser(description="Delete strokes wider than a threshold")
#     ap.add_argument("input", help="Path to input *_diffvg.json")
#     ap.add_argument("output", help="Path to save filtered JSON")
#     ap.add_argument("--thresh", type=float, default=4.2, help="Width threshold (keep <= this)")
#     args = ap.parse_args()

#     with open(args.input, "r") as f:
#         data = json.load(f)

#     strokes = data.get("strokes", [])
#     kept = [s for s in strokes if float(s.get("stroke_width", 0)) <= args.thresh]
#     dropped = len(strokes) - len(kept)

#     data["strokes"] = kept

#     with open(args.output, "w") as f:
#         json.dump(data, f, indent=4)

#     print(f"Filtered {args.input}")
#     print(f"Kept {len(kept)} strokes, dropped {dropped} (> {args.thresh})")
#     print(f"Saved to {args.output}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Filter out strokes based on width and/or y-coordinate from a DiffVG JSON file.
"""

import json
import argparse

def main():
    ap = argparse.ArgumentParser(description="Delete strokes wider than a threshold or above a y cutoff")
    ap.add_argument("input", help="Path to input *_diffvg.json")
    ap.add_argument("output", help="Path to save filtered JSON")
    ap.add_argument("--thresh", type=float, default=4.2, help="Width threshold (keep <= this)")
    ap.add_argument("--ymax", type=float, default=None, help="Drop strokes whose ANY point has y > this value")
    args = ap.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    strokes = data.get("strokes", [])
    kept = []
    dropped = 0

    for s in strokes:
        w = float(s.get("stroke_width", s.get("width", 0)))
        pts = s.get("points", [])
        too_wide = w > args.thresh
        too_high = args.ymax is not None and any(float(p[1]) > args.ymax for p in pts)

        if not too_wide and not too_high:
            kept.append(s)
        else:
            dropped += 1

    data["strokes"] = kept

    with open(args.output, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Filtered {args.input}")
    print(f"Kept {len(kept)} strokes, dropped {dropped} (width>{args.thresh} or y>{args.ymax})")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()












































































































































































































