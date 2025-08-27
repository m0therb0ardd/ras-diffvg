"""
Scream: python painterly_rendering.py imgs/scream.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0 --use_lpips_loss
Baboon: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 250
Baboon Lpips: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 500 --use_lpips_loss
Kitty: python painterly_rendering.py imgs/kitty.jpg --num_paths 1024 --use_blob
"""
import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse

from datetime import datetime
import pickle
import os
import sys
import json
import numpy as np

pydiffvg.set_print_timing(True)

gamma = 1.0

def main(args):
    dir_name = datetime.now().strftime("%Y%m%d_%H%M%S-") + args.target.split('/')[-1].split('.')[0]
    os.makedirs(f'results/{dir_name}', exist_ok=True)

    # Save command used to run the script
    with open(f'results/{dir_name}/command.txt', 'w') as f:
        f.write('python ' + ' '.join(sys.argv))

    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    
    perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())
    
    target = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2) # NHWC -> NCHW
    # Ensure target has only 3 channels (RGB) to match the rendered image
    if target.shape[1] == 4:  # If RGBA
        target = target[:, :3, :, :]  # Remove alpha channel
    #target = torch.nn.functional.interpolate(target, size = [256, 256], mode = 'area')
    canvas_width, canvas_height = target.shape[3], target.shape[2]
    num_paths = args.num_paths
    max_width = args.max_width
    random.seed(1234)
    torch.manual_seed(1234)


    # === Color-only init mask ===
    tgt_rgb = target[0, :3]                          # 3 x H x W
    brightness = tgt_rgb.max(dim=0).values           # H x W in [0,1]
    color_mask = brightness > args.black_thresh      # True where not black

    # Precompute coordinates of allowed pixels
    ys, xs = torch.where(color_mask)                 # GPU tensors are fine
    allowed_xy = torch.stack([xs, ys], dim=1)        # N x 2 (x,y)

    def sample_uv_from_color():
        # Fallback to uniform if no colored pixels (edge case)
        if allowed_xy.numel() == 0:
            return (random.random(), random.random())
        i = random.randrange(allowed_xy.shape[0])
        x, y = allowed_xy[i]
        # jitter inside the pixel & normalize to [0,1]
        u = (float(x) + random.random()) / float(canvas_width)
        v = (float(y) + random.random()) / float(canvas_height)
        # clamp for safety
        u = min(max(u, 0.0), 1.0 - 1e-6)
        v = min(max(v, 0.0), 1.0 - 1e-6)
        return (u, v)


    shapes = []
    shape_groups = []



    if args.use_blob:
        for i in range(num_paths):
            num_segments = random.randint(3, 5)
            num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
            points = []
            # p0 = (random.random(), random.random())
            p0 = sample_uv_from_color() if args.init_on_color else (random.random(), random.random())

            points.append(p0)
            for j in range(num_segments):
                radius = args.radius
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                if j < num_segments - 1:
                    points.append(p3)
                    p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            path = pydiffvg.Path(num_control_points = num_control_points,
                                 points = points,
                                 stroke_width = torch.tensor(1.0),
                                 is_closed = True)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                             fill_color = torch.tensor([random.random(),
                                                                        random.random(),
                                                                        random.random(),
                                                                        random.random()]))
            shape_groups.append(path_group)
    else:
        for i in range(num_paths):
            num_segments = random.randint(1, 3)
            num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
            points = []
            # p0 = (random.random(), random.random())
            p0 = sample_uv_from_color() if args.init_on_color else (random.random(), random.random())

            #dont need right now 
            # if args.use_black_bg:
            #     while (target[0, :, int(p0[1] * canvas_height), int(p0[0] * canvas_width)] == 0.).all():
            #         p0 = (random.random(), random.random())
            # else:
            #     while (target[0, :, int(p0[1] * canvas_height), int(p0[0] * canvas_width)] == 1.).all():
            #         p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = args.radius
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            #points = torch.rand(3 * num_segments + 1, 2) * min(canvas_width, canvas_height)
            path = pydiffvg.Path(num_control_points = num_control_points,
                                 points = points,
                                 stroke_width = torch.tensor(1.0),
                                 is_closed = False)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                             fill_color = None,
                                             stroke_color = torch.tensor([random.random(),
                                                                          random.random(),
                                                                          random.random(),
                                                                          random.random() if not args.ignore_alpha else 1.0]))
            shape_groups.append(path_group)
    
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    pydiffvg.imwrite(img.cpu(), f'results/{dir_name}/init.png', gamma=gamma)

    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    if not args.use_blob:
        for path in shapes:
            path.stroke_width.requires_grad = True
            stroke_width_vars.append(path.stroke_width)
    if args.use_blob:
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)
    else:
        for group in shape_groups:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)
    
    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    if len(stroke_width_vars) > 0:
        width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)
    # Adam iterations.
    for t in range(args.num_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        if len(stroke_width_vars) > 0:
            width_optim.zero_grad()
        color_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, # width
                     canvas_height, # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     t,   # seed
                     None,
                     *scene_args)
        # Compose img with white background
        # if args.use_black_bg:
        #     img = img[:, :, 3:4] * img[:, :, :3] + torch.zeros(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        # else:
        #     img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        # Compose img with chosen background
        # Compose img with chosen background (inline hex parsing, no helper needed)
        alpha = img[:, :, 3:4]
        rgb   = img[:, :, :3]

        if args.bg_color is not None:
            h = args.bg_color.lstrip("#")
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            if len(h) != 6:
                raise ValueError(f"Bad hex color: {args.bg_color}")
            r = int(h[0:2], 16) / 255.0
            g = int(h[2:4], 16) / 255.0
            b = int(h[4:6], 16) / 255.0
            bg = torch.tensor([r, g, b], device=pydiffvg.get_device(), dtype=img.dtype).view(1, 1, 3)
            bg = bg.expand(img.shape[0], img.shape[1], 3)
        else:
            if args.use_black_bg:
                bg = torch.zeros(img.shape[0], img.shape[1], 3, device=pydiffvg.get_device(), dtype=img.dtype)
            else:
                bg = torch.ones (img.shape[0], img.shape[1], 3, device=pydiffvg.get_device(), dtype=img.dtype)

        img = alpha * rgb + (1 - alpha) * bg


        
        # === Whitespace helpers ===
        # bg = torch.zeros_like(img) if args.use_black_bg else torch.ones_like(img)  # HxWx3
        # # Per-pixel distance from background (L2 across channels), shape: HxW
        # bg_dist = torch.norm(img - bg, dim=2)
        # # Background-like pixels ~1, inked pixels ~0
        # whitespace_field = torch.exp(-args.whitespace_k * bg_dist)
        # whitespace_loss = whitespace_field.mean()
        # === Whitespace helpers (reuse the SAME bg you composed with) ===
        bg_dist = torch.norm(img - bg, dim=2)     # HxW
        whitespace_field = torch.exp(-args.whitespace_k * bg_dist)
        whitespace_loss = whitespace_field.mean()


                
        
        
        # Save the intermediate render.
        # pydiffvg.imwrite(img.cpu(), f'results/{dir_name}/iter_{t}.png', gamma=gamma)
        pydiffvg.imwrite(img.cpu(), f'results/{dir_name}/latest.png', gamma=gamma)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        # if args.use_lpips_loss:
        #     loss = perception_loss(img, target) + (img.mean() - target.mean()).pow(2)
        # else:
        #     loss = (img - target).pow(2).mean()
        # print('render loss:', loss.item())

        # === Reconstruction loss (unchanged) ===
        if args.use_lpips_loss:
            recon_loss = perception_loss(img, target) + (img.mean() - target.mean()).pow(2)
        else:
            recon_loss = (img - target).pow(2).mean()

        # === Total loss with whitespace penalty ===
        loss = recon_loss + args.lambda_whitespace * whitespace_loss
        print(f"loss: {loss.item():.6f} | recon: {recon_loss.item():.6f} | white: {whitespace_loss.item():.6f}")

    
        # Backpropagate the gradients.
        loss.backward()

        # Zero out alpha gradients for color variables
        if args.ignore_alpha:
            for color_var in color_vars:
                if color_var.grad is not None:
                    color_var.grad[..., 3] = 0.0

        # Take a gradient descent step.
        points_optim.step()
        if len(stroke_width_vars) > 0:
            width_optim.step()
        color_optim.step()
        if len(stroke_width_vars) > 0:
            for path in shapes:
                path.stroke_width.data.clamp_(1.0, max_width)
        if args.use_blob:
            for group in shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)
        else:
            for group in shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)

        if t % 10 == 0 or t == args.num_iter - 1:
            pydiffvg.save_svg(f'results/{dir_name}/iter_{t}.svg',
                              canvas_width, canvas_height, shapes, shape_groups)
            # Save shapes and shape groups using pickle
            with open(f'results/{dir_name}/shapes.pkl', 'wb') as f:
                pickle.dump(shapes, f)
            with open(f'results/{dir_name}/shape_groups.pkl', 'wb') as f:
                pickle.dump(shape_groups, f)


        # ------------------------
    # after the training loop:
    # ------------------------

    # === Post-pass: fill uncovered neon areas with one stroke per blob ===
    if args.fill_whitespace:
        from skimage.measure import label, regionprops

        # 1) Render with current shapes
        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
        img_rgba = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)

        # 2) Compose on your chosen background (neon if provided)
        alpha = img_rgba[:, :, 3:4]
        rgb   = img_rgba[:, :, :3]
        if args.bg_color is not None:
            h = args.bg_color.lstrip("#")
            if len(h) == 3: h = "".join(c*2 for c in h)
            r = int(h[0:2], 16) / 255.0
            g = int(h[2:4], 16) / 255.0
            b = int(h[4:6], 16) / 255.0
            bg = torch.tensor([r, g, b], device=pydiffvg.get_device(), dtype=img_rgba.dtype).view(1,1,3).expand(img_rgba.shape[0], img_rgba.shape[1], 3)
        else:
            bg = torch.zeros_like(rgb) if args.use_black_bg else torch.ones_like(rgb)

        comp = alpha * rgb + (1 - alpha) * bg  # HxWx3

        # 3) Uncovered mask = pixels near the neon bg
        tau = 0.03
        d_bg = torch.norm(comp - bg, dim=2)           # HxW
        mask = (d_bg < tau).detach().cpu().numpy()    # bool

        # 4) Connected blobs and their centroids
        lab = label(mask)
        props = regionprops(lab)
        props = sorted(props, key=lambda p: p.area, reverse=True)[:200]  # cap

        H, W = comp.shape[0], comp.shape[1]
        L = 0.02 * float(min(H, W))  # short initial length

        new_points, new_widths, new_colors = [], [], []

        for p in props:
            yc, xc = p.centroid
            y = int(round(yc)); x = int(round(xc))

            # short cubic, random direction
            p0 = torch.tensor([float(x), float(y)], device=pydiffvg.get_device())
            d  = (torch.rand(2, device=pydiffvg.get_device()) - 0.5) * 2.0
            d  = d / (torch.norm(d) + 1e-8) * L
            p1 = p0 + 0.33 * d
            p2 = p0 + 0.66 * d
            p3 = p0 + d
            points = torch.stack([p0, p1, p2, p3], dim=0)

            num_control_points = torch.tensor([2], dtype=torch.int32)
            stroke_width = torch.tensor(
                random.uniform(1.0, max_width),   # use 1.0 or getattr(args, "min_width", 1.0)
                device=pydiffvg.get_device(), dtype=torch.float32
            )

            # color from target at (y,x)
            tc = target[0, :3, y, x].detach()
            a  = torch.tensor([1.0], device=pydiffvg.get_device()) if args.ignore_alpha else torch.rand(1, device=pydiffvg.get_device())
            rgba = torch.cat([tc, a]).to(torch.float32)

            path = pydiffvg.Path(num_control_points=num_control_points, points=points,
                                 stroke_width=stroke_width, is_closed=False)
            shapes.append(path)
            group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                        fill_color=None, stroke_color=rgba)
            shape_groups.append(group)

            path.points.requires_grad = True
            path.stroke_width.requires_grad = True
            group.stroke_color.requires_grad = True
            new_points.append(path.points)
            new_widths.append(path.stroke_width)
            new_colors.append(group.stroke_color)

        print(f"[fill_whitespace] added {len(new_points)} strokes")

        if len(new_points) > 0:
            opt_p = torch.optim.Adam(new_points, lr=1.0)
            opt_w = torch.optim.Adam(new_widths, lr=0.1) if len(new_widths) > 0 else None
            opt_c = torch.optim.Adam(new_colors, lr=0.01)

            for k in range(60):
                opt_p.zero_grad()
                if opt_w: opt_w.zero_grad()
                opt_c.zero_grad()

                scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
                img_k = render(canvas_width, canvas_height, 2, 2, k, None, *scene_args)
                a_k, rgb_k = img_k[:, :, 3:4], img_k[:, :, :3]
                comp_k = a_k * rgb_k + (1 - a_k) * bg
                comp_k = comp_k.unsqueeze(0).permute(0,3,1,2)

                if args.use_lpips_loss:
                    loss_k = perception_loss(comp_k, target) + (comp_k.mean() - target.mean()).pow(2)
                else:
                    loss_k = (comp_k - target).pow(2).mean()

                loss_k.backward()
                if args.ignore_alpha:
                    for c in new_colors:
                        if c.grad is not None:
                            c.grad[..., 3] = 0.0

                opt_p.step()
                if opt_w: opt_w.step()
                opt_c.step()

                # # keep widths in bounds
                # for pth in shapes:
                #     pth.stroke_width.data.clamp_(1.0, max_width)

                # keep widths in bounds
                for pth in shapes:
                    pth.stroke_width.data.clamp_(1.0, max_width)

                # NEW: clamp colors into [0,1]; force alpha=1 if ignore_alpha
                for grp in shape_groups:
                    if hasattr(grp, "stroke_color") and grp.stroke_color is not None:
                        grp.stroke_color.data.clamp_(0.0, 1.0)
                        if args.ignore_alpha:
                            grp.stroke_color.data[..., 3] = 1.0


    # === Final render (always do this) ===
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    pydiffvg.imwrite(img.cpu(), f'results/{dir_name}/final.png', gamma=gamma)


    # # === Final render (always do this) ===
    # scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    # # img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    # # pydiffvg.imwrite(img.cpu(), f'results/{dir_name}/final.png', gamma=gamma)




    # # Render the final result.
    # img = render(target.shape[1], # width
    #              target.shape[0], # height
    #              2,   # num_samples_x
    #              2,   # num_samples_y
    #              0,   # seed
    #              None,
    #              *scene_args)
    # # Save the intermediate render.
    # pydiffvg.imwrite(img.cpu(), f'results/{dir_name}/final.png', gamma=gamma)
    # Convert the intermediate renderings to a video.
    # from subprocess import call
    # call(["ffmpeg", "-framerate", "24", "-i",
    #     f"results/{dir_name}/iter_%d.png", "-vb", "20M",
    #     f"results/{dir_name}/out.mp4"])

    # Convert strokes to json
    def bezier_curve(P0, P1, P2, n_points=5):
        """Quadratic Bezier curve."""
        t = np.linspace(0, 1, n_points).reshape(-1, 1)
        return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2

    def cubic_bezier_curve(P0, P1, P2, P3, n_points=5):
        """Cubic Bezier curve."""
        t = np.linspace(0, 1, n_points).reshape(-1, 1)
        return (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3
    
    def parse_hex_color(h: str):
        h = h.lstrip('#')
        if len(h) == 3:  # e.g. #f0a
            h = ''.join(c*2 for c in h)
        if len(h) != 6:
            raise ValueError(f"Bad hex color: {h}")
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        return r, g, b


    json_strokes = []
    for i in range(len(shapes)):
        s = shapes[i].points.detach().cpu().numpy()
        num_control_points = shapes[i].num_control_points.detach().cpu().numpy()
        pts, idx = [], 0
        for n in num_control_points: 
            if n == 0:
                p0, p1 = s[idx:idx+2]
                pts.append(np.stack([p0, p1]))
                idx += 1
            elif n == 1:
                p0, p1, p2 = s[idx:idx+3]
                pts.append(bezier_curve(p0, p1, p2))
                idx += 2
            elif n == 2:
                p0, p1, p2, p3 = s[idx:idx+4]
                pts.append(cubic_bezier_curve(p0, p1, p2, p3))
                idx += 3

        pts = np.concatenate(pts, axis=0)
        color = shape_groups[i].stroke_color.detach().cpu().numpy()
        color = np.clip(color, 0.0, 1.0)
        stroke_width = shapes[i].stroke_width.detach().cpu().numpy()
        json_strokes.append({
            "points": pts.tolist(),
            "color": color.tolist(),
            "stroke_width": float(stroke_width),
        })

    with open(f'results/{dir_name}/{dir_name.split("-")[1]}_diffvg.json', 'w') as f:
        json.dump({"strokes": json_strokes}, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target image path")
    parser.add_argument("--num_paths", type=int, default=512)
    parser.add_argument("--max_width", type=float, default=2.0)
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--use_blob", dest='use_blob', action='store_true')
    parser.add_argument("--ignore_alpha", dest='ignore_alpha', action='store_true')
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--use_black_bg", dest='use_black_bg', action='store_true')
    ## new 
    parser.add_argument("--lambda_whitespace", type=float, default=0.0,
                    help="Strength of whitespace penalty (0 disables)")
    parser.add_argument("--whitespace_k", type=float, default=12.0,
                    help="Sharpness for whitespace penalty (higher -> counts only near-exact background as whitespace)")
    parser.add_argument("--bg_color", type=str, default=None,
        help="Hex RGB like '#ff1493'. If set, overrides white/black background.")

    parser.add_argument("--fill_whitespace", action="store_true",
        help="After main training, add one short stroke per neon-colored hole and briefly refine")
    
    parser.add_argument("--init_on_color", action="store_true",
    help="Initialize stroke start points only on non-black pixels of the target")
    parser.add_argument("--black_thresh", type=float, default=0.05,
        help="Brightness threshold (0..1). Pixels <= this are treated as black for initialization")


    args = parser.parse_args()
    main(args)
