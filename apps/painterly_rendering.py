import pydiffvg
import torch
import skimage.io
import random
import ttools.modules
import argparse
import math
import json
from matplotlib.colors import to_hex

pydiffvg.set_print_timing(True)
gamma = 1.0

def main(args):
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())

    target = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0
    target = target.pow(gamma).to(pydiffvg.get_device()).unsqueeze(0).permute(0, 3, 1, 2)

    # Drop alpha channel if present
    if target.shape[1] == 4:
        target = target[:, :3, :, :]

    canvas_width, canvas_height = target.shape[3], target.shape[2]
    num_paths = args.num_paths
    max_width = args.max_width

    random.seed(1234)
    torch.manual_seed(1234)

    shapes = []
    shape_groups = []

    for i in range(num_paths):
        num_segments = random.randint(1, 3)
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        p0 = (random.random(), random.random())
        points.append(p0)

        for j in range(num_segments):
            radius = 0.05
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points += [p1, p2, p3]
            p0 = p3

        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height

        stroke_width = torch.tensor(random.random() * max_width)
        path = pydiffvg.Path(num_control_points=num_control_points,
                             points=points,
                             stroke_width=stroke_width,
                             is_closed=False)
        shapes.append(path)

        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1]),
            fill_color=None,
            stroke_color=torch.tensor([random.random(), random.random(), random.random(), random.random()])
        )
        shape_groups.append(group)

    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply

    points_vars = [path.points for path in shapes]
    color_vars = [group.stroke_color for group in shape_groups]

    for v in points_vars: v.requires_grad = True
    for c in color_vars: c.requires_grad = True

    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    for t in range(args.num_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        color_optim.zero_grad()

        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)

        img = img[:, :, 3:4] * img[:, :, :3] + (1 - img[:, :, 3:4])
        pydiffvg.imwrite(img.cpu(), f'results/empatica_flower/iter_{t}.png', gamma=gamma)

        img = img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2)

        if args.use_lpips_loss:
            loss = perception_loss(img, target) + (img.mean() - target.mean()).pow(2)
        else:
            loss = (img - target).pow(2).mean()

        print('render loss:', loss.item())
        loss.backward()
        points_optim.step()
        color_optim.step()

        for group in shape_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)

    # Final render
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    pydiffvg.imwrite(img.cpu(), 'results/empatica_flower/final.png', gamma=gamma)



    # Convert the intermediate renderings to a video.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        "results/empatica_flower/iter_%d.png", "-vb", "20M",
        "results/empatica_flower/out.mp4"])
    

    #CREATING JSONimport json
   # Define canvas scaling
    SCALE_WIDTH = 12.0
    SCALE_HEIGHT = 12.0
    X_OFFSET = 24.0
    Y_OFFSET = 12.0

    canvas_w = canvas_width
    canvas_h = canvas_height


    def convert_pixel_to_inches(x_pixel, y_pixel, img_width, img_height,
                            scale_width=12.0, scale_height=12.0,
                            x_offset=0.0, y_offset=0.0):
        # Normalize
        x_norm = x_pixel / img_width
        y_norm = y_pixel / img_height

        # # Flip Y
        y_norm = 1.0 - y_norm

        # Scale and offset
        x_inches = x_norm * scale_width - scale_width / 2 + x_offset
        y_inches = y_norm * scale_height - scale_height / 2 + y_offset
        return round(x_inches, 4), round(y_inches, 4)


    output_strokes = {"strokes": []}

    for path, group in zip(shapes, shape_groups):
        # Convert points to inches with offset
        pts = path.points.detach().cpu().numpy().tolist()
        scaled_pts = []
        for i, (x, y) in enumerate(pts):
            x_inches, y_inches = convert_pixel_to_inches(
                x, y, canvas_w, canvas_h,
                scale_width=SCALE_WIDTH, scale_height=SCALE_HEIGHT,
                x_offset=X_OFFSET, y_offset=Y_OFFSET
            )
            scaled_pts.append([x_inches, y_inches, 1.0])  # fixed pressure

        # Get color
        rgba = group.fill_color if group.fill_color is not None else group.stroke_color
        rgba = rgba.detach().cpu().numpy()
        # rgb = rgba[:3]
        # color_hex = to_hex(rgb)
        r, g, b, a = rgba
        # Composite over white
        composited_rgb = [a * c + (1 - a) * 1.0 for c in [r, g, b]]

        # Convert to hex
        color_hex = to_hex(composited_rgb)

        # Get stroke width
        width_px = path.stroke_width.item()
        width_in = (width_px / canvas_w) * SCALE_WIDTH

        stroke = {
            "color": color_hex,
            "brush": 0,
            "speed": 10.0,
            "points": scaled_pts, 
            "width":width_in
        }


        output_strokes["strokes"].append(stroke)

    # Save the file
    with open("results/empatica_flower/diffvg_strokes.json", "w") as f:
        json.dump(output_strokes, f, indent=2)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target image path")
    parser.add_argument("--num_paths", type=int, default=512)
    parser.add_argument("--max_width", type=float, default=2.0)
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--use_blob", dest='use_blob', action='store_true')
    args = parser.parse_args()
    main(args)
