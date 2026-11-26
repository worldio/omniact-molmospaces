"""Visualize a node-graph WODS trajectory — generic across embodiments.

Reads the precomputed ``node_geometry`` (per-view projected nodes) + ``node_graph``
(edges / axis_mask / node_names) + video straight from a WODS dataset and renders a panel strip

    row 1 (current pose):  [ graph-exhibit | view_1 | view_2 | ... ]
    row 2 (action target): [ graph-exhibit | view_1 | view_2 | ... ]

with the scene-graph overlaid on every camera view. When the dataset carries ``node_action_geometry``
(the action-TARGET node positions), a SECOND row is drawn directly BELOW the current one showing where
the executed action would move the nodes to — overlaid on the SAME current frames (no sim re-render).
When it also carries ``node_pose`` / ``node_action_pose`` (per-node SE(3) ``[R | t]``, the former in the
shard's camera frame, the latter in the world frame with ``c2w`` taking it into the view), each row draws the
node's pose as three short axes (x red, y green, z blue); nodes whose pose entry is zero draw none.
Camera-frame 3D geometry is pinhole-projected to pixels via the stored intrinsics. There is **no sim,
no FK, and no per-embodiment code** — the entire graph (which nodes, which are terminal, how they
connect, where they project) comes from the data, so *any* embodiment that carries it works.
Only the current camera-frame representation and embodiment-in-key layout are supported.
Styling mirrors ``ignore/2026-06-24/_graphviz.py`` (the same ``render_view``: extended viewport with
off-frame nodes in the margin, Liang-Barsky edge clipping, axis tips, labels prefixed by the node index).

The left "exhibit" panel is the node graph drawn ALONE on a dark canvas (the embodiment-agnostic
analog of the old sim-rendered robot portrait): it picks the view in which the most nodes are
in-frame so the skeleton reads cleanly.

  uv run scripts/visualize_node_graph.py \
      --data ignore/2026-06-19/wods_paired_fixed_with_embeds \
      --key PullCubeTool-v1_traj00 --emb droid --out ignore/nodegraph_traj00.mp4

  # auto-pick the first key/embodiment that has node_geometry:
  uv run scripts/visualize_node_graph.py --data <wods_dir>
"""
import argparse
import io
from pathlib import Path

import av
import fsspec
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wods.v2.item import list_wods_keys, list_wods_shards, read_wods_shard

from molmo_spaces.embodiment_gen.node_spec import JOINT_PRISMATIC, JOINT_REVOLUTE

CV_TO_WODS = np.diag([1.0, -1.0, 1.0])   # wods<->OpenCV camera flip (for camera-frame geom): only y, down<->up

EDGE, AX, REV, PRIS, TERM = (210, 210, 215), (110, 150, 255), (255, 235, 40), (60, 190, 255), (40, 255, 120)
POSE_AXES, POSE_LEN = ((255, 80, 80), (80, 255, 80), (30, 60, 255)), 0.05   # node pose x / y / z tips, metres


def project_cam_frame(pc, K, image_size):
    """(N, P, 3) wods camera-frame points + intrinsics K(3,3) -> (N, P, 3) normalized
    (x_norm, y_norm, depth): flip wods->OpenCV (CV_TO_WODS), pinhole-project, normalize x,y by image
    size (off-frame kept raw). depth = forward camera-z (>0 in front)."""
    pc_cv = np.asarray(pc) @ CV_TO_WODS
    height, width = image_size
    z = pc_cv[..., 2]
    zs = np.where(np.abs(z) > 1e-6, z, 1e-6)
    xn = (K[0, 0] * pc_cv[..., 0] / zs + K[0, 2]) / width
    yn = (K[1, 1] * pc_cv[..., 1] / zs + K[1, 2]) / height
    return np.stack([xn, yn, z], axis=-1)
_FONT_CACHE = {}


def get_font(size):
    if size not in _FONT_CACHE:
        import matplotlib.font_manager as fm
        f = None
        for cand in (fm.findfont("DejaVu Sans"), "DejaVuSans.ttf"):
            try:
                f = ImageFont.truetype(cand, size); break
            except Exception:
                pass
        _FONT_CACHE[size] = f or ImageFont.load_default()
    return _FONT_CACHE[size]


def render_view(frame, proj, edges, types, names, W, H, scale=4, margin=0.7, limb=False, title=None, poses=None):
    """proj: per node (center_norm(x,y,depth), axistip_norm or None). Returns HxW3 array.
    ``poses`` (optional): per node its projected pose points (origin, x tip, y tip, z tip).
    margin is (horizontal, vertical) fractions, or a scalar — horizontal kept small so panels sit
    close together, vertical larger to hold off-frame joints. (Verbatim from _graphviz.py.)
    ``title`` (optional) is drawn small in the top margin."""
    mx, my = margin if isinstance(margin, (tuple, list)) else (margin, margin)
    cw, ch = int((1 + 2 * mx) * W * scale), int((1 + 2 * my) * H * scale)
    canvas = Image.new("RGB", (cw, ch), (26, 26, 30))
    ox, oy = int(mx * W * scale), int(my * H * scale)
    if frame is not None:
        canvas.paste(Image.fromarray(frame).resize((W * scale, H * scale), Image.NEAREST), (ox, oy))
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((ox + 2, 1), title, fill=(235, 235, 240), font=get_font(max(9, int(7 * scale))))
    d.rectangle([ox, oy, ox + W * scale, oy + H * scale], outline=(125, 125, 130), width=max(1, scale))
    font = get_font(int(8.5 * scale))

    def c2(p):
        return ((mx + p[0]) * W * scale, (my + p[1]) * H * scale)

    def vis(p):
        return p is not None and p[2] > 0 and -mx <= p[0] <= 1 + mx and -my <= p[1] <= 1 + my

    def clip(pa, pb):                                         # Liang-Barsky clip of segment to the canvas
        x0, y0 = c2(pa); x1, y1 = c2(pb); dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, x0), (dx, cw - x0), (-dy, y0), (dy, ch - y0)):
            if p == 0:
                if q < 0:
                    return None
            else:
                r = q / p
                if p < 0:
                    if r > t1:
                        return None
                    t0 = max(t0, r)
                else:
                    if r < t0:
                        return None
                    t1 = min(t1, r)
        return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)

    lw = max(2, int(round(1.5 * scale)))
    for a, b in edges:                                        # edges drawn even toward off-frame nodes (clipped)
        pa, pb = proj[a][0], proj[b][0]
        if pa[2] > 0 and pb[2] > 0 and (seg := clip(pa, pb)) is not None:
            if limb:                                          # stretched "metal limb" texture
                d.line(seg, fill=(95, 97, 105), width=lw * 5)
                d.line(seg, fill=(175, 178, 188), width=lw * 3)
                d.line(seg, fill=(225, 228, 236), width=lw)
            else:
                d.line(seg, fill=EDGE, width=lw)
    for n in range(len(proj)):
        c, at = proj[n]
        if vis(c) and at is not None and at[2] > 0 and (seg := clip(c, at)) is not None:
            d.line(seg, fill=AX, width=lw)
    for pts in poses or ():
        for tip, col in zip(pts[1:], POSE_AXES):
            if vis(pts[0]) and tip[2] > 0 and (seg := clip(pts[0], tip)) is not None:
                d.line(seg, fill=col, width=lw)
    for n in range(len(proj)):
        c = proj[n][0]
        if not vis(c):
            continue
        u = c2(c); term = types[n] == "term"; col = TERM if term else (PRIS if types[n] == "pris" else REV)
        r = scale * (1.5 if term else 1.35)
        d.ellipse([u[0] - r, u[1] - r, u[0] + r, u[1] + r], fill=col)
        d.text((u[0] + r + 2, u[1] - r - 2 - (n % 2) * int(2.6 * scale)), f"{n}:{names[n]}", fill=col, font=font)
    return np.asarray(canvas)


def decode(raw):
    c = av.open(io.BytesIO(raw), format="mp4")
    fs = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]; c.close(); return fs


def discover_shards(data, key):
    """Return ``{emb: {view: [physical shards]}}`` for current unchunked or chunked WODS."""
    if "/" not in key:
        return {}
    fs, _, paths = fsspec.get_fs_token_paths(data)
    out = {}
    for shard in list_wods_shards(data, key):
        if not fs.exists(f"{paths[0].rstrip('/')}/{key}/{shard}.node_geometry.pt"):
            continue
        parts = shard.split("/")
        if parts[-1].startswith("frame_"):
            parts.pop()
        emb, view = key.rsplit("/", 1)[1], "/".join(parts)
        out.setdefault(emb, {}).setdefault(view, []).append(shard)
    return {emb: {view: sorted(chunks) for view, chunks in sorted(views.items())}
            for emb, views in sorted(out.items())}


def proj_at(geom_v, t, axis_mask):
    """geom_v (T,N,P,3) -> list per node of (center(x,y,depth), axistip or None-if-terminal)."""
    g = geom_v[t]
    return [(tuple(float(x) for x in g[n, 0]),
             tuple(float(x) for x in g[n, 1]) if bool(axis_mask[n]) else None)
            for n in range(g.shape[0])]


def pose_points(P, C=None):
    """(T, N, 3, 4) SE(3) poses -> (T, N, 4, 3) camera-frame points (origin, x tip, y tip, z tip); a zero pose
    gives NaNs, which never draw. ``C`` (T, 3, 4) c2w takes a WORLD track into this view's camera frame."""
    M, o = P[..., :3, :3], P[..., :3, 3:]
    pts = np.concatenate([o, o + POSE_LEN * M], axis=-1).swapaxes(-1, -2)
    pts = np.where(np.any(M != 0, axis=(-1, -2))[..., None, None], pts, np.nan)
    return pts if C is None else (pts - C[:, None, None, :3, 3]) @ C[:, None, :3, :3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="WODS dataset dir (local or gs://)")
    ap.add_argument("--key", default=None, help="episode key (default: first key with node_geometry)")
    ap.add_argument("--emb", default=None, help="embodiment subdir (default: first with node_geometry)")
    ap.add_argument("--views", nargs="+", default=None, help="view names (default: all with node_geometry)")
    ap.add_argument("--exhibit-view", type=int, default=None,
                    help="index into --views for the left skeleton panel (default: most-visible view)")
    ap.add_argument("--out", default=None, help="output .mp4 (default: ignore/nodegraph_<key>_<emb>.mp4)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--scale", type=int, default=2, help="upsample factor for the video (use 4 for the still)")
    ap.add_argument("--still", action="store_true", help="also write a mid-trajectory PNG still")
    args = ap.parse_args()

    key = args.key
    if key is None:                                          # first key that has any node_geometry
        for k in sorted(list_wods_keys(args.data)):
            found = discover_shards(args.data, k)
            if found and (args.emb is None or args.emb in found):
                key = k; break
        if key is None:
            raise SystemExit(f"No key with node_geometry under {args.data}")
    shards = discover_shards(args.data, key)
    if not shards and args.emb is not None and "/" not in key:
        candidate = f"{key}/{args.emb}"
        shards = discover_shards(args.data, candidate)
        if shards:
            key = candidate
    if not shards:
        raise SystemExit(f"{key!r} has no node_geometry — run add_transfer_queries_to_wods.py first")
    emb = args.emb or sorted(shards)[0]
    if emb not in shards:
        raise SystemExit(f"emb {emb!r} has no node_geometry for {key}; available: {sorted(shards)}")
    views = args.views or sorted(shards[emb])

    geom, ageom, pose, apose, Kmap, Cmap, frames, graph = {}, {}, {}, {}, {}, {}, {}, None
    for v in views:
        if v not in shards[emb]:
            raise SystemExit(f"view {v!r} not found; available: {sorted(shards[emb])}")
        gparts, agparts, pparts, apparts, kparts, cparts, fparts = [], [], [], [], [], [], []
        complete_action = complete_pose = True
        for physical_shard in shards[emb][v]:
            r = read_wods_shard(
                args.data, key, physical_shard, include_video=False, include_video_raw=True,
                include_timestamps=False, include_latents=False, include_camera=True,
                include_points=False, include_depth=False, include_mask=False, include_proprio=False,
                include_action=False, include_target=False, include_text_embed=False,
                include_node_geometry=True, include_node_action_geometry=True,
                include_node_graph=True, include_node_proprio=False,
                include_node_pose=True, include_node_action_pose=True,
            )
            if "node_geometry" not in r or "video_raw" not in r:
                raise SystemExit(f"incomplete shard {key}/{physical_shard}: needs video and node_geometry")
            gp = np.asarray(r["node_geometry"])
            gparts.append(gp)
            fparts.extend(decode(r["video_raw"]))
            if "node_action_geometry" in r:
                agparts.append(np.asarray(r["node_action_geometry"]))
            else:
                complete_action = False
            if all(f in r for f in ("node_pose", "node_action_pose", "c2w")):
                pparts.append(np.asarray(r["node_pose"])); apparts.append(np.asarray(r["node_action_pose"]))
                cparts.append(np.asarray(r["c2w"]))
            else:
                complete_pose = False
            if "intrinsics" in r:
                K = np.asarray(r["intrinsics"])
                kparts.append(np.repeat(K[None], len(gp), axis=0) if K.ndim == 2 else K)
            if graph is None:
                graph = r.get("node_graph")
        geom[v] = np.concatenate(gparts)
        if complete_action:
            ageom[v] = np.concatenate(agparts)
        if complete_pose:
            pose[v], apose[v], Cmap[v] = map(np.concatenate, (pparts, apparts, cparts))
        if kparts:
            Kmap[v] = np.concatenate(kparts)
        frames[v] = fparts
    views = [v for v in views if v in geom]
    if not views or graph is None:
        raise SystemExit(f"no usable views for {key}/{emb}")
    has_action = bool(ageom) and all(v in ageom for v in views)
    has_pose = bool(pose) and all(v in pose for v in views)

    T = min(min(g.shape[0] for g in geom.values()), min(len(frames[v]) for v in views))
    geom = {v: g[:T] for v, g in geom.items()}
    frames = {v: f[:T] for v, f in frames.items()}
    if has_action:
        ageom = {v: g[:T] for v, g in ageom.items()}
    if not all(v in Kmap and len(Kmap[v]) >= T for v in views):
        raise SystemExit("camera-frame node geometry needs per-frame intrinsics")

    def project_all(values):
        return {v: np.stack([project_cam_frame(values[v][t], Kmap[v][t], frames[v][t].shape[:2])
                             for t in range(T)]) for v in views}

    geom = project_all(geom)
    if has_action:
        ageom = project_all(ageom)
    if has_pose:
        pose = project_all({v: pose_points(pose[v][:T]) for v in views})
        apose = project_all({v: pose_points(apose[v][:T], Cmap[v][:T]) for v in views})

    edges = [tuple(int(x) for x in e) for e in np.asarray(graph["edges"])]
    axis_mask = np.asarray(graph["axis_mask"]).astype(bool)
    joint_type = np.asarray(graph["joint_type"]).astype(np.int64)
    names = list(graph["node_names"])
    types = [("pris" if joint_type[n] == JOINT_PRISMATIC else
              ("rev" if joint_type[n] == JOINT_REVOLUTE else "term")) for n in range(len(names))]

    H0, W0 = frames[views[0]][0].shape[:2]

    if args.exhibit_view is not None:
        exv = views[args.exhibit_view]
    else:                                                    # view with the most in-frame node centers
        def vis_frac(v):
            c = geom[v][:T, :, 0]
            return float(((c[..., 2] > 0) & (c[..., 0] >= 0) & (c[..., 0] <= 1)
                          & (c[..., 1] >= 0) & (c[..., 1] <= 1)).mean())
        exv = max(views, key=vis_frac)

    def render_t(tt, scale, margin):
        def hcat(ps):
            hh = min(p.shape[0] for p in ps)
            return np.concatenate([p[:hh] for p in ps], axis=1)

        def row(G, suffix, P=None):   # [exhibit | view_1 | view_2 | ...] for the geometry dict G (+ pose dict P)
            def poses(v):
                return [tuple(map(tuple, pts)) for pts in P[v][tt]] if P is not None else None
            pan = [render_view(None, proj_at(G[exv], tt, axis_mask), edges, types, names, W0, H0, scale, margin,
                               title="skeleton" + suffix, poses=poses(exv))]
            pan += [render_view(frames[v][tt], proj_at(G[v], tt, axis_mask), edges, types, names,
                                W0, H0, scale, margin, title=v + suffix, poses=poses(v)) for v in views]
            return hcat(pan)

        comp = row(geom, "", pose if has_pose else None)
        if has_action:        # the action-TARGET pose row, BELOW the current row (same frames, no sim)
            r2 = row(ageom, ": action target", apose if has_pose else None)
            ww = min(comp.shape[1], r2.shape[1])
            comp = np.concatenate([comp[:, :ww], r2[:, :ww]], axis=0)
        return comp[:comp.shape[0] // 2 * 2, :comp.shape[1] // 2 * 2]   # even dims for ffmpeg

    out = Path(args.out or f"ignore/nodegraph_{key.replace('/', '_')}_{emb}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.still:
        imageio.imwrite(out.with_suffix(".png"), render_t(T // 2, 4, (0.15, 0.3)))
    writer = imageio.get_writer(out, fps=args.fps, macro_block_size=1)
    try:
        for tt in range(T):
            writer.append_data(render_t(tt, args.scale, (0.1, 0.2)))
    finally:
        writer.close()
    print(f"WROTE {out.resolve()}  ({T} frames, {len(names)} nodes, views={views}, exhibit={exv}"
          f"{', +action-target panels' if has_action else ''}{', +node poses' if has_pose else ''})", flush=True)


if __name__ == "__main__":
    main()
