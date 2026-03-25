# visualization/plot_3d_global.py
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use("Agg")  # must be called before import pyplot

import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import imageio


def plot_3d_motion(
    args,
    figsize=(6, 6),
    fps=20,
    dpi=152,                 # 6*152=912 => divisible by 16 (avoid imageio resize warning)
    elev=110,
    azim=-90,
    linewidth_root=3.0,
    linewidth_limb=2.5,
    fixed_camera=True,       # True: camera stays fixed (global range); False: adaptive per frame
    half_scale=0.55,         # smaller value = closer camera; recommended 0.45~0.70
    draw_traj=True,          # whether to draw root trajectory
    draw_ground=False,       # whether to draw ground plane
):
    """
    args: [joints, out_name, title]
      - joints: (T, J, 3) numpy array
      - out_name: optional (kept for compatibility; this implementation does not depend on it)
      - title: optional, e.g., ["line1", "line2"] or None

    return:
      out: (T, H, W, 3) uint8 RGB frames
    """
    joints, out_name, title = args

    # joints expected (T, J, 3)
    data = np.asarray(joints).copy()
    if not (data.ndim == 3 and data.shape[2] == 3):
        raise ValueError(f"Expected joints shape (T,J,3), got {data.shape}")
    T, J, _ = data.shape

    # Skeleton chains (same as your original)
    smpl_kinetic_chain = (
        [[0, 11, 12, 13, 14, 15],
         [0, 16, 17, 18, 19, 20],
         [0, 1, 2, 3, 4],
         [3, 5, 6, 7],
         [3, 8, 9, 10]]
        if J == 21 else
        [[0, 2, 5, 8, 11],
         [0, 1, 4, 7, 10],
         [0, 3, 6, 9, 12, 15],
         [9, 14, 17, 19, 21],
         [9, 13, 16, 18, 20]]
    )

    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    # -------------------------
    # Normalize / center (same spirit as your previous)
    # -------------------------
    mins_all = data.min(axis=(0, 1))
    data[:, :, 1] -= mins_all[1]              # put on ground

    trajec = data[:, 0, [0, 2]].copy()        # root xz
    data[..., 0] -= data[:, 0:1, 0]           # root-center x
    data[..., 2] -= data[:, 0:1, 2]           # root-center z

    # -------------------------
    # Fixed camera limits (compute ONCE to avoid jitter)
    # -------------------------
    if fixed_camera:
        xyz_all = data.reshape(-1, 3)
        mn = xyz_all.min(axis=0)
        mx = xyz_all.max(axis=0)
        c = (mn + mx) / 2.0
        span = (mx - mn).max()
        if span < 1e-6:
            span = 1.0
        half = span * half_scale

        x0, x1 = c[0] - half, c[0] + half
        y0, y1 = c[1] - half, c[1] + half
        z0, z1 = c[2] - half, c[2] + half

    def _set_axes_limits(ax, frame_xyz):
        """Either fixed camera or per-frame camera."""
        nonlocal x0, x1, y0, y1, z0, z1

        if fixed_camera:
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_zlim(z0, z1)
        else:
            mn = frame_xyz.min(axis=0)
            mx = frame_xyz.max(axis=0)
            c = (mn + mx) / 2.0
            span = (mx - mn).max()
            if span < 1e-6:
                span = 1.0
            half = span * half_scale
            ax.set_xlim(c[0] - half, c[0] + half)
            ax.set_ylim(c[1] - half, c[1] + half)
            ax.set_zlim(c[2] - half, c[2] + half)

        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    def _draw_ground_plane(ax, frame_xyz):
        # plane in XZ at y=0
        mn = frame_xyz.min(axis=0)
        mx = frame_xyz.max(axis=0)
        minx, maxx = mn[0], mx[0]
        minz, maxz = mn[2], mx[2]
        y = 0.0
        verts = [
            [minx, y, minz],
            [minx, y, maxz],
            [maxx, y, maxz],
            [maxx, y, minz],
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.15))
        xz_plane.set_edgecolor((0.5, 0.5, 0.5, 0.0))
        ax.add_collection3d(xz_plane)

    out_frames = []

    for t in range(T):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        frame = data[t]  # (J,3)

        # camera / limits
        _set_axes_limits(ax, frame)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

        # fill the canvas (remove large white margins)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_position([0, 0, 1, 1])

        # optional ground plane
        if draw_ground:
            _draw_ground_plane(ax, frame)

        # optional trajectory (root xz, y=0)
        if draw_traj and t > 1:
            ax.plot3D(
                trajec[:t, 0] - trajec[t, 0],
                np.zeros_like(trajec[:t, 0]),
                trajec[:t, 1] - trajec[t, 1],
                linewidth=1.0,
                color="blue",
            )

        # draw skeleton
        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors[:len(smpl_kinetic_chain)])):
            lw = linewidth_root if i < 5 else linewidth_limb
            ax.plot3D(
                frame[chain, 0],
                frame[chain, 1],
                frame[chain, 2],
                linewidth=lw,
                color=color,
            )

        # title (optional)
        if title is not None:
            tt0 = '\n'.join(wrap(str(title[0]), 80))
            tt1 = '\n'.join(wrap(str(title[1]), 80))
            fig.suptitle(tt0 + ("\n" + tt1 if tt1 else ""), fontsize=12)

        # render to numpy frame (RGBA -> RGB)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())  # (H,W,4) uint8
        rgb = rgba[..., :3].copy()                   # (H,W,3) uint8
        out_frames.append(rgb)

        plt.close(fig)

    out = np.stack(out_frames, axis=0).astype(np.uint8)  # (T,H,W,3)
    return out


def draw_to_batch(smpl_joints_batch, title_batch=None, outname=None, fps=20):
    """
    smpl_joints_batch: list/array of motions, each is (T,J,3)
    outname: optional list of output mp4 paths (same length as batch)
    return: list of frames arrays if outname is None; otherwise saves videos.
    """
    batch_size = len(smpl_joints_batch)
    outs = []
    for i in range(batch_size):
        title = title_batch[i] if title_batch is not None else None
        frames = plot_3d_motion([smpl_joints_batch[i], None, title])
        outs.append(frames)
        if outname is not None:
            imageio.mimsave(outname[i], frames, fps=fps)
    return outs
