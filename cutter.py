import open3d as o3d
from sdfer import Sdfer, plot_multiple_sdfer
from utils.logger import Logger
from utils.backend import xp, GPU_ENABLED, DTYPE_FLOAT

# ============================================================
# Base Cutter Class
# ============================================================
class BaseCutter:
    """Base cutter class defining the common interface. All cutters have the bottom center as origin and the Z-axis as the axis."""

    def __init__(self, length: DTYPE_FLOAT):
        self.length = DTYPE_FLOAT(length)

    def radius_at(self, h: DTYPE_FLOAT) -> DTYPE_FLOAT:
        """返回高度 h 处刀具的半径"""
        raise NotImplementedError

    # ======== 生成刀具 SDF ========
    def get_sdfer(self, points, origin, direction, vis=False, shape=None):
        """生成刀具的符号距离场 (SDF)"""
        sdf_flat = self._sdf_points(points, origin, direction)
        sdfer = Sdfer(points, sdf_flat, shape)
        if vis:
            sdfer.plot_level(0.0)
        return sdfer

    # ======== 刀具几何核心计算 ========
    def _sdf_points(
        self, pts_flat: xp.ndarray, origin: xp.ndarray, direction: xp.ndarray
    ) -> xp.ndarray:
        """Compute the SDF values of the cutter at the specified points."""

        d = direction / xp.linalg.norm(direction)

        # Basic geometric relations
        rel = pts_flat - origin
        h = rel @ d
        r2 = (
            xp.sum(rel**2, axis=1) - h**2
        )  # Equivalent to xp.linalg.norm(rel - xp.outer(h, d), axis=1)**2, but faster
        r = xp.sqrt(xp.maximum(r2, 0))

        # Radius distribution
        R = xp.asarray(self.radius_at(xp.clip(h, 0, self.length)))

        # Outside region distance
        dh_top = xp.maximum(h - self.length, 0)
        dh_bottom = xp.maximum(-h, 0)
        dr = xp.maximum(r - R, 0)
        outside = xp.sqrt(dr**2 + xp.maximum(dh_top, dh_bottom) ** 2)

        # Inside region distance
        dist_side = R - r
        dist_top = self.length - h
        dist_bottom = h
        inside = -xp.minimum(xp.minimum(dist_side, dist_top), dist_bottom)

        # Combine overall SDF
        sdf = xp.where((r <= R) & (h >= 0) & (h <= self.length), inside, outside)

        return sdf
    
    # ======== Cutting operation ========
    def cut_inplace(
        self,
        sdfer_workpiece: Sdfer,
        origin: xp.ndarray,
        direction: xp.ndarray,
        margin: DTYPE_FLOAT = 1.0,
    ) -> int:
        """Perform cutting operation in the local region of the cutter (in-place modification of sdfer_workpiece)"""
        d = direction / xp.linalg.norm(direction)
        L = self.length
        Rmax = max(
            getattr(self, "radius_base", 0.0),
            getattr(self, "radius", 0.0),
            getattr(self, "radius_tip", 0.0),
        )
        margin = DTYPE_FLOAT(margin)

        # --- Block coarse filtering ---
        rel = sdfer_workpiece.block_centers - origin
        h = rel @ d
        r2 = xp.sum(rel**2, axis=1) - h**2
        r = xp.sqrt(xp.maximum(r2, 0))

        # Effective expanded radius = block half diagonal
        block_r = sdfer_workpiece.block_size_in_pitch * sdfer_workpiece.pitch * xp.sqrt(3) / 2
        mask_block = (
            (h >= -margin - block_r) &
            (h <= L + margin + block_r) &
            (r <= Rmax + margin + block_r)
        )
        hit_blocks = xp.nonzero(mask_block)[0]
        if hit_blocks.size == 0:
            return 0

        starts = sdfer_workpiece.block_starts[hit_blocks]
        counts = sdfer_workpiece.block_counts[hit_blocks]

        cand_idx = xp.concatenate(
            [sdfer_workpiece.block_order[s:s + c] for s, c in zip(starts, counts)]
        )
        if cand_idx.size == 0:
            return 0

        # --- Filter inactive voxels ---
        active_mask = sdfer_workpiece.active_mask[cand_idx]
        if not xp.any(active_mask):
            return 0
        cand_idx = cand_idx[active_mask]

        # --- Geometric local filtering ---
        pts = sdfer_workpiece.pts[cand_idx]
        rel = pts - origin
        h = rel @ d
        r2 = xp.sum(rel**2, axis=1) - h**2
        r = xp.sqrt(xp.maximum(r2, 0))

        local_mask = (h >= -margin) & (h <= L + margin) & (r <= Rmax + margin)
        if not xp.any(local_mask):
            return 0

        idx_final = cand_idx[local_mask]
        pts_sub = sdfer_workpiece.pts[idx_final]
        sdf_tool = self._sdf_points(pts_sub, origin, direction)

        # --- Update in-place ---
        old = sdfer_workpiece.sdf[idx_final]
        new = xp.maximum(old, -sdf_tool)
        removed = (old <= 0) & (new > 0)
        sdfer_workpiece.sdf[idx_final] = new
        sdfer_workpiece.deactivate_outer_volume()

        return int(xp.count_nonzero(removed))

# ============================================================
# Ball cutter
# ============================================================
class BallCutter(BaseCutter):
    """Ball-end cylindrical cutter with fixed radius."""

    def __init__(self, radius: DTYPE_FLOAT, length: DTYPE_FLOAT):
        super().__init__(length)
        self.radius = radius

    def radius_at(self, h):
        """
        Return the cross-sectional radius at height h.
        Conventions:
            - origin is the bottom of the ball (h = 0)
            - 0 ≤ h ≤ radius     → hemisphere
            - radius ≤ h ≤ length → cylindrical section
        Hemisphere formula:
            r(h) = sqrt(R^2 - (R - h)^2)
        """
        R = self.radius
        h = xp.asarray(h, dtype=DTYPE_FLOAT)

        # 半球段
        mask_ball = h < R
        r = xp.empty_like(h, dtype=DTYPE_FLOAT)

        # 半球公式
        r_ball = xp.sqrt(xp.maximum(R * R - (R - h) ** 2, 0))

        # 柱段：半径恒定 R
        r_cyl = xp.full_like(h, R, dtype=DTYPE_FLOAT)

        r[mask_ball] = r_ball[mask_ball]
        r[~mask_ball] = r_cyl[~mask_ball]

        return r


# ============================================================
# Tapered and flat-bottom cutters
# ============================================================
class TapperCutter(BaseCutter):
    """Tapered flat-bottom cutter with different base and tip radii."""

    def __init__(
        self, radius_base: DTYPE_FLOAT, radius_tip: DTYPE_FLOAT, length: DTYPE_FLOAT
    ):
        super().__init__(length)
        assert radius_base >= radius_tip
        self.radius_base = radius_base
        self.radius_tip = radius_tip

    def radius_at(self, h):
        """Linearly interpolate the radius at height h"""
        return self.radius_tip + (self.radius_base - self.radius_tip) * (
            h / self.length
        )


class CylinderCutter(BaseCutter):
    """Cylindrical flat-bottom cutter with fixed radius."""

    def __init__(self, radius: DTYPE_FLOAT, length: DTYPE_FLOAT):
        super().__init__(length)
        self.radius = radius

    def radius_at(self, h):
        """Fixed radius"""
        return xp.full_like(h, self.radius, dtype=DTYPE_FLOAT)


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    cutter = TapperCutter(radius_base=0.5, radius_tip=0.2, length=8.0)
    ball_cutter = BallCutter(radius=0.5, length=8.0)
    pitch = 0.1
    mesh = o3d.geometry.TriangleMesh.create_box(width=8, height=8, depth=8)
    mesh.translate([-4, -4, 0])
    sdfer_workpiece = Sdfer.from_mesh(mesh, pitch, block_size_in_pitch=4)

    sdfer_cutter = cutter.get_sdfer(
        sdfer_workpiece.pts,
        xp.array([0, 0, 0]),
        xp.array([0, 1, 1]),
        vis=False,
        shape=sdfer_workpiece.shape,
    )
    sdfer_ball_cutter = ball_cutter.get_sdfer(
        sdfer_workpiece.pts,
        xp.array([0, 0, 1]),
        xp.array([0, -1, 1]),
        vis=False,
        shape=sdfer_workpiece.shape,
    )

    plot_multiple_sdfer([sdfer_workpiece, sdfer_cutter, sdfer_ball_cutter], opacities=[0.3, 0.7, 0.7])

    chip = ball_cutter.cut_inplace(
        sdfer_workpiece, xp.array([0, 0, 0]), xp.array([0, 0, 1]), margin=2 * pitch
    )

    plot_multiple_sdfer([sdfer_workpiece], opacities=[0.9])
