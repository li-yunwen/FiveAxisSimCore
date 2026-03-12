### Dev environment
Ubuntu 24.04

### Dependencies
Run `uv sync`
#### For GPU acceleration
Cupy: use `nvidia-sim` to check the correct cupy version for your GPU, and modify `cupy-cuda12x` in `pyproject.toml`.

---

## Algorithm Summary

FiveAxisSimCore simulates 5-axis CNC cutting by tracking how a moving cutting tool removes material from a workpiece. The core representation is a **Signed Distance Field (SDF)** voxel grid, and all operations—from workpiece initialization to per-step material removal—are expressed as SDF updates.

### 1. Workpiece Representation (SDF Voxelization)

The workpiece is discretized into a uniform 3-D voxel grid. For each voxel center **p**, the signed distance to the workpiece surface is stored:

- **SDF < 0**: inside the workpiece (solid material)
- **SDF = 0**: on the surface
- **SDF > 0**: outside the workpiece (air)

The SDF is computed from an input triangle mesh using Open3D's ray-casting engine (`RaycastingScene.compute_signed_distance`). An `active_mask` boolean array tracks which voxels still contain material (SDF ≤ 0 + one-pitch margin), allowing inactive (already-cut or exterior) voxels to be skipped in subsequent steps.

### 2. Spatial Acceleration: Block Index

To avoid testing every voxel against every tool position, voxels are grouped into **cubic blocks** of configurable size (`block_size_in_pitch` × `pitch`). Each block is identified by a compact 64-bit hash key formed by bit-shifting its integer 3-D coordinates:

```
key = (ix << 40) | (iy << 20) | iz
```

The sorted key array enables **O(log N) binary search** (`searchsorted`) to locate blocks that overlap a given region, returning only candidate voxels for further evaluation.

### 3. Cutting Tool Geometry

Each tool type defines a `radius_at(h)` function that returns the tool's cross-sectional radius at axial height `h`. The tool SDF at any 3-D point is derived from two geometric quantities:

- **h** = axial projection of the point along the tool axis
- **r** = radial distance from the tool axis

The signed distance combines an *outside distance* (Euclidean to the nearest surface) and an *inside distance* (minimum clearance to the three boundaries: side wall, top face, bottom face):

```
dh      = max(max(h - L, 0), max(-h, 0))   # axial overshoot beyond [0, L]
outside = sqrt(max(r - R, 0)^2 + dh^2)
inside  = -min(R - r,  L - h,  h)
sdf     = inside  if  r ≤ R and 0 ≤ h ≤ L,  else  outside
```

Three concrete tool profiles are provided:

| Tool | `radius_at(h)` |
|------|----------------|
| **BallCutter** | `sqrt(R² - (R - h)²)` for `h < R` (hemispherical tip: circle cross-section of a sphere of radius R, measured from the ball's bottom); constant `R` for `h ≥ R` |
| **TapperCutter** | Linear interpolation: `r_tip + (r_base - r_tip) × (h / L)` |
| **CylinderCutter** | Constant `R` at all heights |

### 4. Material Removal (Per-Step Cutting)

At each trajectory point `(origin, direction)` the cutting operation (`cut_inplace`) proceeds in three stages:

1. **Block coarse filter** — Reject blocks whose AABB does not intersect the tool's bounding cylinder (extended by a configurable `margin`). Only voxels belonging to surviving blocks proceed.
2. **Geometric fine filter** — For the remaining candidate voxels, apply the same axial/radial inequality check at the voxel level.
3. **SDF update** — Compute the tool SDF at each surviving voxel and apply the Boolean subtraction rule:

```
new_sdf = max(old_sdf, -tool_sdf)
```

A voxel transitions from solid to air (is *removed*) when `old_sdf ≤ 0` and `new_sdf > 0`. The count of removed voxels is returned as the **chip volume** for that step.

After each update, `deactivate_outer_volume()` marks newly air voxels as inactive so they are excluded from future steps.

### 5. Simulation Loop

```
Simulator.run(trajectory):
    1. _prefilter_workpiece(trajectory)        # AABB over full trajectory + deactivate exterior
    2. for each (origin, direction) in trajectory:
           cutter.cut_inplace(workpiece, origin, direction)   # Stages 1–3 above
           record chip_voxel_count
    3. return updated workpiece SDF + chip volume list
```

**Trajectory-aware global prefilter** (step 1): A single bounding box that encloses every tool position in the entire trajectory (augmented by the tool radius and length) is computed. Voxels outside this box are deactivated before the loop begins, dramatically reducing the active set.

### 6. G-code Input

Tool paths are read from standard G-code files. A regex-based stateful parser extracts `X Y Z I J K` fields (with scientific-notation support), producing an `N × 6` trajectory matrix where columns 0–2 are the tool-tip position and columns 3–5 are the (normalized) tool-axis direction vector.

### 7. GPU / CPU Backend

All numerical arrays use a unified backend module that transparently switches between **CuPy** (GPU) and **NumPy** (CPU). The `xp` alias resolves to whichever library is available, so every algorithm runs unchanged on both backends.