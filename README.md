### Dev environment
Ubuntu 24.04

### Dependencies
Run `uv sync`
#### For GPU acceleration
Cupy: use `nvidia-sim` to check the correct cupy version for your GPU, and modify `cupy-cuda12x` in `pyproject.toml`.

---

## Usage (written by Copilot)

### Quick start

Run the built-in end-to-end demo (loads the sample G-code, builds a box workpiece, runs a ball-end cutting simulation and visualises the result):

```bash
python simulate.py
```

### Step-by-step API

#### 1. Parse a G-code file into a trajectory matrix

```python
from utils.gcode_reader import GCodeReader

with open("assets/gcode/test.ncc", "r", encoding="utf-8") as fp:
    trajectory = GCodeReader.load(fp)
# trajectory: float64 ndarray of shape (N, 6)
# columns: [X, Y, Z, I, J, K]  (position + tool-axis direction)
```

#### 2. Create a workpiece mesh

Any `open3d.geometry.TriangleMesh` is accepted. You can load a file or build a primitive:

```python
import open3d as o3d

# Primitive box (width × height × depth in mm, centred at the origin)
mesh = o3d.geometry.TriangleMesh.create_box(width=10, height=10, depth=5)
mesh.translate((-5, -5, -2.5))

# Or load from a file (STL, OBJ, PLY, …)
# mesh = o3d.io.read_triangle_mesh("workpiece.stl")
```

#### 3. Choose a cutter

| Class | Parameters | Description |
|-------|-----------|-------------|
| `BallCutter` | `radius`, `length` | Ball-end (hemispherical tip + cylindrical shank) |
| `TapperCutter` | `radius_base`, `radius_tip`, `length` | Linearly tapered flat-bottom cutter (`radius_base ≥ radius_tip`) |
| `CylinderCutter` | `radius`, `length` | Flat-bottom cylindrical cutter |

```python
from cutter import BallCutter, TapperCutter, CylinderCutter

cutter = BallCutter(radius=0.5, length=8.0)
# cutter = TapperCutter(radius_base=0.5, radius_tip=0.2, length=8.0)
# cutter = CylinderCutter(radius=0.5, length=8.0)
```

#### 4. Create a `Simulator` and run the simulation

```python
from simulate import Simulator
from utils.backend import as_xp_array

# pitch         – voxel edge length (mm). Smaller = finer detail, more memory/time.
# block_size_in_pitch – voxels per block side for spatial acceleration (≥ 1).
sim = Simulator(cutter, mesh, pitch=0.1, block_size_in_pitch=10)

traj = as_xp_array(trajectory)  # move to GPU if CuPy is active

# vis_interval – render an intermediate view every N steps (set to 0 to disable).
workpiece_sdf, chip_voxel_list = sim.run(traj, vis_interval=500)
```

`run()` returns:

| Return value | Type | Description |
|---|---|---|
| `workpiece_sdf` | `Sdfer` | Updated workpiece signed-distance field after all cuts |
| `chip_voxel_list` | `list[int]` | Number of voxels removed at each trajectory step |

#### 5. Convert chip counts to chip volume

```python
pitch = 0.1  # mm (must match the value passed to Simulator)
chip_volume_mm3 = [n * pitch**3 for n in chip_voxel_list]
```

#### 6. Visualise results

```python
from sdfer import plot_multiple_sdfer

# Compare original and cut workpiece side-by-side
from sdfer import Sdfer
sdf_original = Sdfer.from_mesh(mesh, pitch=0.1, block_size_in_pitch=10)
plot_multiple_sdfer(
    [sdf_original, workpiece_sdf],
    opacities=[0.1, 1.0],
    colors=["lightgrey", "lightgrey"],
    legends=["Original", "Cut"],
)

# 3-D chip-volume heat map along the trajectory
sim.visualize_chip_volume_3d(traj[:, :3], chip_voxel_list, pitch=0.1)
```

### Configuration

Edit `config.py` to tune global settings:

| Variable | Default | Description |
|---|---|---|
| `FLOAT_TYPE_NAME` | `"float32"` | Floating-point precision. `"float32"` is recommended. |
| `PITCH_DEFAULT` | `0.5` | Fallback voxel edge length (mm) used when `pitch` is not passed explicitly. |
| `USE_GPU` | `True` | Set to `False` to force NumPy/CPU mode even when a CUDA GPU is available. |

---

## Algorithm Summary (summarized by Copilot)

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
