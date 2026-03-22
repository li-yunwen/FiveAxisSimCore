import numpy as np
import open3d as o3d
from cutter import BaseCutter, TapperCutter, CylinderCutter, BallCutter
from utils.logger import Logger
from utils.gcode_reader import GCodeReader
from sdfer import Sdfer, plot_multiple_sdfer
from tqdm import tqdm
from utils.backend import xp, GPU_ENABLED, as_xp_array, DTYPE_FLOAT


PITCH = 0.1


class Simulator:
    """Five-axis cutting simulator."""

    def __init__(
        self,
        cutter: BaseCutter,
        workpiece_mesh: o3d.geometry.TriangleMesh,
        pitch: DTYPE_FLOAT = 0.2,
        block_size_in_pitch: int = 5,
    ):
        self.cutter = cutter
        self.sdfer_workpiece = Sdfer.from_mesh(
            workpiece_mesh, pitch=pitch, block_size_in_pitch=block_size_in_pitch
        )
        self.pitch = pitch
        self.logger = Logger("Simulator")

    # ------------------------------------------------------------
    # Trajectory-aware AABB global clipping, only keep voxels that may be cut
    # ------------------------------------------------------------
    def _prefilter_workpiece(self, trajectory: xp.ndarray):
        """Pre-filter workpiece voxels based on trajectory direction-aware bounding box (GPU compatible)"""
        cutter = self.cutter
        pts = as_xp_array(self.sdfer_workpiece.pts)

        # 1. Trajectory positions and directions (keep on CPU)
        pos = as_xp_array(trajectory[:, :3])
        dirs = as_xp_array(trajectory[:, 3:6])
        dirs /= xp.linalg.norm(dirs, axis=1, keepdims=True)

        # 2. Direction-aware bounding box
        Rmax = max(
            getattr(cutter, "radius", 0.0),
            getattr(cutter, "radius_base", 0.0),
            getattr(cutter, "radius_tip", 0.0),
        )
        p_min = pos - dirs * cutter.length - Rmax
        p_max = pos + dirs * cutter.length + Rmax
        traj_min = xp.min(p_min, axis=0)
        traj_max = xp.max(p_max, axis=0)
        margin = 2 * self.pitch

        # 3. Global mask filtering (entirely on GPU)
        mask_global = (
            (pts[:, 0] >= traj_min[0] - margin)
            & (pts[:, 0] <= traj_max[0] + margin)
            & (pts[:, 1] >= traj_min[1] - margin)
            & (pts[:, 1] <= traj_max[1] + margin)
            & (pts[:, 2] >= traj_min[2] - margin)
            & (pts[:, 2] <= traj_max[2] + margin)
        )

        # Directly set active mask inside Sdfer
        self.sdfer_workpiece.active_mask = mask_global # AABB Prefiltering
        self.sdfer_workpiece.deactivate_outer_volume() # Deactivate outer voxels

    # ------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------
    def run(self, trajectory: xp.ndarray, vis_interval: int = 500):
        """
        Execute cutting step-by-step according to the trajectory.
        Parameters
        ----
        trajectory : ndarray [N,6]
            [X,Y,Z,I,J,K]
        vis_interval : int
            How many steps between each visualization update
        """
        # ---- Step 0: Global prefilter ----
        self._prefilter_workpiece(trajectory)
        wp_cut = self.sdfer_workpiece # alias, changes will be in-place to self.sdfer_workpiece
        cutter = self.cutter

        # ---- Step 1: Main cutting loop ----
        chip_voxel_list = []
        with tqdm(total=len(trajectory), ncols=100, desc="Simulating") as pbar:
            for i, row in enumerate(trajectory):
                x, y, z, i_, j_, k_ = row
                origin = xp.array([x, y, z])
                direction = xp.array([i_, j_, k_]) / xp.linalg.norm(xp.array([i_, j_, k_]))

                num_chip_voxels = cutter.cut_inplace(
                    wp_cut, origin, direction, margin=self.pitch * 4
                )
                chip_voxel_list.append(num_chip_voxels)

                if (i + 1) % vis_interval == 0:
                    try:
                        cutter_sdf = cutter.get_sdfer(
                            as_xp_array(wp_cut.pts), origin, direction, shape=wp_cut.shape
                        )
                        plot_multiple_sdfer([wp_cut, cutter_sdf], opacities=[0.9], colors=['lightgrey', 'red'], legends=['Workpiece', 'Cutter SDF'])
                    except Exception as e:
                        print(f"[warn] Visualization skipped: {e}")

                pbar.update(1)

        return wp_cut, chip_voxel_list
    
    def visualize_chip_volume_3d(self, trajectory: xp.ndarray, chip_voxel_list: list[int], pitch: float = 0.1):
        """
        Visualize the chip volume at each trajectory point using PyVista.
        trajectory: (N, 3)
        chip_voxel_list: (N,) int
        """
        import pyvista as pv
        trajectory = xp.asnumpy(trajectory) if GPU_ENABLED else trajectory
        chip_voxel_list = np.asarray(chip_voxel_list)
        chip_volume_list = chip_voxel_list * (pitch ** 3)

        # Construct point cloud
        cloud = pv.PolyData(trajectory)
        cloud["chip_volume_list [mm^3]"] = chip_volume_list
        pl = pv.Plotter()
        pl.add_mesh(
            cloud,
            scalars="chip_volume_list [mm^3]",
            point_size=8,
            render_points_as_spheres=True,
            cmap="jet",
            opacity=0.9,
        )
        pl.show()


# ------------------------------------------------------------
# Self-test + line-by-line performance analysis
# ------------------------------------------------------------
if __name__ == "__main__":
    cutter = BallCutter(radius=0.5, length=8.0)
    mesh = o3d.geometry.TriangleMesh.create_box(width=10, height=10, depth=5)
    mesh.translate((-5, -5, -2.5))
    sdf_original = Sdfer.from_mesh(mesh, pitch=PITCH, block_size_in_pitch=10)

    traj_file = f"assets/gcode/test.ncc"
    with open(traj_file, "r", encoding="utf-8") as fp:
        traj = GCodeReader.load(fp)
        traj = as_xp_array(traj)

    sim = Simulator(cutter, mesh, pitch=PITCH, block_size_in_pitch=10)
    sdfer_cut, chip_voxel_list = sim.run(traj, vis_interval=1000)

    # Results visualization
    plot_multiple_sdfer([sdf_original, sdfer_cut], opacities=[0.1, 1], colors = ['lightgrey', 'lightgrey'], legends=['Original', 'Cut'])
    sim.visualize_chip_volume_3d(traj[:, :3], chip_voxel_list, pitch=PITCH)