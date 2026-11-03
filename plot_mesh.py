from pathlib import Path
from typing import Annotated

import numpy as np
import pyvista as pv
import typer
from dolfinx import io, plot
from mpi4py import MPI

app = typer.Typer(add_completion=False)

CSF_ID, PAR_ID, LV_ID, V4_ID, V3_ID = 1, 2, 3, 4, 5
ID_TO_COLOR = {
    CSF_ID: "#A0C8F0",
    PAR_ID: "#FFC3A0",
    LV_ID: "#57ABFF",
    V4_ID: "#0080FF",
    V3_ID: "#004183",
}

CLIP_NORMAL = "x"
CLIP_ORIGIN = (-0.002, 0.0, 0.0)
CLIP_INVERT = True


def _vtk_grid(msh, dim):
    topology, cell_types, geometry = plot.vtk_mesh(msh, dim)
    cell_types = np.where(cell_types == 71, 10, cell_types)
    return pv.UnstructuredGrid(topology, cell_types, geometry)


def _get_camera(grid_for_bounds):
    pl_temp = pv.Plotter(off_screen=True)
    pl_temp.add_mesh(grid_for_bounds)
    pl_temp.camera_position = "yz"
    pl_temp.camera.azimuth += 30
    pl_temp.camera.zoom(1.1)
    camera = pl_temp.camera.copy()
    pl_temp.close()
    return camera


HIDDEN_OPACITY = 0.4


@app.command()
def plot_mesh(
    mesh_file: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, resolve_path=True, help="XDMF mesh file"
        ),
    ],
    output: Annotated[Path, typer.Argument(help="Output PNG (an .svg is also saved)")],
    region_tag: Annotated[
        str, typer.Option(help="Meshtag with region IDs 1-5")
    ] = "subdomains_ftetwild",
    include_parenchyma: Annotated[bool, typer.Option()] = True,
    include_csf: Annotated[bool, typer.Option()] = True,
    include_ventricles: Annotated[bool, typer.Option()] = True,
    include_outline: Annotated[
        bool, typer.Option(help="Translucent whole-brain outline")
    ] = True,
    clip: Annotated[
        bool, typer.Option(help="Clip parenchyma/CSF along the midline")
    ] = True,
    ventricles_within_outline: Annotated[
        bool, typer.Option(help="Clip the ventricles too")
    ] = False,
    parenchyma_opacity: Annotated[float, typer.Option()] = 1.0,
    csf_opacity: Annotated[float, typer.Option()] = 1.0,
    ventricle_opacity: Annotated[float, typer.Option()] = 1.0,
    outline_opacity: Annotated[float, typer.Option()] = 0.15,
    ambient: Annotated[float, typer.Option(help="Ambient lighting (0-1)")] = 0.3,
    window_size: Annotated[int, typer.Option(help="Render size in pixels")] = 1400,
):
    print(f"Loading mesh '{mesh_file}'...")
    comm = MPI.COMM_WORLD
    with io.XDMFFile(comm, str(mesh_file), "r") as xdmf:
        msh = xdmf.read_mesh()
        region_marker = xdmf.read_meshtags(msh, name=region_tag)

    grid = _vtk_grid(msh, msh.topology.dim)
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    region = np.zeros(num_cells, dtype=np.int32)
    region[region_marker.indices] = region_marker.values
    grid.cell_data["region"] = region

    camera = _get_camera(grid)
    clip_kwargs = {"normal": CLIP_NORMAL, "origin": CLIP_ORIGIN, "invert": CLIP_INVERT}

    plotter = pv.Plotter(off_screen=True, window_size=(window_size, window_size))
    plotter.set_background(None)
    plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)

    if include_outline:
        outline = grid.extract_surface().connectivity("largest")
        plotter.add_mesh(
            outline, color=ID_TO_COLOR[CSF_ID], opacity=outline_opacity, ambient=ambient
        )

    if include_parenchyma:
        grid_parenchyma = grid.extract_cells(region == PAR_ID)
        mesh_p = grid_parenchyma.clip(**clip_kwargs) if clip else grid_parenchyma
        plotter.add_mesh(
            mesh_p,
            color=ID_TO_COLOR[PAR_ID],
            opacity=parenchyma_opacity,
            smooth_shading=True,
            ambient=ambient,
        )

    if include_csf:
        grid_csf = grid.extract_cells(region == CSF_ID)
        mesh_c = grid_csf.clip(**clip_kwargs) if clip else grid_csf
        plotter.add_mesh(
            mesh_c,
            color=ID_TO_COLOR[CSF_ID],
            opacity=csf_opacity,
            smooth_shading=True,
            ambient=ambient,
        )

    if include_ventricles:
        clip_ventricles = clip and ventricles_within_outline
        for region_id in (LV_ID, V4_ID, V3_ID):
            vol = grid.extract_cells(region == region_id)
            if clip_ventricles:
                vol = vol.clip(**clip_kwargs)
            plotter.add_mesh(
                vol,
                color=ID_TO_COLOR[region_id],
                opacity=ventricle_opacity,
                smooth_shading=True,
                ambient=ambient,
            )

    plotter.camera = camera
    plotter.remove_bounding_box()
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output), transparent_background=True)
    plotter.save_graphic(str(output.with_suffix(".svg")))
    plotter.close()
    print(f"Saved to: {output} (+ .svg)")


if __name__ == "__main__":
    app()
