import json
from pathlib import Path

import numpy as np
import typer
import pytetwild
import dolfinx
import ufl
import basix
from mpi4py import MPI
from dolfinx.io import XDMFFile

app = typer.Typer(help="CLI for generating realistic brain meshes (Volumetric only).")

# Global domain IDs for Biot-Stokes
POROUS_ID = 1
FLUID_ID = 2

# Original markers from fTetWild mesh
CSF_FTW = 1
PAR_FTW = 2
LV_FTW = 3
V4_FTW = 4
V3_FTW = 5


@app.command()
def mesh(
    surface_dir: Path = typer.Option(
        Path("surfaces"), help="Directory containing input PLY surfaces."
    ),
    output_dir: Path = typer.Option(
        Path("mesh_out"), help="Directory to save the base FEniCSx XDMF file."
    ),
):
    """Generates volumetric mesh using fTetWild and maps base cell domains."""
    output_dir.mkdir(exist_ok=True, parents=True)
    surface_dir = surface_dir.resolve()
    print(f"DEBUG: Looking for surfaces in exactly: {surface_dir}")

    print("Loading surfaces from PLY files...")
    required_files = [
        "skull.ply",
        "parenchyma_incl_ventr.ply",
        "LV.ply",
        "V4.ply",
        "V3_conn.ply",
    ]
    missing_files = [f for f in required_files if not (surface_dir / f).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required PLYs: {missing_files}. Run extractSurfaces.py first."
        )

    # Build CSG tree for tetrahedralization
    csg_dict = {
        "operation": "union",
        "left": str(surface_dir / "skull.ply"),
        "right": {
            "operation": "union",
            "left": str(surface_dir / "parenchyma_incl_ventr.ply"),
            "right": {
                "operation": "union",
                "left": str(surface_dir / "LV.ply"),
                "right": {
                    "operation": "union",
                    "left": str(surface_dir / "V4.ply"),
                    "right": str(surface_dir / "V3_conn.ply"),
                },
            },
        },
    }

    import tempfile

    csg_file = Path(tempfile.gettempdir()) / "brain_mesh_csg.json"
    with open(csg_file, "w") as f:
        json.dump(csg_dict, f, indent=2)
    print(f"CSG file written to: {csg_file}")

    print("Tetrahedralizing with pytetwild using CSG approach...")
    tet_mesh = pytetwild.tetrahedralize_csg(
        str(csg_file),
        epsilon=1e-3,
        edge_length_r=0.02,
        stop_energy=10.0,
        coarsen=True,
        num_threads=0,
        loglevel=4,
    )

    print(
        f"Generated mesh with {tet_mesh.n_cells} cells and {tet_mesh.n_points} points"
    )

    # Extract mesh data
    available_cells = getattr(tet_mesh, "cells_dict", {})
    cells = available_cells.get("tetra", np.array([], dtype=np.int64))
    if cells.size == 0 and tet_mesh.n_cells > 0:
        connectivity = np.asarray(tet_mesh.cells, dtype=np.int64)
        if connectivity.size > 0:
            node_count = int(connectivity[0])
            stride = node_count + 1
            if connectivity.size % stride == 0:
                reshaped = connectivity.reshape(-1, stride)
                if np.all(reshaped[:, 0] == node_count):
                    cells = reshaped[:, 1:]
    if cells.size == 0:
        cells = available_cells.get("hexahedron", np.array([], dtype=np.int64))
    if cells.size == 0:
        raise ValueError("No tetrahedral or hexahedral cells found in output mesh")

    point_array = tet_mesh.points
    cell_array = cells

    # Get markers from the mesh
    if "marker" in tet_mesh.cell_data:
        raw_markers = np.asarray(tet_mesh["marker"], dtype=np.int32)
    else:
        print("Warning: No marker field found in mesh, using default markers")
        raw_markers = np.ones(len(cell_array), dtype=np.int32)

    labels = np.copy(raw_markers)

    print(f"Marker value distribution: {np.bincount(raw_markers)}")

    print("Mapping fTetWild markers to subdomain IDs...")
    subdomains = np.copy(raw_markers)
    subdomains[np.isin(raw_markers, [PAR_FTW])] = 100
    subdomains[np.isin(raw_markers, [CSF_FTW, LV_FTW, V4_FTW, V3_FTW])] = FLUID_ID
    subdomains[np.isin(subdomains, [100])] = POROUS_ID

    print(
        f"Marked {np.sum(subdomains == FLUID_ID)} fluid cells and {np.sum(subdomains == POROUS_ID)} porous cells"
    )

    print("Constructing FEniCSx mesh...")
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD,
        cells=cell_array.astype(np.int64),
        x=point_array,
        e=ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
    )

    def create_cell_tags(mesh, values_array):
        local_entities, local_values = dolfinx.io.distribute_entity_data(
            mesh,
            mesh.topology.dim,
            cell_array.astype(np.int64),
            values_array.astype(np.int32),
        )
        adj = dolfinx.graph.adjacencylist(local_entities)
        return dolfinx.mesh.meshtags_from_entities(
            mesh, mesh.topology.dim, adj, local_values.astype(np.int32, copy=False)
        )

    ct = create_cell_tags(domain, subdomains)
    ct2 = create_cell_tags(domain, labels)

    ct.name = "subdomains"
    ct2.name = "subdomains_ftetwild"

    meshfile = (output_dir / output_dir.name).with_suffix(".xdmf")

    print(f"Exporting base mesh and cell tags to {meshfile}...")
    with XDMFFile(domain.comm, meshfile, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(ct, domain.geometry)
        xdmf.write_meshtags(ct2, domain.geometry)

    print(
        f"Number of cells: {domain.topology.index_map(domain.topology.dim).size_local}"
    )
    print("Process complete.")


if __name__ == "__main__":
    app()
