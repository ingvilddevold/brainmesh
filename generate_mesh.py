import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pyvista as pv
import typer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance

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

# Surfaces in CSG order: fTetWild marks each cell with the (1-based) index of
# the last surface containing it, which gives the *_FTW markers above
SURFACES = ["skull", "parenchyma_incl_ventr", "LV", "V4", "V3_conn"]


def _implicit(surface: Path) -> vtkImplicitPolyDataDistance:
    """Signed distance to a closed surface, negative inside it."""
    surf = pv.read(surface)
    f = vtkImplicitPolyDataDistance()
    f.SetInput(surf)
    # The sign follows the triangle orientation, so check that it points outwards
    outside = np.array(surf.center) + [2 * surf.length, 0.0, 0.0]
    if f.EvaluateFunction(outside) < 0:
        raise ValueError(f"{surface.name} has inward-facing triangles")
    return f


def _sizing_field(
    surface_dir: Path,
    h_sas: float,
    h_lv: float,
    h_v3v4: float,
    h_coarse: float,
    grading: float,
):
    """fTetWild sizing function: edge length h_sas in the SAS (inside the
    skull surface, outside the parenchyma surface), h_lv in the lateral
    ventricles and h_v3v4 in the third and fourth ventricles (including the
    aqueduct), each growing by `grading` per unit distance, up to h_coarse."""
    skull, brain, lv, v3, v4 = (
        _implicit(surface_dir / f"{name}.ply")
        for name in ["skull", "parenchyma_incl_ventr", "LV", "V3_conn", "V4"]
    )

    def size(x):
        # Signed distances to each region, negative inside it
        d_sas = max(skull.EvaluateFunction(x), -brain.EvaluateFunction(x))
        d_lv = lv.EvaluateFunction(x)
        d_v3v4 = min(v3.EvaluateFunction(x), v4.EvaluateFunction(x))
        return min(
            h_sas + grading * max(d_sas, 0.0),
            h_lv + grading * max(d_lv, 0.0),
            h_v3v4 + grading * max(d_v3v4, 0.0),
            h_coarse,
        )

    return size


def _merge_stray_cells(cells: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Make the parenchyma, LV, V4 and V3 one connected region each, by giving any
    smaller cluster of their cells the marker it shares most faces with.

    The LV and V3 surfaces run almost parallel around the foramina of Monro, and the
    V3 and V4 surfaces share a cap at the aqueduct, where fTetWild can't always separate
    them. It leaves a few cells with the wrong marker, which show up as fragments on the
    interfaces there and can partly block the aqueduct. The SAS is left out, as it has
    real pockets enclosed by the parenchyma surface.
    """
    sorted_cells = np.sort(cells, axis=1)
    faces = np.vstack([np.delete(sorted_cells, i, axis=1) for i in range(4)])
    owners = np.tile(np.arange(len(cells)), 4)
    _, face_ids = np.unique(faces, axis=0, return_inverse=True)
    order = np.argsort(face_ids.ravel())
    face_ids, owners = face_ids.ravel()[order], owners[order]
    # Interior faces: each appears twice, so sorted copies are neighbours
    shared = face_ids[1:] == face_ids[:-1]
    c0, c1 = owners[:-1][shared], owners[1:][shared]

    markers = markers.copy()
    num_changed = 0
    while True:
        same = markers[c0] == markers[c1]
        adjacency = coo_matrix(
            (np.ones(same.sum()), (c0[same], c1[same])), shape=(len(cells), len(cells))
        )
        _, clusters = connected_components(adjacency, directed=False)
        strays = []
        for region in (PAR_FTW, LV_FTW, V4_FTW, V3_FTW):
            ids, counts = np.unique(clusters[markers == region], return_counts=True)
            strays.extend(ids[counts < counts.max()] if len(ids) else [])
        if not strays:
            break
        for stray in strays:
            member = clusters == stray
            # Skip a cluster already merged into its neighbour earlier in this pass
            boundary = member[c0] != member[c1]
            neighbours = np.r_[markers[c0[boundary & ~member[c0]]], markers[c1[boundary & ~member[c1]]]]
            neighbours = neighbours[neighbours != markers[member][0]]
            if len(neighbours):
                num_changed += member.sum()
                markers[member] = np.bincount(neighbours).argmax()
    print(f"Merged {num_changed} stray cells")
    return markers


@app.command()
def mesh(
    surface_dir: Annotated[
        Path, typer.Option(help="Directory containing input PLY surfaces.")
    ] = Path("surfaces"),
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save the base FEniCSx XDMF file.")
    ] = Path("mesh_out"),
    sas_edge_mm: Annotated[
        float, typer.Option(help="Target edge length (mm) in the SAS.")
    ] = 2.0,
    lv_edge_mm: Annotated[
        float, typer.Option(help="Target edge length (mm) in the lateral ventricles.")
    ] = 2.0,
    v3v4_edge_mm: Annotated[
        float,
        typer.Option(
            help="Target edge length (mm) in the third and fourth ventricles and the aqueduct."
        ),
    ] = 1.0,
    coarse_edge_mm: Annotated[
        float, typer.Option(help="Edge length (mm) away from the SAS and ventricles.")
    ] = 10.0,
    grading: Annotated[
        float,
        typer.Option(
            help="Growth of the edge length per unit distance from the SAS and "
            "ventricles. Neighbouring cells differ in size by roughly a factor "
            "1 + grading."
        ),
    ] = 1.0,
    epsilon: Annotated[
        float,
        typer.Option(
            help="fTetWild envelope: how far the mesh surfaces may deviate from "
            "the PLYs, relative to the bounding box diagonal (~0.3 m). Larger "
            "values fragment and enlarge the contact between ventricles, and "
            "fTetWild can abort while simplifying the surfaces."
        ),
    ] = 1e-3,
    coarsen: Annotated[
        bool,
        typer.Option(
            help="Let fTetWild coarsen the mesh where the sizing field allows. "
            "Cuts the cell count in the parenchyma considerably."
        ),
    ] = True,
):
    """Generates volumetric mesh using fTetWild and maps base cell domains."""
    # Import order matters: fTetWild segfaults if dolfinx is imported before it,
    # so wildmeshing is imported here and dolfinx only after meshing below
    import wildmeshing as wm

    output_dir.mkdir(exist_ok=True, parents=True)
    surface_dir = surface_dir.resolve()

    missing_files = [
        f"{name}.ply" for name in SURFACES if not (surface_dir / f"{name}.ply").exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required PLYs: {missing_files}. Run extract_surfaces.py first."
        )

    # Build CSG tree for tetrahedralization: nested unions in SURFACES order
    csg_tree = str(surface_dir / f"{SURFACES[-1]}.ply")
    for name in reversed(SURFACES[:-1]):
        csg_tree = {
            "operation": "union",
            "left": str(surface_dir / f"{name}.ply"),
            "right": csg_tree,
        }

    print(
        f"Tetrahedralizing with fTetWild (SAS edge {sas_edge_mm} mm, LV edge "
        f"{lv_edge_mm} mm, V3/V4 edge {v3v4_edge_mm} mm, coarse edge {coarse_edge_mm} mm)..."
    )
    # edge_length_r caps the edge length, so lift it to the coarse size
    bounds = np.array(pv.read(surface_dir / "skull.ply").bounds).reshape(3, 2)
    edge_length_r = coarse_edge_mm * 1e-3 / np.linalg.norm(np.diff(bounds))
    tetra = wm.Tetrahedralizer(
        epsilon=epsilon, edge_length_r=edge_length_r, stop_quality=10.0, coarsen=coarsen
    )
    tetra.load_csg_tree(json.dumps(csg_tree))
    tetra.set_sizing_field_from_func(
        _sizing_field(
            surface_dir,
            sas_edge_mm * 1e-3,
            lv_edge_mm * 1e-3,
            v3v4_edge_mm * 1e-3,
            coarse_edge_mm * 1e-3,
            grading,
        )
    )
    tetra.tetrahedralize()
    point_array, cell_array, marker = tetra.get_tet_mesh()

    import basix
    import dolfinx
    import ufl
    from dolfinx.io import XDMFFile
    from mpi4py import MPI

    print(f"Generated mesh with {len(cell_array)} cells and {len(point_array)} points")

    raw_markers = _merge_stray_cells(
        cell_array, np.asarray(marker, dtype=np.int32).ravel()
    )
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
