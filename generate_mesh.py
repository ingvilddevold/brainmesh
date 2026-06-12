import json
from pathlib import Path

import numpy as np
import typer
import pytetwild
import dolfinx
import ufl
import basix
import yaml
from mpi4py import MPI
from dolfinx.io import XDMFFile

app = typer.Typer(help="CLI for generating realistic brain meshes and FEniCSx tags.")

# Global domain IDs for Biot-Stokes
POROUS_ID = 1
FLUID_ID = 2

# Boundary IDs
INTERFACE_ID = 1
PIA_ID = 11
EPENDYMA_ID = 12
SKULL_ID = 2
SPINAL_CANAL_ID = 3
FIXED_STEM_ID = 4
AQUEDUCT_MID_ID = 5

# Original markers from fTetWild mesh
CSF_FTW = 1
PAR_FTW = 2
LV_FTW = 3
V4_FTW = 4
V3_FTW = 5

mm2m = 1e-3


@app.command()
def mesh(
    surface_dir: Path = typer.Option(
        Path("surfaces"), help="Directory containing input PLY surfaces."
    ),
    output_dir: Path = typer.Option(
        Path("mesh_out"), help="Directory to save FEniCSx XDMF files."
    ),
    configfile: Path = typer.Option(..., help="Path to the mesh config file (yaml)."),
    separate_interfaces: bool = typer.Option(
        False,
        "--separate-interfaces",
        help="Tag Pia (11) and Ependyma (12) separately instead of a unified interface tag (1).",
    ),
):
    """Generates volumetric mesh mapping using original fTetWild np.isin approach for real brain."""
    output_dir.mkdir(exist_ok=True, parents=True)
    surface_dir = surface_dir.resolve()
    print(f"DEBUG: Looking for surfaces in exactly: {surface_dir}")

    # Read the config file for the clip planes
    with open(configfile) as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.UnsafeLoader)
    config = mesh_config["surface"]

    clip_origin_z = config["clip_origin"]["z"] * mm2m

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
    # Union of skull with the inner structures (parenchyma + ventricles)
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

    # Write CSG to temporary file
    import tempfile

    csg_file = Path(tempfile.gettempdir()) / "brain_mesh_csg.json"
    with open(csg_file, "w") as f:
        json.dump(csg_dict, f, indent=2)
    print(f"CSG file written to: {csg_file}")

    print("Tetrahedralizing with pytetwild using CSG approach...")

    # Use tetrahedralize_csg which automatically assigns markers based on containment
    tet_mesh = pytetwild.tetrahedralize_csg(
        str(csg_file),
        epsilon=1e-3,
        edge_length_r=0.05,
        stop_energy=10.0,
        coarsen=False,
        num_threads=0,
        loglevel=3,
    )

    print(
        f"Generated mesh with {tet_mesh.n_cells} cells and {tet_mesh.n_points} points"
    )

    # Extract the mesh data and markers from the returned PyVista object.
    # pytetwild returns a PyVista mesh with a cell array and a marker cell array.
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

    # Get markers from the mesh, using the PyVista cell-data view if available.
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

    print("Marking boundaries and interfaces...")
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_entities(fdim)

    num_facets = domain.topology.index_map(fdim).size_local
    marker_values = np.zeros(num_facets, dtype=np.int32)

    def get_internal_interface_facets(cell_tags, doms=None):
        domain.topology.create_connectivity(fdim, tdim)
        f_to_c = domain.topology.connectivity(fdim, tdim)
        internal_facets = []
        for f in range(num_facets):
            cells = f_to_c.links(f)
            if len(cells) == 2:
                domains = {cell_tags.values[cells[0]], cell_tags.values[cells[1]]}
                if doms is None or set(doms) == domains:
                    internal_facets.append(f)
        return np.array(internal_facets, dtype=np.int32)

    def get_external_boundary_facets(cell_tags, subdomain_ids):
        domain.topology.create_connectivity(fdim, tdim)
        f_to_c = domain.topology.connectivity(fdim, tdim)
        boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
        target_facets = []
        for f in boundary_facets:
            c = f_to_c.links(f)[0]
            if cell_tags.values[c] in subdomain_ids:
                target_facets.append(f)
        return np.array(target_facets, dtype=np.int32)

    # Aqueduct plane = interface interface between V3 (5) and V4 (4)
    aqueduct_facets = get_internal_interface_facets(ct2, doms=[V4_FTW, V3_FTW])

    # Extract exterior boundary facets for specific subdomains
    csf_outer_facets = get_external_boundary_facets(ct2, [CSF_FTW])
    par_outer_facets = get_external_boundary_facets(ct2, [PAR_FTW])

    # Locate the cut plane on the CSF boundary (Spinal Outlet)
    def is_cut_plane(x):
        return np.isclose(x[2], clip_origin_z, atol=5e-4)

    bottom_facets_geom = dolfinx.mesh.locate_entities_boundary(
        domain, fdim, is_cut_plane
    )

    # Isolate the CSF outlet and remove it from the general skull tag
    spinal_outlet_facets = np.intersect1d(csf_outer_facets, bottom_facets_geom)
    skull_facets = np.setdiff1d(csf_outer_facets, spinal_outlet_facets)

    if separate_interfaces:
        print("Using separate tags for Pia (11) and Ependyma (12)...")
        pia_facets = get_internal_interface_facets(ct2, doms=[CSF_FTW, PAR_FTW])

        ependyma_facets_1 = get_internal_interface_facets(ct2, doms=[PAR_FTW, LV_FTW])
        ependyma_facets_2 = get_internal_interface_facets(ct2, doms=[PAR_FTW, V4_FTW])
        ependyma_facets_3 = get_internal_interface_facets(ct2, doms=[PAR_FTW, V3_FTW])
        ependyma_facets = np.concatenate(
            [ependyma_facets_1, ependyma_facets_2, ependyma_facets_3]
        )

        marker_values[pia_facets] = PIA_ID
        marker_values[ependyma_facets] = EPENDYMA_ID
    else:
        print("Using unified interface tag (1)...")
        tissue_csf_facets = get_internal_interface_facets(
            ct, doms=[FLUID_ID, POROUS_ID]
        )
        marker_values[tissue_csf_facets] = INTERFACE_ID

    # Apply common markers
    marker_values[aqueduct_facets] = AQUEDUCT_MID_ID
    marker_values[skull_facets] = SKULL_ID
    marker_values[spinal_outlet_facets] = SPINAL_CANAL_ID
    marker_values[par_outer_facets] = FIXED_STEM_ID

    # Filter unmarked facets and create MeshTags
    tagged_indices = np.where(marker_values != 0)[0].astype(np.int32)
    tagged_values = marker_values[tagged_indices]

    bm = dolfinx.mesh.meshtags(domain, fdim, tagged_indices, tagged_values)

    print("Exporting mesh and mesh tags to XDMF...")

    bm.name = "boundaries"
    ct.name = "subdomains"
    ct2.name = "subdomains_ftetwild"

    meshfile = (output_dir / output_dir.name).with_suffix(".xdmf")

    with XDMFFile(domain.comm, meshfile, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(ct, domain.geometry)
        xdmf.write_meshtags(ct2, domain.geometry)
        xdmf.write_meshtags(bm, domain.geometry)

    print(f"Number of cells: {domain.topology.index_map(tdim).size_local}")
    print(f"Process complete. Mesh written to {output_dir.absolute()}.")


if __name__ == "__main__":
    app()
