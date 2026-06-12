from pathlib import Path

import numpy as np
import typer
from mpi4py import MPI
import dolfinx
from dolfinx.io import XDMFFile
import scifem

app = typer.Typer(
    help="Refine an input XDMF mesh locally around SSAS/Aqueduct and update FEniCSx tags."
)

# Target Mapping Values
V4_FTW = 4  # 4th Ventricle (from subdomains_ftetwild)
V3_FTW = 5  # 3rd Ventricle (from subdomains_ftetwild)
SSAS_FACET_TAG = 3  # Spinal Subarachnoid Space outlet (from boundaries)


def read_all_tags(
    mesh_path: Path, cell_tag_names: list[str], facet_tag_names: list[str]
) -> tuple[
    dolfinx.mesh.Mesh,
    dict[str, dolfinx.mesh.MeshTags],
    dict[str, dolfinx.mesh.MeshTags],
]:
    """Reads the mesh and extracts dictionaries of existing cell and facet tags."""
    cell_tags = {}
    facet_tags = {}

    with XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as xdmf:
        mesh = xdmf.read_mesh(dolfinx.cpp.mesh.GhostMode.none)

        # Read cell tags
        for name in cell_tag_names:
            try:
                cell_tags[name] = xdmf.read_meshtags(mesh, name=name)
            except RuntimeError:
                if MPI.COMM_WORLD.rank == 0:
                    print(
                        f"Warning: Cell tag '{name}' not found in the input mesh. Skipping."
                    )

        # Read facet tags
        for name in facet_tag_names:
            try:
                facet_tags[name] = xdmf.read_meshtags(mesh, name=name)
            except RuntimeError:
                if MPI.COMM_WORLD.rank == 0:
                    print(
                        f"Warning: Facet tag '{name}' not found in the input mesh. Skipping."
                    )

    return mesh, cell_tags, facet_tags


def refine_mesh(
    mesh: dolfinx.mesh.Mesh,
    cell_tags: dict[str, dolfinx.mesh.MeshTags],
    facet_tags: dict[str, dolfinx.mesh.MeshTags],
    primary_region_name: str,
    num_refinements: int,
) -> tuple[
    dolfinx.mesh.Mesh,
    dict[str, dolfinx.mesh.MeshTags],
    dict[str, dolfinx.mesh.MeshTags],
]:

    tdim = mesh.topology.dim
    fdim = tdim - 1

    # Print the initial cell count
    initial_cells = mesh.topology.index_map(tdim).size_global
    if mesh.comm.rank == 0:
        print(f"Initial mesh cell count: {initial_cells}")

    if num_refinements <= 0:
        return mesh, cell_tags, facet_tags

    for i in range(num_refinements):
        # 1. Target the Aqueduct by refining the entire V3 and V4 volumes
        ftetwild_markers = cell_tags["subdomains_ftetwild"]
        v3_v4_cells = ftetwild_markers.indices[
            np.isin(ftetwild_markers.values, [V4_FTW, V3_FTW])
        ]

        # 2. Target the SSAS by finding cells attached to the SSAS boundary facets
        mesh.topology.create_connectivity(fdim, tdim)
        ssas_cells = np.array([], dtype=np.int32)

        if "boundaries" in facet_tags:
            boundaries = facet_tags["boundaries"]
            ssas_facets = boundaries.indices[boundaries.values == SSAS_FACET_TAG]
            ssas_cells = dolfinx.mesh.compute_incident_entities(
                mesh.topology, ssas_facets, fdim, tdim
            )
        else:
            if mesh.comm.rank == 0:
                print(
                    "Warning: 'boundaries' tag not found. Cannot explicitly target SSAS facets."
                )

        # Combine localized targets
        cells_to_refine = np.unique(np.hstack([v3_v4_cells, ssas_cells])).astype(
            np.int32
        )

        # Create cell-to-edge connectivity (tdim, 1) for refinement marking
        mesh.topology.create_connectivity(tdim, 1)
        edges_to_refine = dolfinx.mesh.compute_incident_entities(
            mesh.topology, cells_to_refine, tdim, 1
        )
        edge_map = mesh.topology.index_map(1)
        edges_to_refine = scifem.reverse_mark_entities(edge_map, edges_to_refine)

        # Store names prior to transfer so they don't default back to Dolfinx internal IDs
        c_names = {k: v.name for k, v in cell_tags.items()}
        f_names = {k: v.name for k, v in facet_tags.items()}

        # Perform the mesh refinement
        mesh, parent_cell, parent_facet = dolfinx.mesh.refine(
            mesh,
            edges_to_refine,
            partitioner=None,
            option=dolfinx.mesh.RefinementOption.parent_cell_and_facet,
        )

        # Transfer all cell tags
        for k, ct in cell_tags.items():
            cell_tags[k] = dolfinx.mesh.transfer_meshtag(ct, mesh, parent_cell)
            cell_tags[k].name = c_names[k]  # Restore names

        # Transfer all facet tags
        mesh.topology.create_connectivity(fdim, tdim)
        for k, ft in facet_tags.items():
            facet_tags[k] = dolfinx.mesh.transfer_meshtag(
                ft, mesh, parent_cell, parent_facet
            )
            facet_tags[k].name = f_names[k]  # Restore names

        # Print the updated cell count
        if mesh.comm.rank == 0:
            print(
                f"Refinement {i + 1}/{num_refinements}: cell count = {mesh.topology.index_map(tdim).size_global}"
            )

    return mesh, cell_tags, facet_tags


def write_mesh_with_tags(
    output_path: Path,
    mesh: dolfinx.mesh.Mesh,
    cell_tags: dict[str, dolfinx.mesh.MeshTags],
    facet_tags: dict[str, dolfinx.mesh.MeshTags],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with XDMFFile(MPI.COMM_WORLD, output_path, "w") as xdmf:
        xdmf.write_mesh(mesh)

        for tag in cell_tags.values():
            xdmf.write_meshtags(tag, mesh.geometry)

        for tag in facet_tags.values():
            xdmf.write_meshtags(tag, mesh.geometry)


@app.command()
def refine(
    input_mesh: Path = typer.Argument(
        ..., help="Path to the input XDMF mesh containing tags."
    ),
    output_mesh: Path = typer.Option(
        None,
        "-o",
        "--output",
        help="Path to the output refined XDMF mesh.",
    ),
    mesh_tags_name: str = typer.Option(
        "subdomains",
        help="Name of the main region marker MeshTags inside the input XDMF file (used to decide refinement areas).",
    ),
    refinement_steps: int = typer.Option(
        1,
        "--refine",
        "-r",
        help="Number of local refinement iterations to apply.",
    ),
):
    if output_mesh is None:
        output_mesh = input_mesh.with_name(input_mesh.stem + "_refined.xdmf")

    # Standard tags that we want to preserve through the refinement
    expected_cell_tags = [mesh_tags_name, "subdomains_ftetwild"]
    expected_facet_tags = ["boundaries"]

    mesh, cell_tags, facet_tags = read_all_tags(
        input_mesh, expected_cell_tags, expected_facet_tags
    )

    if "subdomains_ftetwild" not in cell_tags:
        raise ValueError(
            "Required cell tag 'subdomains_ftetwild' was not found in the input mesh."
        )

    # Apply highly localized refinement and transfer all loaded tags
    mesh, cell_tags, facet_tags = refine_mesh(
        mesh,
        cell_tags,
        facet_tags,
        primary_region_name=mesh_tags_name,
        num_refinements=refinement_steps,
    )

    # Save the result
    write_mesh_with_tags(output_mesh, mesh, cell_tags, facet_tags)

    if MPI.COMM_WORLD.rank == 0:
        typer.echo(f"Wrote refined tagged mesh to {output_mesh}")


if __name__ == "__main__":
    app()
