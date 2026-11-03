import ufl
import dolfinx
import numpy as np
import warnings
from tqdm import tqdm
import typer
from pathlib import Path
from mpi4py import MPI
from dolfinx.io import XDMFFile

app = typer.Typer(
    help="Refine overconstrained cells in dolfinx meshes and preserve all tags."
)


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

        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

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


def create_new_cells(current_cell, needed_vertex, new_vertex):
    """
    Given a cell `[v0, v1, v2, v3]`, a required vertex `v2`, and a new vertex `v`,
    creates `[v1,v2,v3,v], [v0,v2,v3,v], [v0,v1,v2,v]`
    """
    include = np.argwhere(current_cell == needed_vertex)
    new_cells = np.full((3, 4), -1, dtype=np.int64)
    assert len(include) == 1
    mask = np.full(4, False, dtype=np.bool_)
    loop = np.arange(4)
    loop = np.delete(loop, include)
    for i, insert_pos in enumerate(loop):
        mask[:] = True
        mask[insert_pos] = False
        new_cells[i] = np.hstack([current_cell[mask], [new_vertex]])
    return new_cells


def create_facet_marker_structure(facet, new_vertex):
    """Split facet of current cell in 3 and create marker structure"""
    mask = np.full(3, True, dtype=np.bool_)
    new_facets = np.full((3, 3), -1, dtype=np.int64)
    for i in range(3):
        mask[:] = True
        mask[i] = False
        new_facets[i] = np.hstack([facet[mask], [new_vertex]])
    return new_facets


def fix_overconstrained_cells(
    mesh: dolfinx.mesh.Mesh,
    cell_tags: dict[str, dolfinx.mesh.MeshTags],
    facet_tags: dict[str, dolfinx.mesh.MeshTags],
) -> tuple[
    int,
    dolfinx.mesh.Mesh,
    dict[str, dolfinx.mesh.MeshTags],
    dict[str, dolfinx.mesh.MeshTags],
]:
    """Refine cells that are overconstrained and transfer all dictionaries of cell and facet tags."""

    assert mesh.comm.size == 1

    tdim = mesh.topology.dim
    num_vertices = mesh.topology.index_map(0).size_local

    mesh.topology.create_connectivity(tdim, tdim - 1)
    c_to_f = mesh.topology.connectivity(tdim, tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim - 1, 0)
    f_to_v = mesh.topology.connectivity(tdim - 1, 0)
    mesh.topology.create_connectivity(tdim, 0)
    c_to_v = mesh.topology.connectivity(tdim, 0)

    cell_map = mesh.topology.index_map(tdim)
    num_cells = cell_map.size_local

    facet_map = mesh.topology.index_map(tdim - 1)
    num_facets = facet_map.size_local
    facet_midpoints = dolfinx.mesh.compute_midpoints(
        mesh, mesh.topology.dim - 1, np.arange(num_facets, dtype=np.int32)
    )

    new_vertex_coordinates = []
    new_cells = []
    removed_cells = []
    removed_facets = []
    new_vertex_counter = 0

    # Data structures to handle arbitrary numbers of tags dynamically
    new_cell_marker_arrays = {k: [] for k in cell_tags.keys()}
    new_facet_marker_arrays = {k: [] for k in facet_tags.keys()}
    new_marked_facets = {k: [] for k in facet_tags.keys()}

    # O(1) facet lookups instead of slow np.flatnonzero loops
    facet_tag_lookups = {
        k: dict(zip(ft.indices, ft.values)) for k, ft in facet_tags.items()
    }

    for i in tqdm(range(num_cells), desc="Checking cells"):
        if i in removed_cells:
            continue

        facets = c_to_f.links(i)
        exterior_facets = np.array(
            [facet for facet in facets if len(f_to_c.links(facet)) == 1], dtype=np.int32
        )
        interior_facets = [facet for facet in facets if len(f_to_c.links(facet)) == 2]

        if not len(interior_facets) > 0:
            assert len(exterior_facets) == 4
            removed_cells.append(i)
            removed_facets.extend(exterior_facets)
            continue

        exterior_vertices = dolfinx.mesh.compute_incident_entities(
            mesh.topology, exterior_facets, mesh.topology.dim - 1, 0
        )
        assert len(np.unique(exterior_vertices)) == len(exterior_vertices)

        if len(exterior_vertices) == 4:
            all_cells = f_to_c.links(interior_facets[0])
            other_cell = np.setdiff1d(all_cells, [i])[0]
            if other_cell in removed_cells:
                warnings.warn("Cell already removed, should call this function again")
                continue

            current_vertices = c_to_v.links(i)
            interior_facet_vertices = f_to_v.links(interior_facets[0])

            # Get position of new vertex on midpoint of facet
            coord = facet_midpoints[interior_facets[0]]
            new_vertex_coordinates.append(coord)
            all_needs = np.setdiff1d(current_vertices, interior_facet_vertices)

            # Get all new sub-facets
            split_facets = create_facet_marker_structure(
                interior_facet_vertices, num_vertices + new_vertex_counter
            )

            # Check all facet tags to see if this split facet was originally tagged
            for k in facet_tags.keys():
                if interior_facets[0] in facet_tag_lookups[k]:
                    new_marked_facets[k].append(split_facets)
                    for _ in range(3):
                        new_facet_marker_arrays[k].append(
                            facet_tag_lookups[k][interior_facets[0]]
                        )

            removed_facets.append(interior_facets[0])

            # Split troublesome cell in 3
            assert len(all_needs) == 1
            include = np.argwhere(current_vertices == all_needs[0])
            assert len(include) == 1
            new_cells.append(
                create_new_cells(
                    current_vertices, all_needs[0], num_vertices + new_vertex_counter
                )
            )

            other_cell_connectivity = c_to_v.links(other_cell)
            other_needs = np.setdiff1d(other_cell_connectivity, interior_facet_vertices)
            new_cells.append(
                create_new_cells(
                    other_cell_connectivity,
                    other_needs[0],
                    num_vertices + new_vertex_counter,
                )
            )

            # Transfer cell tags dynamically
            for k, ct in cell_tags.items():
                for _ in range(3):
                    new_cell_marker_arrays[k].append(ct.values[i])
                for _ in range(3):
                    new_cell_marker_arrays[k].append(ct.values[other_cell])

            new_vertex_counter += 1
            removed_cells.append(i)
            removed_cells.append(other_cell)

    # ----------------------------------------------------
    # EARLY EXIT OPTIMIZATION:
    # If no cells needed to be split, return early.
    # ----------------------------------------------------
    if len(new_cells) == 0:
        return 0, mesh, cell_tags, facet_tags

    new_cells_as_array = np.vstack(new_cells)

    if len(removed_cells) == 0:
        removed_cells = np.array([], dtype=np.int64)
    else:
        removed_cells = np.unique(np.hstack(removed_cells).astype(np.int64))

    # Gather all cells
    remaining_cells = np.arange(num_cells, dtype=np.int32)
    remaining_cells = np.delete(remaining_cells, removed_cells)

    all_cells = dolfinx.mesh.entities_to_geometry(
        mesh, mesh.topology.dim, remaining_cells
    )
    all_new_cells = np.vstack([all_cells, new_cells_as_array]).astype(np.int64)

    new_vertex_numbering = np.unique(all_new_cells.flatten())
    all_to_reduced_num_vertices = np.full(
        num_vertices + new_vertex_counter, -1, dtype=np.int64
    )
    all_to_reduced_num_vertices[new_vertex_numbering] = np.arange(
        len(new_vertex_numbering)
    )
    all_new_cells = all_to_reduced_num_vertices[all_new_cells]

    # Gather all coordinates
    all_coords = np.zeros((num_vertices + new_vertex_counter, 3), dtype=np.float64)
    if new_vertex_counter > 0:
        all_new_vertex_coordinates = np.vstack(new_vertex_coordinates)
    else:
        all_new_vertex_coordinates = np.zeros((0, 3), dtype=np.float64)
    all_coords[:num_vertices, :] = mesh.geometry.x
    all_coords[num_vertices:, :] = all_new_vertex_coordinates
    all_coords = all_coords[new_vertex_numbering]

    # Initialize new mesh (USING 'e' INSTEAD OF 'domain' TO AVOID THE VALUE/TYPE ERRORS)
    ufl_domain = ufl.Mesh(mesh._ufl_domain.ufl_coordinate_element())
    new_mesh = dolfinx.mesh.create_mesh(
        mesh.comm, cells=all_new_cells, x=all_coords, e=ufl_domain
    )

    # ----------------------------------------------------
    # Rebuild all cell MeshTags
    # ----------------------------------------------------
    print("Transferring cell markers...")
    out_cell_tags = {}
    mask = np.full(num_cells, True, dtype=np.bool_)
    mask[removed_cells] = False

    for k, ct in cell_tags.items():
        new_values = ct.values[mask]
        tag_array = np.array(new_cell_marker_arrays[k], dtype=np.int32)
        all_values = np.hstack([new_values, tag_array])

        local_entities, local_values = dolfinx.io.distribute_entity_data(
            new_mesh, new_mesh.topology.dim, all_new_cells, all_values
        )
        new_mesh.topology.create_connectivity(mesh.topology.dim, 0)
        adj = dolfinx.graph.adjacencylist(local_entities)
        new_ct = dolfinx.mesh.meshtags_from_entities(
            new_mesh,
            new_mesh.topology.dim,
            adj,
            local_values.astype(np.int32, copy=False),
        )
        new_ct.name = k
        out_cell_tags[k] = new_ct

    # ----------------------------------------------------
    # Rebuild all facet MeshTags
    # ----------------------------------------------------
    print("Transferring facet markers...")
    out_facet_tags = {}
    assert np.allclose(f_to_v.offsets[1:] - f_to_v.offsets[:-1], 3)
    conn_arr = f_to_v.array.reshape(-1, 3)

    for k, ft in facet_tags.items():
        facets_to_copy = ft.indices.copy()
        facets_to_keep = np.invert(np.isin(facets_to_copy, removed_facets))
        new_facet_array = conn_arr[facets_to_copy[facets_to_keep], :]

        new_facet_values_array = ft.values[facets_to_keep]
        new_facet_values_array = np.hstack(
            new_facet_values_array, dtype=np.int32
        ).reshape(-1)

        # Merge preserved old facets with split new facets for this specific tag
        if len(new_marked_facets[k]) > 0:
            marked_facets_array = np.array(
                new_marked_facets[k], dtype=np.int64
            ).reshape(-1, 3)
            facet_connectivity = np.vstack([new_facet_array, marked_facets_array])
            tag_array = np.array(new_facet_marker_arrays[k], dtype=np.int32)
            facet_values = np.hstack([new_facet_values_array, tag_array])
        else:
            facet_connectivity = new_facet_array
            facet_values = new_facet_values_array

        facet_connectivity = all_to_reduced_num_vertices[facet_connectivity].astype(
            np.int64
        )
        assert (facet_connectivity != -1).all()

        local_entities, local_values = dolfinx.io.distribute_entity_data(
            new_mesh, new_mesh.topology.dim - 1, facet_connectivity, facet_values
        )

        new_mesh.topology.create_connectivity(mesh.topology.dim, 0)
        adj = dolfinx.graph.adjacencylist(local_entities)
        new_mesh.topology.create_connectivity(
            new_mesh.topology.dim - 1, new_mesh.topology.dim
        )

        new_ft = dolfinx.mesh.meshtags_from_entities(
            new_mesh,
            new_mesh.topology.dim - 1,
            adj,
            local_values.astype(np.int32, copy=False),
        )
        new_ft.name = k
        out_facet_tags[k] = new_ft

    return new_cells_as_array.shape[0], new_mesh, out_cell_tags, out_facet_tags


@app.command()
def fix(
    infile: Path = typer.Argument(..., help="The input XDMF mesh file"),
    outfile: Path = typer.Option(
        None, "-o", "--output", help="The output XDMF mesh file"
    ),
    cell_tag_name1: str = typer.Option(
        "subdomains", help="Name of the primary cell MeshTags"
    ),
    cell_tag_name2: str = typer.Option(
        "subdomains_ftetwild", help="Name of the secondary cell MeshTags"
    ),
    facet_tag_name: str = typer.Option("boundaries", help="Name of the facet MeshTags"),
    facet_tag_name2: str = typer.Option("boundaries_split", help="Name of the second facet MeshTags"),
):
    """Refine cells that are overconstrained iteratively and preserve all cell and facet markers."""

    if outfile is None:
        outfile = infile.with_name(infile.stem + "_fixed.xdmf")

    typer.echo(f"Reading in mesh: {infile}")

    expected_cell_tags = [cell_tag_name1, cell_tag_name2]
    expected_facet_tags = [facet_tag_name, facet_tag_name2]

    mesh, cell_tags, facet_tags = read_all_tags(
        infile, expected_cell_tags, expected_facet_tags
    )

    print(
        f"Number of cells before fix: {mesh.topology.index_map(mesh.topology.dim).size_global}"
    )

    iteration = 1
    while True:
        typer.echo(f"--- Fixing Iteration {iteration} ---")
        num_new_cells, mesh, cell_tags, facet_tags = fix_overconstrained_cells(
            mesh, cell_tags, facet_tags
        )

        if num_new_cells == 0:
            typer.echo("No more overconstrained cells found. Fix complete.")
            break

        iteration += 1

    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    typer.echo(f"Number of cells after fix: {num_cells}")
    typer.echo(f"Writing fixed mesh and all tags to file: {outfile}")

    # Write everything into a single, consolidated XDMF file
    with XDMFFile(mesh.comm, outfile, "w") as xdmf:
        xdmf.write_mesh(mesh)

        for ct in cell_tags.values():
            xdmf.write_meshtags(ct, mesh.geometry)

        for ft in facet_tags.values():
            xdmf.write_meshtags(ft, mesh.geometry)


if __name__ == "__main__":
    app()
