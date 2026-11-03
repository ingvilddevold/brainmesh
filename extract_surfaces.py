from pathlib import Path
from typing import Annotated

import nibabel
import numpy as np
import pyvista as pv
import scipy.ndimage as ndi
import skimage.draw
import skimage.morphology as skim
import typer
import yaml

app = typer.Typer()

FS_V3 = 14
FS_V4 = 15
FS_LV = [4, 43]
FS_VENTRICLE = [FS_V3, FS_V4] + FS_LV
FS_CSF = [24]

mm2m = 1e-3  # unit conversion factor


def get_closest_point(a, b, img):
    dist = ndi.distance_transform_edt(a == False)
    dist[b == False] = np.inf
    minidx = np.unravel_index(np.argmin(dist), img.shape)
    return minidx


def get_N_closest_points(a, b, img, N):
    dist = ndi.distance_transform_edt(a == False)
    dist[b == False] = np.inf
    ix_sort = np.argsort(dist.flatten())
    minidx = np.unravel_index(ix_sort[:N], img.shape)
    return np.array(minidx).T


def extract_surface(img, resolution=(1, 1, 1), origin=(0, 0, 0)):
    grid = pv.ImageData(dimensions=img.shape, spacing=resolution, origin=origin)
    mesh = grid.contour([0.5], img.flatten(order="F"), method="marching_cubes")

    surf = mesh.extract_surface(algorithm=None)
    return surf


def remove_degenerate_triangles(
    surf: pv.PolyData, area_thresh: float = 1e-11, max_iter: int = 20
) -> pv.PolyData:
    """Locally collapse zero/near-zero-area triangles by welding each one's
    closest vertex pair, rather than a global tolerance-based clean() (which
    can also weld unrelated, legitimately close-together points elsewhere in
    a complex surface like skull, and break watertightness there). Marching
    cubes + Taubin smoothing routinely bakes in a handful of these
    degenerate slivers; pyvista's manifold check doesn't catch them (they
    can still satisfy the "each edge shared by exactly 2 faces" rule while
    having ~0 area), but fTetWild's CSG boolean does -- it silently returns
    0 cells if any survive."""
    for _ in range(max_iter):
        faces = surf.faces.reshape(-1, 4)[:, 1:].copy()
        pts = surf.points.copy()
        v0, v1, v2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        bad = np.where(area < area_thresh)[0]
        if len(bad) == 0:
            break
        remap = np.arange(len(pts))
        for fi in bad:
            a, b, c = faces[fi]
            ra, rb, rc = remap[a], remap[b], remap[c]
            d_ab = np.linalg.norm(pts[ra] - pts[rb])
            d_bc = np.linalg.norm(pts[rb] - pts[rc])
            d_ca = np.linalg.norm(pts[rc] - pts[ra])
            _, i, j = min(
                [(d_ab, ra, rb), (d_bc, rb, rc), (d_ca, rc, ra)], key=lambda x: x[0]
            )
            lo, hi = min(i, j), max(i, j)
            remap[remap == hi] = lo
        new_faces = remap[faces]
        keep = (
            (new_faces[:, 0] != new_faces[:, 1])
            & (new_faces[:, 1] != new_faces[:, 2])
            & (new_faces[:, 0] != new_faces[:, 2])
        )
        new_faces = new_faces[keep]
        new_cells = (
            np.hstack([np.full((len(new_faces), 1), 3), new_faces])
            .astype(np.int64)
            .ravel()
        )
        surf = pv.PolyData(pts, new_cells).clean(tolerance=0.0, absolute=True)
    return surf


def save_ply(surf: pv.PolyData, output_path: Path, hole_size: float = 0.02):
    """Produce a watertight, non-degenerate surface before saving, so the
    mesh generation step's fTetWild CSG boolean gets valid input:
    1. Collapse zero/near-zero-area triangles baked in by marching cubes +
       Taubin smoothing (see remove_degenerate_triangles).
    2. Seal any real unclosed tears left by clip_closed_surface at a
       ragged/patchy cut.
    3. Collapse again, since fill_holes's own new triangulation can
       introduce fresh degenerate slivers of its own.
    """
    surf = remove_degenerate_triangles(surf)
    surf = surf.fill_holes(hole_size)
    surf = remove_degenerate_triangles(surf)
    surf.clear_data()
    surf_32 = pv.PolyData(surf.points, surf.faces.astype(np.int32))
    pv.save_meshio(output_path, surf_32)


@app.command()
def extract_surfaces(
    input: Annotated[
        Path, typer.Option("--input", "-i", help="Input file (white matter segmented MRI)")
    ],
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output directory")],
    config_file: Annotated[Path, typer.Option("--config", "-c", help="Config file")],
    csf_gap: Annotated[
        int,
        typer.Option(
            "--csf-gap",
            help=(
                "Minimum CSF gap [voxels] kept between the parenchyma and the skull surface (0 disables). "
                "2 gives a closest approach of about 1.1 mm at no extra CSF volume; 3 gives about 2.2 mm but "
                "adds 10-13 ml. Use 3 for a subject that fails --min-gap-mm."
            ),
        ),
    ] = 2,
    min_gap_mm: Annotated[
        float,
        typer.Option(
            "--min-gap-mm",
            help="Fail if the parenchyma-skull distance, away from the cut plane, ends up below this [mm] (0 disables).",
        ),
    ] = 1.0,
):

    with open(config_file) as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.UnsafeLoader)

    config = mesh_config["surface"]

    clip_origin = (
        config["clip_origin"]["x"],
        config["clip_origin"]["y"],
        config["clip_origin"]["z"],
    )
    clip_normal = (
        config["clip_normal"]["x"],
        config["clip_normal"]["y"],
        config["clip_normal"]["z"],
    )
    clip_aq_origin = (
        config["clip_aq_origin"]["x"],
        config["clip_aq_origin"]["y"],
        config["clip_aq_origin"]["z"],
    )
    clip_aq_normal = (
        config["clip_aq_normal"]["x"],
        config["clip_aq_normal"]["y"],
        config["clip_aq_normal"]["z"],
    )
    clip_aq_normal_V4 = (
        -config["clip_aq_normal"]["x"],
        -config["clip_aq_normal"]["y"],
        -config["clip_aq_normal"]["z"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    pad_width = 5
    seg = nibabel.load(input)
    img = np.pad(seg.get_fdata(), pad_width=pad_width)
    resolution = np.array(seg.header["pixdim"][1:4])
    origin = -resolution * pad_width

    # ---------------------------------------------------------
    # Skull Surface
    # ---------------------------------------------------------
    skull_mask = skim.remove_small_holes(img > 0, max_size=999)
    par_mask = np.isin(img, FS_CSF + FS_VENTRICLE + [0]) == False

    skull_mask = np.logical_or(skull_mask, skim.dilation(par_mask))
    skull_mask = skim.dilation(skull_mask, skim.ball(2))

    # The skull surface is extracted at the end, once the parenchyma mask exists (see below).

    # ---------------------------------------------------------
    # Parenchyma
    # ---------------------------------------------------------
    par_mask = skim.remove_small_holes(par_mask, max_size=999)

    # ---------------------------------------------------------
    # Connections and Ventricles
    # ---------------------------------------------------------
    V3_mask = np.isin(img, FS_V3)
    V4_mask = np.isin(img, FS_V4)
    pointa = get_closest_point(V3_mask, V4_mask, img)
    pointb = get_closest_point(V4_mask, V3_mask, img)

    # Connect V3 and V4
    line = np.array(skimage.draw.line_nd(pointa, pointb, endpoint=True))
    conn_V3_V4 = np.zeros(img.shape)
    i, j, k = line
    conn_V3_V4[i, j, k] = 1
    conn_V3_V4 = skim.dilation(conn_V3_V4, footprint=skim.ball(2))

    # Connect LV and V3 (Foramen of Monro)
    LV = np.isin(img, FS_LV)
    N_points = 10
    pointsa = get_N_closest_points(LV, V3_mask, img, N_points)
    pointsb = get_N_closest_points(V3_mask, LV, img, N_points)

    conn_V3_LV = np.zeros(img.shape)
    for p in range(N_points):
        line = np.array(skimage.draw.line_nd(pointsa[p], pointsb[p], endpoint=True))
        i, j, k = line
        conn_V3_LV[i, j, k] = 1
    conn_V3_LV = skim.dilation(conn_V3_LV, footprint=skim.ball(1))

    LV_mask = LV + conn_V3_LV
    LV_mask = skim.dilation(LV_mask, footprint=skim.ball(1))

    LV_surf = extract_surface(LV_mask, resolution=resolution, origin=origin)
    LV_surf = LV_surf.smooth_taubin(n_iter=20, pass_band=0.025)
    LV_surf.points = nibabel.affines.apply_affine(seg.affine, LV_surf.points)
    LV_surf = LV_surf.clip_closed_surface(normal=clip_normal, origin=clip_origin)
    LV_surf.compute_normals(inplace=True, flip_normals=False)

    save_ply(LV_surf.scale(mm2m), output_dir / "LV.ply")

    # ---------------------------------------------------------
    # Unified V3 & V4 Surface generation
    # ---------------------------------------------------------
    # Use boolean OR (|) to prevent integer summation artifacts at overlaps
    V3_V4_mask = V3_mask | V4_mask | (conn_V3_V4 > 0) | (conn_V3_LV > 0)

    # 1. Morphological closing to seal deep "bites" left by choroid plexus
    V3_V4_mask = skim.closing(V3_V4_mask, footprint=skim.ball(2))

    # 2. Fill any fully enclosed internal holes
    V3_V4_mask = ndi.binary_fill_holes(V3_V4_mask)

    # 3. Final slight dilation to ensure a thick, manifold topology for clipping
    V3_V4_mask = skim.dilation(V3_V4_mask, footprint=skim.ball(1))

    # Extract and smooth
    V3_V4_surf = extract_surface(V3_V4_mask, resolution=resolution, origin=origin)
    V3_V4_surf = V3_V4_surf.smooth_taubin(n_iter=30, pass_band=0.025)
    V3_V4_surf.points = nibabel.affines.apply_affine(seg.affine, V3_V4_surf.points)

    # The clipping will now successfully generate closed "caps" on the cut faces!
    V3_surf_conn = V3_V4_surf.clip_closed_surface(
        normal=clip_aq_normal, origin=clip_aq_origin
    )
    V3_surf_conn.compute_normals(inplace=True, flip_normals=False)
    save_ply(V3_surf_conn.scale(mm2m), output_dir / "V3_conn.ply")

    V4_surf = V3_V4_surf.clip_closed_surface(
        normal=clip_aq_normal_V4, origin=clip_aq_origin
    )
    V4_surf.compute_normals(inplace=True, flip_normals=False)
    save_ply(V4_surf.scale(mm2m), output_dir / "V4.ply")

    # ---------------------------------------------------------
    # Parenchyma + Ventricles Combined
    # ---------------------------------------------------------
    ventricle_mask = V3_V4_mask + LV_mask

    ventr_idx = np.argwhere(ventricle_mask)
    ind = np.argsort(ventr_idx[:, 2])
    lowest_ventr_point = ventr_idx[ind][0]
    ventr_outlet = np.zeros(img.shape)
    ventr_outlet[tuple(lowest_ventr_point)] = 1
    ventr_outlet = skim.dilation(ventr_outlet, footprint=skim.ball(5))

    ventricle_extended = skim.dilation(ventricle_mask, footprint=skim.ball(2.5))
    ventricle_extended = np.logical_and(ventricle_extended, ventr_outlet == False)

    par_ventr_mask = par_mask + ventricle_extended

    par_surf = extract_surface(par_ventr_mask, resolution=resolution, origin=origin)
    par_surf = par_surf.smooth_taubin(n_iter=20, pass_band=0.025)
    par_surf.points = nibabel.affines.apply_affine(seg.affine, par_surf.points)
    par_surf = par_surf.clip_closed_surface(normal=clip_normal, origin=clip_origin)
    par_surf.compute_normals(inplace=True, flip_normals=False)

    save_ply(par_surf.scale(mm2m), output_dir / "parenchyma_incl_ventr.ply")

    # ---------------------------------------------------------
    # Skull Surface (built last so it can keep a CSF gap around the parenchyma)
    # ---------------------------------------------------------
    # The skull mask above (intracranial volume + 2 voxels) leaves almost no CSF where the brain reaches
    # the intracranial boundary (skull base near the sella), and the smoothing below plus fTetWild's
    # snapping close what is left, so parenchyma ends up on the outer boundary. Guarantee `csf_gap`
    # voxels around the parenchyma. This only adds voxels where the layer is thinner than that.
    if csf_gap > 0:
        added = skim.dilation(par_ventr_mask, skim.ball(csf_gap)) & ~skull_mask
        print(
            f"CSF gap: adding {int(added.sum())} voxels to the skull mask ({csf_gap}-voxel shell around the parenchyma)"
        )
        skull_mask = skull_mask | added

    skull_surf = extract_surface(skull_mask, resolution=resolution, origin=origin)
    skull_surf = skull_surf.smooth_taubin(n_iter=50, pass_band=0.01)
    skull_surf.points = nibabel.affines.apply_affine(seg.affine, skull_surf.points)
    skull_surf = skull_surf.clip_closed_surface(normal=clip_normal, origin=clip_origin)
    skull_surf.compute_normals(inplace=True, flip_normals=False)

    save_ply(skull_surf.scale(mm2m), output_dir / "skull.ply")

    # Check the resulting CSF layer (both surfaces are still in mm here). The bottom cut is excluded:
    # the parenchyma reaches the boundary there by design (brainstem).
    if min_gap_mm > 0:
        # Closest approach = unsigned distance from parenchyma points to the skull surface. The sign is not
        # used: a point touching or crossing the skull surface has a distance near zero either way, and the
        # implicit distance's own sign follows the surface normals, which are not always outward.
        signed = -np.abs(
            par_surf.compute_implicit_distance(skull_surf)["implicit_distance"]
        )
        normal = np.asarray(clip_normal, dtype=float)
        normal /= np.linalg.norm(normal)
        above_cut = (par_surf.points - np.asarray(clip_origin)) @ normal > 3.0
        gap = -signed[above_cut]
        worst = int(np.argmin(gap))
        where = np.round(par_surf.points[above_cut][worst], 1)
        print(
            f"Minimum parenchyma-skull gap away from the cut plane: {gap[worst]:.2f} mm at {where} mm"
        )
        if gap[worst] < min_gap_mm:
            typer.secho(
                f"ERROR: the CSF layer is only {gap[worst]:.2f} mm thick at {where} mm "
                f"(minimum {min_gap_mm} mm). Increase --csf-gap.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)



if __name__ == "__main__":
    app()
