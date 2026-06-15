import nibabel
import pyvista as pv
import numpy as np
import skimage.morphology as skim
import skimage.draw
import scipy.ndimage as ndi
import typer
import yaml
from pathlib import Path

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
    # img should be a binary 3D np.array
    grid = pv.ImageData(dimensions=img.shape, spacing=resolution, origin=origin)
    mesh = grid.contour([0.5], img.flatten(order="F"), method="marching_cubes")
    
    # Silence PyVistaFutureWarning
    surf = mesh.extract_surface(algorithm=None)
    return surf


def save_ply(surf: pv.PolyData, output_path: Path):
    """Safely clear metadata and downcast to 32-bit before saving to PLY."""
    surf.clear_data()
    # Explicitly cast faces to 32-bit to silence the PLY 64-bit warning
    surf_32 = pv.PolyData(surf.points, surf.faces.astype(np.int32))
    pv.save_meshio(output_path, surf_32)


@app.command()
def extract_surfaces(
    input: Path = typer.Option(
        ..., "--input", "-i", help="Input file (white matter segmented MRI)"
    ),
    output_dir: Path = typer.Option(
        ..., "--output", "-o", help="Output directory"
    ),
    config_file: Path = typer.Option(..., "--config", "-c", help="Config file"),
):

    # Open config file
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

    # load white matter data
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

    skull_surf = extract_surface(skull_mask, resolution=resolution, origin=origin)
    skull_surf = skull_surf.smooth_taubin(n_iter=50, pass_band=0.01)
    skull_surf.points = nibabel.affines.apply_affine(seg.affine, skull_surf.points)
    skull_surf = skull_surf.clip_closed_surface(normal=clip_normal, origin=clip_origin)
    skull_surf.compute_normals(inplace=True, flip_normals=False)
    
    save_ply(skull_surf.scale(mm2m), output_dir / "skull.ply")

    # ---------------------------------------------------------
    # Parenchyma
    # ---------------------------------------------------------
    par_mask = skim.remove_small_objects(par_mask, max_size=999)
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

    # Compute Lateral Ventricle (LV) surface
    LV_mask = np.isin(img, FS_LV) + conn_V3_LV
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
    # Combine everything into a single continuous voxel mask
    V3_V4_mask = V3_mask + V4_mask + conn_V3_V4 + conn_V3_LV
    V3_V4_mask = skim.dilation(V3_V4_mask, footprint=skim.ball(1))

    # Extract and smooth as a single object so the aqueduct is perfectly uniform
    V3_V4_surf = extract_surface(V3_V4_mask, resolution=resolution, origin=origin)
    V3_V4_surf = V3_V4_surf.smooth_taubin(n_iter=30, pass_band=0.025)
    V3_V4_surf.points = nibabel.affines.apply_affine(seg.affine, V3_V4_surf.points)

    # Cut the unified surface perfectly in half at the Aqueduct
    V3_surf_conn = V3_V4_surf.clip_closed_surface(normal=clip_aq_normal, origin=clip_aq_origin)
    V3_surf_conn.compute_normals(inplace=True, flip_normals=False)
    save_ply(V3_surf_conn.scale(mm2m), output_dir / "V3_conn.ply")

    V4_surf = V3_V4_surf.clip_closed_surface(normal=clip_aq_normal_V4, origin=clip_aq_origin)
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

    # Extend ventricles to create tissue sheet and subtract cisterna magna outlet
    ventricle_extended = skim.dilation(ventricle_mask, footprint=skim.ball(2.5))
    ventricle_extended = np.logical_and(ventricle_extended, ventr_outlet == False)

    par_ventr_mask = par_mask + ventricle_extended
    par_surf = extract_surface(par_ventr_mask, resolution=resolution, origin=origin)
    par_surf = par_surf.smooth_taubin(n_iter=20, pass_band=0.025)
    par_surf.points = nibabel.affines.apply_affine(seg.affine, par_surf.points)
    par_surf = par_surf.clip_closed_surface(normal=clip_normal, origin=clip_origin)
    par_surf.compute_normals(inplace=True, flip_normals=False)
    
    save_ply(par_surf.scale(mm2m), output_dir / "parenchyma_incl_ventr.ply")

    return


if __name__ == "__main__":
    app()
    