import nibabel
import pyvista as pv
import numpy as np
import skimage.morphology as skim
import scipy.ndimage as ndi
import typer
import skimage
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
    surf = mesh.extract_geometry()
    surf.clear_data()
    return surf


def binary_smoothing(img, footprint=skim.ball(1)):
    openend = skim.binary_opening(img, footprint=footprint)
    return skim.binary_closing(openend, footprint=footprint)


def infer_output_dir(input: Path, config_file: Path) -> Path:
    config_name = config_file.stem
    if "sub-" in config_name:
        candidate = config_name.split("sub-")[-1].split("_")[0]
        if candidate:
            return Path("surfaces") / f"sub-{candidate}"
    input_name = input.stem
    return Path("surfaces") / input_name


@app.command()
def extract_surfaces(
    input: Path = typer.Option(..., "--input", "-i", help="Input file (white matter segmented MRI)"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
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

    # generate skull surface - everything but background
    # skull_mask = img > 0
    skull_mask = skim.remove_small_holes(img > 0, 1e3)
    par_mask = np.isin(img, FS_CSF + FS_VENTRICLE + [0]) == False

    skull_mask = np.logical_or(skull_mask, skim.binary_dilation(par_mask))

    skull_mask = skim.binary_dilation(skull_mask, skim.ball(2))

    skull_surf = extract_surface(skull_mask, resolution=resolution, origin=origin)
    skull_surf = skull_surf.smooth_taubin(n_iter=50, pass_band=0.01)
    skull_surf.points = nibabel.affines.apply_affine(seg.affine, skull_surf.points)
    skull_surf = skull_surf.clip_closed_surface(normal=clip_normal, origin=clip_origin)
    skull_surf.compute_normals(inplace=True, flip_normals=False)
    pv.save_meshio(Path(output_dir) / "skull.ply", skull_surf.scale(mm2m))

    # generate parenchyma surface - everything but CSF space
    par_mask = skim.remove_small_objects(par_mask, 1e3)
    par_mask = skim.remove_small_holes(par_mask, 1e3)
    # par_surf = extract_surface(par_mask, resolution=resolution, origin=origin)
    # par_surf = par_surf.smooth_taubin(n_iter=20, pass_band=0.025)
    # par_surf.points = nibabel.affines.apply_affine(seg.affine, par_surf.points)
    # par_surf = par_surf.clip_closed_surface(normal=clip_normal, origin=clip_origin)
    # par_surf.compute_normals(inplace=True, flip_normals=False)
    # pv.save_meshio(Path(output) / "parenchyma.ply", par_surf.scale(mm2m))

    # compute connection between V3 and V4:
    V3_mask = np.isin(img, FS_V3)
    V4_mask = np.isin(img, FS_V4)
    pointa = get_closest_point(V3_mask, V4_mask, img)
    pointb = get_closest_point(V4_mask, V3_mask, img)

    # add a line between the closest points to connect V3 and V4
    line = np.array(skimage.draw.line_nd(pointa, pointb, endpoint=True))
    conn_V3_V4 = np.zeros(img.shape)
    i, j, k = line
    conn_V3_V4[i, j, k] = 1
    conn_V3_V4 = skim.binary_dilation(conn_V3_V4, footprint=skim.ball(2))

    # compute ventricle surface
    # ventricle_mask = np.isin(img, FS_VENTRICLE) + conn_V3_V4
    # ventricle_mask = skim.binary_dilation(ventricle_mask, footprint=skim.ball(1))
    # ventricle_surf = extract_surface(
    #    ventricle_mask, resolution=resolution, origin=origin
    # )
    # ventricle_surf = ventricle_surf.smooth_taubin(n_iter=20, pass_band=0.025)
    # ventricle_surf.points = nibabel.affines.apply_affine(
    #    seg.affine, ventricle_surf.points
    # )
    # ventricle_surf = ventricle_surf.clip_closed_surface(
    #    normal=clip_normal, origin=clip_origin
    # )
    # ventricle_surf.compute_normals(inplace=True, flip_normals=False)
    # pv.save_meshio(Path(output) / "ventricles.ply", ventricle_surf.scale(mm2m))

    # Intraventricular foramen / foramen of Monro
    # Using 2 closest to connect both sides
    # compute connection between LV and V3:
    LV = np.isin(img, FS_LV)
    N_points = 10
    pointsa = get_N_closest_points(LV, V3_mask, img, N_points)
    pointsb = get_N_closest_points(V3_mask, LV, img, N_points)

    # add a line between the closest points to connect V3 and LV
    conn_V3_LV = np.zeros(img.shape)
    for p in range(N_points):
        line = np.array(skimage.draw.line_nd(pointsa[p], pointsb[p], endpoint=True))
        i, j, k = line
        conn_V3_LV[i, j, k] = 1
    conn_V3_LV = skim.binary_dilation(conn_V3_LV, footprint=skim.ball(1))

    # compute lateral ventricle surface
    LV_mask = np.isin(img, FS_LV) + conn_V3_LV
    LV_mask = skim.binary_dilation(LV_mask, footprint=skim.ball(1))
    LV_surf = extract_surface(LV_mask, resolution=resolution, origin=origin)
    LV_surf = LV_surf.smooth_taubin(n_iter=20, pass_band=0.025)
    LV_surf.points = nibabel.affines.apply_affine(seg.affine, LV_surf.points)
    LV_surf = LV_surf.clip_closed_surface(normal=clip_normal, origin=clip_origin)
    LV_surf.compute_normals(inplace=True, flip_normals=False)
    pv.save_meshio(Path(output_dir) / "LV.ply", LV_surf.scale(mm2m))

    # compute V3 and V4 surface
    # V34_mask = np.isin(img, [FS_V3, FS_V4]) + conn_V3_LV
    # V34_mask = skim.binary_dilation(V34_mask, footprint=skim.ball(1))
    # V34_surf = extract_surface(V34_mask, resolution=resolution, origin=origin)
    # V34_surf = V34_surf.smooth_taubin(n_iter=20, pass_band=0.025)
    # V34_surf.points = nibabel.affines.apply_affine(seg.affine, V34_surf.points)
    # V34_surf = V34_surf.clip_closed_surface(normal=clip_normal, origin=clip_origin)
    # V34_surf.compute_normals(inplace=True, flip_normals=False)
    # pv.save_meshio(Path(output) / "V34.ply", V34_surf.scale(mm2m))

    # separate files for V3 and V4
    V3_mask1 = V3_mask
    V3_mask1 = skim.binary_dilation(V3_mask1, footprint=skim.ball(1))
    V3_surf = extract_surface(V3_mask1, resolution=resolution, origin=origin)
    V3_surf = V3_surf.smooth_taubin(n_iter=20, pass_band=0.025)
    V3_surf.points = nibabel.affines.apply_affine(seg.affine, V3_surf.points)
    # V3_surf = V3_surf.clip_closed_surface(normal=clip_normal, origin=clip_origin)
    V3_surf.compute_normals(inplace=True, flip_normals=False)
    pv.save_meshio(Path(output_dir) / "V3.ply", V3_surf.scale(mm2m))

    V3_mask2 = V3_mask #+ conn_V3_V4 + conn_V3_LV
    V3_mask2 = skim.binary_dilation(V3_mask2, footprint=skim.ball(1))
    V3_mask2 = V3_mask2 + conn_V3_V4 + conn_V3_LV # add conncetions after dilation
    V3_surf = extract_surface(V3_mask2, resolution=resolution, origin=origin)
    V3_surf = V3_surf.smooth_taubin(n_iter=30, pass_band=0.025)
    V3_surf.points = nibabel.affines.apply_affine(seg.affine, V3_surf.points)
    # clip to get aqueduct cross-section
    V3_surf = V3_surf.clip_closed_surface(normal=clip_aq_normal, origin=clip_aq_origin)
    V3_surf.compute_normals(inplace=True, flip_normals=False)
    pv.save_meshio(Path(output_dir) / "V3_conn.ply", V3_surf.scale(mm2m))

    V4_mask1 = V4_mask #+ V3_mask2
    V4_mask1 = skim.binary_dilation(V4_mask1, footprint=skim.ball(1.5))
    V4_mask1 = V4_mask1 + conn_V3_V4 + V3_mask2 # Note: adding V3_mask2 to get smooth cut
    V4_surf = extract_surface(V4_mask1, resolution=resolution, origin=origin)
    V4_surf = V4_surf.smooth_taubin(n_iter=30, pass_band=0.025)
    V4_surf.points = nibabel.affines.apply_affine(seg.affine, V4_surf.points)
    # clip to get an aqueduct cross-section
    V4_surf = V4_surf.clip_closed_surface(normal=clip_aq_normal_V4, origin=clip_aq_origin)

    V4_surf.compute_normals(inplace=True, flip_normals=False)
    pv.save_meshio(Path(output_dir) / "V4.ply", V4_surf.scale(mm2m))

    # compute a layer of parenchymal tissue around the ventricles
    # to get a watertight ventricular system and
    # generate combined mask of ventricles and parenchyma

    # first find lowest point of V4 to generate an outlet of the tissue sheet into
    # the cisterna magna
    ventricle_mask = V4_mask1 + V3_mask2 + LV_mask

    ventr_idx = np.argwhere(ventricle_mask)
    ind = np.argsort(ventr_idx[:, 2])
    lowest_ventr_point = ventr_idx[ind][0]
    ventr_outlet = np.zeros(img.shape)
    ventr_outlet[tuple(lowest_ventr_point)] = 1
    ventr_outlet = skim.binary_dilation(ventr_outlet, footprint=skim.ball(5))
    outlet_surf = extract_surface(ventr_outlet, resolution=resolution, origin=origin)
    # outlet_surf = outlet_surf.smooth_taubin(n_iter=20, pass_band=0.025)
    outlet_surf.points = nibabel.affines.apply_affine(seg.affine, outlet_surf.points)
    # pv.save_meshio(Path(output) / "ventr_outlet.ply", outlet_surf.scale(mm2m))

    # extend ventricles to create tissue sheet and subtract cisterna magna outlet
    ventricle_extended = skim.binary_dilation(ventricle_mask, footprint=skim.ball(2.5))
    ventricle_extended = np.logical_and(ventricle_extended, ventr_outlet == False)

    # finally, generate combined mask of parenchyma and ventricles
    par_ventr_mask = par_mask + ventricle_extended
    par_surf = extract_surface(par_ventr_mask, resolution=resolution, origin=origin)
    par_surf = par_surf.smooth_taubin(n_iter=20, pass_band=0.025)
    par_surf.points = nibabel.affines.apply_affine(seg.affine, par_surf.points)
    par_surf = par_surf.clip_closed_surface(normal=clip_normal, origin=clip_origin)
    par_surf.compute_normals(inplace=True, flip_normals=False)
    pv.save_meshio(Path(output_dir) / "parenchyma_incl_ventr.ply", par_surf.scale(mm2m))

    return


if __name__ == "__main__":
    app()
