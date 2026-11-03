from pathlib import Path
from typing import Annotated

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import skimage.measure
import typer

app = typer.Typer()

# Define the standard FreeSurfer fluid labels to track
# 4=Left-Lateral, 43=Right-Lateral, 14=3rd-Ventricle, 15=4th-Ventricle
# 24=CSF, 5=Left-Inf-Lat-Vent, 44=Right-Inf-Lat-Vent
FLUID_LABELS = [4, 5, 14, 15, 24, 43, 44]


@app.command()
def clean_ventricle_artifacts(
    input_file: Annotated[Path, typer.Argument(help="Input path to the raw synthseg/parc NIfTI file")],
    output_file: Annotated[Path, typer.Argument(help="Output path for the cleaned NIfTI file")],
):
    print(f"Loading {input_file}...")
    img = nib.load(input_file)
    data = img.get_fdata().astype(np.int32)
    cleaned_data = np.copy(data)

    bad_voxels_mask = np.zeros_like(data, dtype=bool)

    print("\nChecking ventricles for disconnected components...")

    # Discard the Inferior Lateral Ventricles entirely (Labels 5 and 44)
    for inf_horn in [5, 44]:
        mask = data == inf_horn
        if np.any(mask):
            bad_voxels_mask |= mask
            print(
                f" -> Label {inf_horn:2d}: Discarding entirely ({np.sum(mask)} voxels)."
            )

    # Process the main ventricles to find floating pieces
    for v_label in [4, 43, 14, 15]:
        mask = data == v_label
        if not np.any(mask):
            continue

        # Use connectivity=1 to ensure diagonal-only touching isn't counted as a connection
        labeled_mask, num_features = skimage.measure.label(
            mask, connectivity=1, return_num=True
        )

        if num_features <= 1:
            print(f" -> Label {v_label:2d}: 0 disconnected pieces.")
            continue

        # Identify the largest component (the true ventricle body)
        counts = np.bincount(labeled_mask.flat)
        counts[0] = 0
        largest_cc = np.argmax(counts)

        # Flag everything that is NOT the main body
        flying_pieces = (labeled_mask > 0) & (labeled_mask != largest_cc)
        bad_voxels_mask |= flying_pieces

        num_bad = np.sum(flying_pieces)
        print(
            f" -> Label {v_label:2d}: Found {num_features - 1} disconnected pieces ({num_bad} voxels)."
        )

    if not np.any(bad_voxels_mask):
        print("\nNo disconnected pieces found. Saving identical image.")
        nib.save(nib.Nifti1Image(cleaned_data, img.affine, img.header), output_file)
        return

    print(f"\nTotal disconnected voxels to reassign: {np.sum(bad_voxels_mask)}")

    # ---------------------------------------------------------
    # Reassign artifacts to the nearest valid solid tissue
    # ---------------------------------------------------------
    solid_mask = (data > 0) & (~np.isin(data, FLUID_LABELS)) & (~bad_voxels_mask)

    print("Computing 3D distance transform to nearest solid tissue...")
    _, indices = ndi.distance_transform_edt(~solid_mask, return_indices=True)

    nearest_solid_labels = data[tuple(indices[:, bad_voxels_mask])]
    cleaned_data[bad_voxels_mask] = nearest_solid_labels

    print(f"\nSaving cleaned parcellation to {output_file}...")
    out_img = nib.Nifti1Image(cleaned_data, img.affine, img.header)
    nib.save(out_img, output_file)

    labels_new, counts_new = np.unique(nearest_solid_labels, return_counts=True)
    reassigned_to = sorted(zip(labels_new, counts_new), key=lambda t: -t[1])
    print("Reassigned to: " + ", ".join(f"{lab} ({n})" for lab, n in reassigned_to[:6]))


if __name__ == "__main__":
    app()
