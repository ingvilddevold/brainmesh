#!/usr/bin/env python3
"""
Manage the full brain mesh generation pipeline:
1. Extract surfaces from MRI segmentation
2. Generate base volumetric mesh with fTetWild
3. Tag boundaries and interfaces
4. Fix overconstrained cells
5. Refine mesh locally in SSAS and aqueduct regions
"""

import subprocess
import shutil
from pathlib import Path
import typer

app = typer.Typer(help="Complete brain mesh generation pipeline")


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and report status."""
    typer.echo(f"\n{'=' * 60}")
    typer.echo(f" {description}")
    typer.echo(f"{'=' * 60}")
    typer.echo(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        typer.secho(f"Failed at: {description}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho(f"✓ Completed: {description}", fg=typer.colors.GREEN)
    return result.returncode


@app.command()
def full(
    subject_id: str = typer.Argument(
        ..., help="Subject identifier for naming files/folders (e.g., sub01)"
    ),
    input_mri: Path = typer.Argument(
        ..., help="Input segmented MRI file (e.g., sub-01_synthseg.nii.gz)"
    ),
    config_file: Path = typer.Argument(..., help="Mesh config YAML file"),
    skip_surfaces: bool = typer.Option(
        False, "--skip-surfaces", help="Skip surface extraction"
    ),
    skip_mesh: bool = typer.Option(
        False, "--skip-mesh", help="Skip volumetric mesh generation"
    ),
    skip_tags: bool = typer.Option(
        False, "--skip-tags", help="Skip boundary and interface tagging"
    ),
    skip_fix: bool = typer.Option(
        False, "--skip-fix", help="Skip overconstrained cell fixing"
    ),
    skip_refine: bool = typer.Option(
        False, "--skip-refine", help="Skip mesh refinement"
    ),
    refinement_steps: int = typer.Option(
        1, "--refine-steps", "-r", help="Number of refinement iterations"
    ),
):
    """Run the complete mesh generation pipeline."""

    if not input_mri.exists():
        typer.secho(f"Error: Input MRI not found: {input_mri}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not config_file.exists():
        typer.secho(f"Error: Config file not found: {config_file}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    surfaces_dir = Path("surfaces") / subject_id
    mesh_dir = Path("meshes") / subject_id
    work_dir = mesh_dir / "work"

    surfaces_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    # Only create work_dir if there are intermediate steps
    if not (skip_tags and skip_fix and skip_refine):
        work_dir.mkdir(parents=True, exist_ok=True)

    target_final_mesh = mesh_dir / f"{subject_id}.xdmf"

    if skip_tags and skip_fix and skip_refine:
        step2_out_dir = mesh_dir
        step2_out_file = target_final_mesh
    else:
        step2_out_dir = work_dir
        step2_out_file = work_dir / "work.xdmf"

    step3_out_file = (
        target_final_mesh
        if (skip_fix and skip_refine)
        else work_dir / f"{subject_id}_marked.xdmf"
    )
    step4_out_file = (
        target_final_mesh if skip_refine else work_dir / f"{subject_id}_fixed.xdmf"
    )
    step5_out_file = target_final_mesh

    typer.echo(f"Brain Mesh Generation Pipeline")
    typer.echo(f"Mesh name: {subject_id}")
    typer.echo(f"Input MRI: {input_mri.resolve()}")
    typer.echo(f"Config: {config_file.resolve()}")
    typer.echo(f"Surfaces directory: {surfaces_dir.resolve()}")
    typer.echo(f"Mesh target directory: {mesh_dir.resolve()}\n")

    # Step 1: Extract Surfaces
    if not skip_surfaces:
        run_command(
            [
                "python",
                "extract_surfaces.py",
                "--input",
                str(input_mri),
                "--config",
                str(config_file),
                "--output",
                str(surfaces_dir),
            ],
            "Step 1: Extract surfaces from MRI segmentation",
        )
    else:
        typer.secho("⏭ Skipping surface extraction", fg=typer.colors.YELLOW)
        if not surfaces_dir.exists() or not list(surfaces_dir.glob("*.ply")):
            typer.secho(
                f"Error: Required PLY surfaces not found in {surfaces_dir}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    # Pointer tracking for linear pipeline flow
    current_mesh_file = None

    # Step 2: Generate Mesh
    if not skip_mesh:
        run_command(
            [
                "python",
                "generate_mesh.py",
                "--surface-dir",
                str(surfaces_dir),
                "--output-dir",
                str(step2_out_dir),
            ],
            "Step 2: Generate base volumetric mesh with fTetWild",
        )
        current_mesh_file = step2_out_file
    elif step2_out_file.exists():
        current_mesh_file = step2_out_file
    elif target_final_mesh.exists():
        current_mesh_file = target_final_mesh

    if not current_mesh_file or not current_mesh_file.exists():
        typer.secho(
            "Error: Cannot find an input mesh to continue the pipeline.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # Step 3: Tag Boundaries and Interfaces
    if not skip_tags:
        run_command(
            [
                "python",
                "tag_boundaries.py",
                str(current_mesh_file),
                "--output",
                str(step3_out_file),
                "--configfile",
                str(config_file),
            ],
            "Step 3: Generate boundary and interface markers",
        )
        current_mesh_file = step3_out_file
    elif step3_out_file.exists():
        current_mesh_file = step3_out_file

    # Step 4: Fix Overconstrained Cells
    if not skip_fix:
        run_command(
            [
                "python",
                "fix_overconstrained_cells.py",
                str(current_mesh_file),
                "--output",
                str(step4_out_file),
            ],
            "Step 4: Fix overconstrained cells",
        )
        current_mesh_file = step4_out_file
    elif step4_out_file.exists():
        current_mesh_file = step4_out_file

    # Step 5: Refine Mesh
    if not skip_refine:
        run_command(
            [
                "python",
                "refine_mesh.py",
                str(current_mesh_file),
                "--output",
                str(step5_out_file),
                "--refine",
                str(refinement_steps),
            ],
            "Step 5: Refine mesh locally",
        )
        current_mesh_file = step5_out_file
    elif step5_out_file.exists():
        current_mesh_file = step5_out_file
    else:
        typer.secho("⏭ Skipping mesh refinement", fg=typer.colors.YELLOW)


    # Copy the config to ensure full reproducibility of the resulting mesh
    final_config = mesh_dir / f"{subject_id}{config_file.suffix}"
    shutil.copy(str(config_file), str(final_config))
    typer.echo(f"\n✓ Copied configuration settings to: {final_config.name}")

    # Delete the temporary work directory holding any intermediate steps
    # if work_dir.exists():
    #    shutil.rmtree(work_dir)
    #    typer.echo("✓ Cleaned up intermediate workspace files.")

    typer.secho(f"\nPipeline complete!", fg=typer.colors.GREEN)
    typer.echo(f"Final records directory: {mesh_dir.resolve()}")


@app.command()
def steps():
    """Show the pipeline steps."""
    steps_text = """
Pipeline Steps:

1. EXTRACT SURFACES
   python extract_surfaces.py --input MRI.nii.gz --config config.yml --output surfaces/sub01/

2. GENERATE BASE MESH
   python generate_mesh.py --surface-dir surfaces/sub01/ --output-dir meshes/sub01/work/
   (Note: If this is the last active step, it outputs directly to meshes/sub01/sub01.xdmf)

3. TAG BOUNDARIES & INTERFACES
   python tag_boundaries.py input.xdmf -o meshes/sub01/work/sub01_marked.xdmf --configfile config.yml
   (Generates both 'boundaries' and 'boundaries_split' inside the same file)

4. FIX OVERCONSTRAINED CELLS
   python fix_overconstrained_cells.py input.xdmf -o meshes/sub01/work/sub01_fixed.xdmf

5. REFINE MESH LOCALLY
   python refine_mesh.py input.xdmf -o meshes/sub01/sub01.xdmf --refine 1
   (Last step natively writes the final mesh to ensure HDF5 links are perfectly encoded)

6. CLEANUP & ARCHIVE CONFIG
   Copies 'config.yml' over as 'meshes/sub01/sub01.yml' for audit reproducibility.
   Deletes the 'meshes/sub01/work/' temporary workspace directory.

OR RUN FULL PIPELINE:
   python mesh_workflow.py full sub01 sub-01_synthseg.nii.gz config.yml
"""
    typer.echo(steps_text)


if __name__ == "__main__":
    app()
