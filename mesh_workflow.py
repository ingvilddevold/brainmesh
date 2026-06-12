#!/usr/bin/env python3
"""
Manage the full brain mesh generation pipeline:
1. Extract surfaces from MRI segmentation
2. Generate mesh with fTetWild
3. Fix overconstrained cells
4. Refine mesh locally in SSAS and aqueduct regions
"""

import subprocess
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
    skip_mesh: bool = typer.Option(False, "--skip-mesh", help="Skip mesh generation"),
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

    # -------------------------------------------------------------------
    # STRICT DIRECTORY STRUCTURE
    # -------------------------------------------------------------------
    surfaces_dir = Path("surfaces") / subject_id
    mesh_dir = Path("meshes") / subject_id

    surfaces_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Brain Mesh Generation Pipeline")
    typer.echo(f"Subject ID: {subject_id}")
    typer.echo(f"Input MRI: {input_mri.resolve()}")
    typer.echo(f"Config: {config_file.resolve()}")
    typer.echo(f"Surfaces directory: {surfaces_dir.resolve()}")
    typer.echo(f"Mesh directory: {mesh_dir.resolve()}\n")

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

    current_mesh_file = mesh_dir / f"{subject_id}.xdmf"

    # Step 2: Generate Mesh
    if not skip_mesh:
        run_command(
            [
                "python",
                "generate_mesh.py",
                "--surface-dir",
                str(surfaces_dir),
                "--output-dir",
                str(mesh_dir),
                "--configfile",
                str(config_file),
            ],
            "Step 2: Generate volumetric mesh with fTetWild",
        )
    else:
        typer.secho("⏭ Skipping mesh generation", fg=typer.colors.YELLOW)
        if not current_mesh_file.exists():
            typer.secho(
                f"Error: Expected mesh file {current_mesh_file} not found to resume pipeline.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    # Step 3: Fix Overconstrained Cells
    if not skip_fix:
        fixed_mesh = mesh_dir / f"{subject_id}_fixed.xdmf"
        run_command(
            [
                "python",
                "fix_overconstrained_cells.py",
                str(current_mesh_file),
                "--output",
                str(fixed_mesh),
            ],
            "Step 3: Fix overconstrained cells",
        )
        current_mesh_file = fixed_mesh
    else:
        typer.secho("⏭ Skipping overconstrained cell fixing", fg=typer.colors.YELLOW)

    # Step 4: Refine Mesh
    if not skip_refine:
        final_mesh = mesh_dir / f"{subject_id}_refined.xdmf"
        run_command(
            [
                "python",
                "refine_mesh.py",
                str(current_mesh_file),
                "--output",
                str(final_mesh),
                "--refine",
                str(refinement_steps),
            ],
            "Step 4: Refine mesh locally",
        )
        current_mesh_file = final_mesh
    else:
        typer.secho("⏭ Skipping mesh refinement", fg=typer.colors.YELLOW)

    typer.secho(f"\nPipeline complete!", fg=typer.colors.GREEN)
    typer.echo(f"Final mesh: {current_mesh_file.resolve()}")


@app.command()
def steps():
    """Show the pipeline steps."""
    steps_text = """
Pipeline Steps:

1. EXTRACT SURFACES
   python extract_surfaces.py --input MRI.nii.gz --config_file config.yml --output-dir surfaces/sub01/
   
   Extracts PLY surfaces from segmented MRI:
   - skull, parenchyma_incl_ventr, LV, V3, V4 ventricles
   
   Output: surfaces/sub01/*.ply

2. GENERATE MESH
   python generate_mesh.py --surface-dir surfaces/sub01/ --configfile config.yml --output-dir mesh/sub01/
   
   Uses fTetWild to tetrahedralize the brain geometry
   Creates subdomain markers (fluid/porous) and boundary markers
   
   Output: mesh/sub01/sub01.xdmf

3. FIX OVERCONSTRAINED CELLS
   python fix_overconstrained_cells.py fix mesh/sub01/sub01.xdmf -o mesh/sub01/sub01_fixed.xdmf
   
   Refines cells where all 4 vertices are on the boundary
   Preserves all markers during refinement
   
   Output: mesh/sub01/sub01_fixed.xdmf 

4. REFINE MESH LOCALLY
   python refine_mesh.py refine mesh/sub01/sub01_fixed.xdmf -o mesh/sub01/sub01_refined.xdmf --refine 1
   
   Locally refines mesh exclusively around the Aqueduct and SSAS
   Preserves all existing mesh tags correctly.
   
   Output: mesh/sub01/sub01_refined.xdmf 

OR RUN FULL PIPELINE:
   python mesh_workflow.py full sub01 sub-01_synthseg.nii.gz config.yml -o output_dir/
"""
    typer.echo(steps_text)


if __name__ == "__main__":
    app()
