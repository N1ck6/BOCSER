"""
recalc_ensemble_xtb2.py — recalculate energies for all conformers in an XYZ
ensemble at XTB2 level using ORCA via Slurm, then write a new ensemble file
with XTB2 energies (relative to ensemble minimum, in kcal/mol).

Output: NonBR_energies_res.xyz  (same folder as input ensemble)

Usage (from /home/xray/Nicki/BOCSER/):
    python recalc_ensemble_xtb2.py

All paths are configured in the CONFIG block below.

nohup python tests/NonBR_copy/recalc_ensemble.py > "tests/NonBR_copy/run.log" 2>&1 &
"""

import os
import sys
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BOCSER_DIR       = Path("/home/xray/Nicki/BOCSER/bocser")      # source code
WORK_DIR         = Path("/home/xray/Nicki/BOCSER/tests/NonBR_copy") # working folder
ENSEMBLE_FILE    = WORK_DIR / "NonBR-TSs-energies.xyz"          # input ensemble
OUTPUT_FILE      = WORK_DIR / "NonBR_energies_res.xyz"          # output
SBATCH_TEMPLATE  = Path("/home/xray/Nicki/BOCSER/sbatch_temp")  # Slurm template
SP_DIR           = WORK_DIR / "sp_calcs"                        # temp ORCA files

ORCA_CMD         = "/opt/orca_6.1.0/orca"
ORCA_METHOD      = "XTB2"
NUM_PROCS        = 1
CHARGE           = 0
MULTIPLICITY     = 1
TIMEOUT_MINUTES  = 10                   # per structure; XTB2 is fast


HARTREE_TO_KCAL  = 627.509474063
BROKEN_ENERGY    = None                 # None = skip broken structures in output
# ──────────────────────────────────────────────────────────────────────────────


def parse_xyz_blocks(path: Path) -> list[list[str]]:
    """Parse a multi-XYZ file into a list of blocks (each block = list of lines)."""
    blocks = []
    lines = path.read_text().splitlines(keepends=True)
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        try:
            n_atoms = int(stripped)
        except ValueError:
            i += 1
            continue
        block = lines[i : i + 2 + n_atoms]
        if len(block) < 2 + n_atoms:
            log.warning("Incomplete block at line %d — skipping", i)
            break
        blocks.append(block)
        i += 2 + n_atoms
    return blocks


def block_to_coord_string(block: list[str]) -> str:
    """Return coordinate lines (everything after the 2-line header) as a string."""
    return "".join(block[2:])


def write_orca_inp(inp_path: Path, coords: str) -> None:
    """Write a single-point ORCA input file."""
    inp_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"!{ORCA_METHOD} SP\n"
        f"%pal\nnprocs {NUM_PROCS}\nend\n"
        f"* xyz {CHARGE} {MULTIPLICITY}\n"
        f"{coords}"
        f"END\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", dir=str(inp_path.parent), delete=False, suffix=".tmp"
    ) as tmp:
        tmp.write(content)
        tmp_name = tmp.name
    os.replace(tmp_name, inp_path)


def run_orca_via_sbatch(inp_path: Path) -> Path:
    """Copy sbatch template, append ORCA call, submit with -W (wait), return .out path."""
    out_path  = inp_path.with_suffix(".out")
    sh_path   = inp_path.with_suffix(".sh")

    shutil.copy(SBATCH_TEMPLATE, sh_path)
    with open(sh_path, "a") as fh:
        fh.write(f"{ORCA_CMD} {inp_path} > {out_path}\n")

    # -W waits for job completion; -t sets wall time limit
    subprocess.run(
        ["sbatch", "-W", "-t", str(TIMEOUT_MINUTES), str(sh_path)],
        check=False,
    )
    return out_path


def parse_energy_hartree(out_path: Path) -> tuple[float, bool]:
    """Parse FINAL SINGLE POINT ENERGY from ORCA output. Returns (energy_hartree, ok)."""
    import re
    energy_re = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")

    if not out_path.exists():
        log.error("Output file not found: %s", out_path)
        return 0.0, False

    from collections import deque
    with open(out_path, errors="ignore") as fh:
        last_lines = deque(fh, maxlen=500)
    joined = "\n".join(last_lines)

    m = energy_re.search(joined)
    if m:
        return float(m.group(1)), True

    log.warning("No energy found in %s", out_path)
    return 0.0, False


def main() -> None:
    SP_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Work dir    : %s", WORK_DIR)
    log.info("Ensemble    : %s", ENSEMBLE_FILE)
    log.info("Output      : %s", OUTPUT_FILE)
    log.info("ORCA method : %s  procs=%d", ORCA_METHOD, NUM_PROCS)

    if not ENSEMBLE_FILE.exists():
        log.error("Ensemble file not found: %s", ENSEMBLE_FILE)
        sys.exit(1)

    if not SBATCH_TEMPLATE.exists():
        log.error("sbatch template not found: %s", SBATCH_TEMPLATE)
        sys.exit(1)
        
    blocks = parse_xyz_blocks(ENSEMBLE_FILE)
    log.info("Total structures: %d", len(blocks))

    results: list[tuple[int, list[str], float]] = []  # (idx, block, energy_hartree)

    for idx, block in enumerate(blocks):
        inp_path = SP_DIR / f"sp_{idx:04d}.inp"
        coords   = block_to_coord_string(block)

        log.info("[%d/%d] Writing input: %s", idx + 1, len(blocks), inp_path.name)
        write_orca_inp(inp_path, coords)

        log.info("[%d/%d] Submitting ORCA job...", idx + 1, len(blocks))
        out_path = run_orca_via_sbatch(inp_path)

        energy_h, ok = parse_energy_hartree(out_path)
        if ok:
            energy_kcal = energy_h * HARTREE_TO_KCAL
            log.info("[%d/%d] Energy = %.6f Hartree = %.3f kcal/mol",
                     idx + 1, len(blocks), energy_h, energy_kcal)
            results.append((idx, block, energy_kcal))
        else:
            log.warning("[%d/%d] Calculation failed — structure will be skipped in output",
                        idx + 1, len(blocks))

    if not results:
        log.error("All calculations failed. Nothing to write.")
        sys.exit(1)

    # Relative energies: subtract minimum
    energies  = [r[2] for r in results]
    e_min     = min(energies)
    e_max     = max(energies)
    log.info("Done. %d/%d structures succeeded.", len(results), len(blocks))
    log.info("Absolute energy range: %.4f to %.4f kcal/mol", e_min, e_max)
    log.info("Relative energy range: 0.000 to %.4f kcal/mol", e_max - e_min)

    log.info("Writing output: %s", OUTPUT_FILE)
    with open(OUTPUT_FILE, "w") as fout:
        for orig_idx, block, e_abs in results:
            e_rel = e_abs - e_min
            n_atoms = int(block[0].strip())
            # Line 0: atom count
            fout.write(f"{n_atoms}\n")
            # Line 1: relative energy (kcal/mol) — compatible with EnsembleProcessor
            fout.write(f"{e_rel:.6f}\n")
            # Coordinate lines
            fout.writelines(block[2:])
    
    cleanup_slurm_logs()

    log.info("Output written: %s", OUTPUT_FILE)
    log.info("")
    log.info("Summary:")
    log.info("  Structures processed : %d / %d", len(results), len(blocks))
    log.info("  Relative energy min  : 0.000000 kcal/mol")
    log.info("  Relative energy max  : %.6f kcal/mol", e_max - e_min)
    log.info("  Output file          : %s", OUTPUT_FILE)
    log.info("")
    log.info("Use config2.yaml for the next run.")

def cleanup_slurm_logs() -> None:
    """Delete slurm-*.out files that Slurm drops in the BOCSER root folder."""
    bocser_root = Path("/home/xray/Nicki/BOCSER")
    slurm_files = list(bocser_root.glob("slurm-*.out"))
    if not slurm_files:
        log.info("No slurm-*.out files to clean up.")
        return
    for f in slurm_files:
        try:
            f.unlink()
            log.info("Deleted: %s", f.name)
        except OSError as e:
            log.warning("Could not delete %s: %s", f.name, e)
    log.info("Cleaned up %d slurm log file(s).", len(slurm_files))


if __name__ == "__main__":
    print("Started!")
    main()