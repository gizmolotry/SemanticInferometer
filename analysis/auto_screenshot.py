import argparse
import re
import subprocess
from pathlib import Path


PLAYWRIGHT = r"C:\Users\Andrew\AppData\Roaming\npm\playwright-cli.cmd"
SESSION = "aestheticloop"
URL = "http://127.0.0.1:8050"


def run_cli(*args: str) -> None:
    cmd = [PLAYWRIGHT, "-s", SESSION, *args]
    subprocess.run(cmd, check=True)


def latest_snapshot() -> Path | None:
    snapshot_dir = Path(r"C:\Users\Andrew\.playwright-cli")
    matches = sorted(snapshot_dir.glob("page-*.yml"), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def find_ref(snapshot_path: Path, label: str) -> str | None:
    pattern = re.compile(rf'option "{re.escape(label)}".*?\[ref=(e\d+)\]')
    for line in snapshot_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            return match.group(1)
    return None


def crop_manifold(full_path: Path, crop_path: Path) -> None:
    try:
        from PIL import Image
    except Exception:
        return

    with Image.open(full_path) as image:
        width, height = image.size
        left = min(max(int(width * 0.19), 250), width - 2)
        top = int(height * 0.02)
        right = width
        bottom = height
        crop = image.crop((left, top, right, bottom))
        crop.save(crop_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--output-dir",
        default=r"C:\Users\Andrew\output\playwright",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / f"dash_full_{args.tag}.png"
    crop_path = output_dir / f"dash_manifold_{args.tag}.png"

    run_cli("open", URL, "--headed")
    run_cli("run-code", "await page.waitForTimeout(12000)")
    run_cli("snapshot")
    snapshot_path = latest_snapshot()
    if snapshot_path is not None:
        global_ref = find_ref(snapshot_path, "Global")
        baseline_ref = find_ref(snapshot_path, "Baseline")
        if global_ref:
            run_cli("click", global_ref)
        if baseline_ref:
            run_cli("click", baseline_ref)
        run_cli("run-code", "await page.waitForTimeout(5000)")
    run_cli("screenshot", "--filename", str(full_path), "--full-page")
    crop_manifold(full_path, crop_path)

    print(full_path)
    print(crop_path)


if __name__ == "__main__":
    main()
