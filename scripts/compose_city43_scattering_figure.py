"""Combine the City43 building/link render and its paired scattering metrics."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Any

from artifacts.provenance import atomic_json, sha256_file
from PIL import Image, ImageDraw, ImageFont


def compose_figure(scene_path: Path, metrics_path: Path, output_path: Path) -> dict[str, Any]:
    """Place the actual City43 scene above the matching solver-response plots."""
    scene = Image.open(scene_path).convert("RGB")
    metrics = Image.open(metrics_path).convert("RGB")
    width = max(scene.width, metrics.width)
    metrics_height = round(metrics.height * width / metrics.width)
    metrics = metrics.resize((width, metrics_height), Image.Resampling.LANCZOS)
    margin = max(32, width // 80)
    header_height = max(100, width // 30)
    caption_height = max(112, width // 28)
    gap = max(24, width // 100)
    canvas = Image.new(
        "RGB",
        (width, header_height + scene.height + gap + metrics_height + caption_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    title_font = (
        ImageFont.truetype(str(font_path), max(30, width // 65))
        if font_path.is_file()
        else ImageFont.load_default()
    )
    body_font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    body_font = (
        ImageFont.truetype(str(body_font_path), max(20, width // 115))
        if body_font_path.is_file()
        else ImageFont.load_default()
    )
    title = "City43 buildings and diffuse-scattering response"
    draw.text((margin, margin // 2), title, fill="#18212b", font=title_font)
    y = header_height
    canvas.paste(scene, ((width - scene.width) // 2, y))
    y += scene.height + gap
    canvas.paste(metrics, (0, y))
    y += metrics_height + 14
    caption = (
        "Top: all 16 building meshes and the audited LOS/NLOS links in the City43 scene. "
        "Bottom: diffuse-path count, power, delay spread, and angular spread versus ray budget. "
        "The concrete coefficient S=0.5 is a synthetic solver-mechanics control, not a calibrated material value."
    )
    draw.multiline_text(
        (margin, y),
        textwrap.fill(caption, width=130),
        fill="#303943",
        font=body_font,
        spacing=8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    sidecar = {
        "schema": "loslab.city43_scattering_figure.v1",
        "inputs": [
            {"path": str(scene_path), "sha256": sha256_file(scene_path)},
            {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
        ],
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "description": caption,
    }
    atomic_json(output_path.with_suffix(".json"), sidecar)
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="3D building/link scene PNG")
    parser.add_argument("--metrics", type=Path, required=True, help="paired scattering-budget PNG")
    parser.add_argument("--output", type=Path, required=True, help="combined figure PNG path")
    args = parser.parse_args()
    result = compose_figure(args.scene, args.metrics, args.output)
    print(f"Saved combined City43 figure to {result['output']['path']}")


if __name__ == "__main__":
    main()
