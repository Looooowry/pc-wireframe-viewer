# PC & Wireframe Viewer

A lightweight, browser-based visualization tool for **point clouds** and **wireframe OBJ models**, designed for creating clean comparison figures in research papers.

Built with Python stdlib (no backend dependencies) + Three.js (bundled locally, works offline).

## Features

- **Single overlay mode** — Stack multiple layers (point cloud + wireframes) in one view
- **Grid comparison mode** — 2×2 / 1×4 / 1×3 layouts with synced camera angles
- **Whole area mode** — Load entire folders of point clouds & wireframes at once
  - *Compact grid*: Each building is centered, scaled, and tiled on a grid for compact overview screenshots
  - *Original coords*: Preserve real-world coordinates
- **Layer control** — Toggle visibility, custom colors, per-layer opacity
- **Point rendering** — Soft circular points (not squares), adjustable size, optional RGB from file
- **Real line width** — Uses `LineSegments2` + `LineMaterial` for true pixel-width lines (not capped at 1px)
- **Screenshot export** — 2× resolution PNG; grid mode exports separate files per cell
- **Clean mode** — Hide all UI for paper-ready screenshots
- **White background** — Minimal, clean aesthetic for publication figures
- **Offline** — Three.js bundled locally, no CDN required

## Quick Start

```bash
# 1. Place the viewer folder inside your data root
#    Structure:
#    /data_root/
#      ├── viewer/        <- this repo
#      ├── xyz/            <- point clouds (*.xyz)
#      ├── gt/             <- ground truth wireframes (*.obj)
#      └── pred_*/        <- prediction wireframes (*.obj)

# 2. Start the server
python server.py

# 3. Open in browser
# http://127.0.0.1:8766
```

Or on Windows: double-click `start.bat`.

## Data Format

### Point Cloud (`.xyz`)

Whitespace-delimited, one point per line:
```
x y z [r g b] [extra...]
```
- `x y z` — required, float
- `r g b` — optional, 0–255 integers
- Additional columns ignored

### Wireframe (`.obj`)

Standard OBJ with `v` (vertices) and `l` (line segments) / `f` (faces as polygons):
```
v 1.0 2.0 3.0
v 4.0 5.0 6.0
l 1 2
```

## Configuration

Edit `server.py` to point at your data folders:

```python
FOLDERS = {
    "xyz":   (path_to_xyz_folder,   ".xyz"),
    "gt":    (path_to_gt_folder,    ".obj"),
    "pred":  (path_to_pred_folder,  ".obj"),
    # Add more...
}
```

Filename suffixes are configurable (e.g., `<id>.obj` vs `<id>_pred.obj`).

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| ← / → | Previous / next sample |
| F | Fit view |
| C | Toggle clean mode |
| S | Screenshot |

## Tech Stack

- **Backend**: Python 3 stdlib (`http.server`, `ThreadingHTTPServer`) — zero dependencies
- **Frontend**: Three.js r160 (bundled in `lib/`)
- **Line rendering**: `LineSegments2` / `LineMaterial` for real line widths

## License

MIT
