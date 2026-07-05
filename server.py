#!/usr/bin/env python3
"""Point cloud & wireframe OBJ visualization server (stdlib only)."""

import json
import os
import re
import struct
import sys
import traceback
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote, parse_qs

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(ROOT)  # D:\MAI

# Each entry: (path, filename_suffix). file = <id><suffix>
# e.g. suffix ".obj"  -> 1001.obj
#      suffix "_pred.obj" -> 1001_pred.obj
FOLDERS = {
    "xyz":          (os.path.join(DATA_ROOT, "xyz"),                                  ".xyz"),
    "gt":           (os.path.join(DATA_ROOT, "gt"),                                   ".obj"),
    "pred_2048":    (os.path.join(DATA_ROOT, "pred_wireframe_2048_bwformer"),         ".obj"),
    "pred_roof":    (os.path.join(DATA_ROOT, "pred_wireframe_roof_bwformerv3"),       ".obj"),
    "test_results": (os.path.join(DATA_ROOT, "test_results"),                         "_pred.obj"),
    "patch32":      (os.path.join(DATA_ROOT, "patch32sigma0.01clip0.01"),             "_pred.obj"),
    "patch32_test": (os.path.join(DATA_ROOT, "patch32sigma0.01clip0.01_test"),        "_pred.obj"),
}

# Folders whose file presence is required for a sample to be listed.
# Optional folders only contribute data when a file happens to exist.
CORE_FOLDERS = ["xyz", "gt", "pred_2048", "pred_roof"]

# Back-compat: EXT and FOLDERS-path maps used by older code paths
EXT = {k: v[1] for k, v in FOLDERS.items()}

# Pre-compute sample list (thread-safe read)
SAMPLE_LIST = None
SAMPLE_LIST_LOCK = threading.Lock()

# Whole-area cache: { folder_key: bytes }
WHOLE_CACHE = {}
WHOLE_CACHE_LOCK = threading.Lock()
WHOLE_LOADING = {}  # folder_key -> bool


def get_id_list():
    global SAMPLE_LIST
    with SAMPLE_LIST_LOCK:
        if SAMPLE_LIST is not None:
            return SAMPLE_LIST
    sets = []
    for key in CORE_FOLDERS:
        folder, suffix = FOLDERS[key]
        ids = set()
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.endswith(suffix):
                    base = f[:-len(suffix)]
                    ids.add(base)
        sets.append(ids)
    common = set.intersection(*sets) if sets else set()
    result = sorted(common, key=lambda x: (len(x), x))
    with SAMPLE_LIST_LOCK:
        SAMPLE_LIST = result
    return result


def _parse_xyz_file(fpath):
    """Parse one .xyz file -> (positions list, colors list, has_rgb)."""
    positions = []
    colors = []
    has_rgb = False
    with open(fpath, "r", errors="ignore") as f:
        first = True
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                x = float(parts[0]); y = float(parts[1]); z = float(parts[2])
            except ValueError:
                continue
            positions.append(x); positions.append(y); positions.append(z)
            if len(parts) >= 6:
                if first:
                    has_rgb = True
                    first = False
                try:
                    r = float(parts[3]) / 255.0
                    g = float(parts[4]) / 255.0
                    b = float(parts[5]) / 255.0
                except ValueError:
                    r = g = b = 0.5
                colors.append(r); colors.append(g); colors.append(b)
            else:
                colors.append(0.5); colors.append(0.5); colors.append(0.5)
    return positions, colors, has_rgb


def _parse_obj_file(fpath):
    """Parse one .obj file -> (vertices list, indices list)."""
    vertices = []
    indices = []
    with open(fpath, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()[1:4]
                try:
                    vertices.append(float(parts[0]))
                    vertices.append(float(parts[1]))
                    vertices.append(float(parts[2]))
                except (ValueError, IndexError):
                    pass
            elif line.startswith("l "):
                pts = line.split()[1:]
                nums = []
                for p in pts:
                    try:
                        nums.append(int(p.split("/")[0]) - 1)
                    except ValueError:
                        pass
                for i in range(len(nums) - 1):
                    indices.append(nums[i])
                    indices.append(nums[i + 1])
            elif line.startswith("f "):
                pts = line.split()[1:]
                nums = []
                for p in pts:
                    try:
                        nums.append(int(p.split("/")[0]) - 1)
                    except ValueError:
                        pass
                for i in range(len(nums)):
                    indices.append(nums[i])
                    indices.append(nums[(i + 1) % len(nums)])
    return vertices, indices


def build_whole_data(folder_key, max_files=None, compact=False, spacing=1.2):
    """Merge all files in a folder into a single binary blob.

    Two modes:
      - original (compact=False): merge in original world coords.
      - compact (compact=True): center & scale each building, then place them
        on a regular grid so the whole dataset fits compactly.

    Point cloud (xyz):
      header: 4 bytes magic 'PC00', 4 bytes uint32 point_count, 1 byte has_rgb,
              3 bytes padding (total 12 bytes, 4-byte aligned for Float32Array)
      body:   Float32Array positions (3*N) [+ Float32Array colors (3*N) if has_rgb]

    Wireframe (obj):
      header: 4 bytes magic 'WF00', 4 bytes uint32 vertex_count, 4 bytes uint32 index_count (12 bytes)
      body:   Float32Array vertices (3*V) + Uint32Array indices (I)
    """
    folder, suffix = FOLDERS[folder_key]
    files = sorted([f for f in os.listdir(folder) if f.endswith(suffix)],
                   key=lambda x: (len(x), x))
    if max_files:
        files = files[:max_files]

    n_files = len(files)
    cols = int(n_files ** 0.5) or 1
    rows = (n_files + cols - 1) // cols

    if folder_key == "xyz":
        all_pos = []
        all_col = []
        has_rgb_any = False
        for fi, f in enumerate(files):
            fp = os.path.join(folder, f)
            p, c, hr = _parse_xyz_file(fp)
            if hr:
                has_rgb_any = True
            if compact and len(p) >= 3:
                # center and scale per-building
                minx = miny = minz = float('inf')
                maxx = maxy = maxz = float('-inf')
                for i in range(0, len(p), 3):
                    x, y, z = p[i], p[i+1], p[i+2]
                    if x < minx: minx = x
                    if x > maxx: maxx = x
                    if y < miny: miny = y
                    if y > maxy: maxy = y
                    if z < minz: minz = z
                    if z > maxz: maxz = z
                cx = (minx + maxx) / 2.0
                cy = (miny + maxy) / 2.0
                cz = (minz + maxz) / 2.0
                sx = maxx - minx if maxx > minx else 1.0
                sy = maxy - miny if maxy > miny else 1.0
                sz = maxz - minz if maxz > minz else 1.0
                scale = max(sx, sy, sz)
                # grid position on x-y plane, z starts at 0
                gcol = fi % cols
                grow = fi // cols
                gx = (gcol - (cols - 1) / 2.0) * spacing
                gy = (grow - (rows - 1) / 2.0) * spacing
                gz = 0.0
                for i in range(0, len(p), 3):
                    p[i]   = (p[i]   - cx) / scale + gx
                    p[i+1] = (p[i+1] - cy) / scale + gy
                    p[i+2] = (p[i+2] - cz) / scale + gz
            all_pos.extend(p)
            all_col.extend(c)
        n = len(all_pos) // 3
        header = b"PC00" + struct.pack("<I", n) + (b"\x01" if has_rgb_any else b"\x00") + b"\x00\x00\x00"
        import array
        pos_arr = array.array("f", all_pos)
        col_arr = array.array("f", all_col)
        body = pos_arr.tobytes()
        if has_rgb_any:
            body += col_arr.tobytes()
        return header + body
    else:
        all_verts = []
        all_idx = []
        offset = 0
        for fi, f in enumerate(files):
            fp = os.path.join(folder, f)
            v, idx = _parse_obj_file(fp)
            if compact and len(v) >= 3:
                minx = miny = minz = float('inf')
                maxx = maxy = maxz = float('-inf')
                for i in range(0, len(v), 3):
                    x, y, z = v[i], v[i+1], v[i+2]
                    if x < minx: minx = x
                    if x > maxx: maxx = x
                    if y < miny: miny = y
                    if y > maxy: maxy = y
                    if z < minz: minz = z
                    if z > maxz: maxz = z
                cx = (minx + maxx) / 2.0
                cy = (miny + maxy) / 2.0
                cz = (minz + maxz) / 2.0
                sx = maxx - minx if maxx > minx else 1.0
                sy = maxy - miny if maxy > miny else 1.0
                sz = maxz - minz if maxz > minz else 1.0
                scale = max(sx, sy, sz)
                gcol = fi % cols
                grow = fi // cols
                gx = (gcol - (cols - 1) / 2.0) * spacing
                gy = (grow - (rows - 1) / 2.0) * spacing
                gz = 0.0
                for i in range(0, len(v), 3):
                    v[i]   = (v[i]   - cx) / scale + gx
                    v[i+1] = (v[i+1] - cy) / scale + gy
                    v[i+2] = (v[i+2] - cz) / scale + gz
            all_verts.extend(v)
            for i in idx:
                all_idx.append(i + offset)
            offset += len(v) // 3
        nv = len(all_verts) // 3
        ni = len(all_idx)
        header = b"WF00" + struct.pack("<II", nv, ni)
        import array
        v_arr = array.array("f", all_verts)
        i_arr = array.array("I", all_idx)
        return header + v_arr.tobytes() + i_arr.tobytes()


def get_whole_data(folder_key, max_files=None, compact=False):
    """Return cached whole-area binary data, building if needed."""
    cache_key = f"{folder_key}:{max_files}:{int(compact)}"
    with WHOLE_CACHE_LOCK:
        if cache_key in WHOLE_CACHE:
            return WHOLE_CACHE[cache_key]
        if WHOLE_LOADING.get(cache_key):
            return None  # still building

    # Build it (no lock held during heavy work)
    with WHOLE_CACHE_LOCK:
        WHOLE_LOADING[cache_key] = True

    try:
        print(f"Building whole-area data for {folder_key} (max_files={max_files}, compact={compact})...", flush=True)
        data = build_whole_data(folder_key, max_files, compact)
        with WHOLE_CACHE_LOCK:
            WHOLE_CACHE[cache_key] = data
            WHOLE_LOADING.pop(cache_key, None)
        print(f"  Done: {folder_key} = {len(data)} bytes", flush=True)
        return data
    except Exception:
        with WHOLE_CACHE_LOCK:
            WHOLE_LOADING.pop(cache_key, None)
        raise


class Handler(BaseHTTPRequestHandler):
    # Use HTTP/1.0 to avoid keep-alive complications with single-thread issues
    # (ThreadingHTTPServer handles concurrency anyway, but keep it simple)
    protocol_version = "HTTP/1.1"

    def _send_data(self, data, content_type="application/octet-stream"):
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def _send_404(self):
        msg = b"Not found"
        try:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(msg)
        except:
            pass

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)

            if path == "/" or path == "/index.html":
                fpath = os.path.join(ROOT, "index.html")
                with open(fpath, "rb") as f:
                    data = f.read()
                self._send_data(data, "text/html; charset=utf-8")

            elif path == "/api/list":
                data = json.dumps(get_id_list()).encode()
                self._send_data(data, "application/json")

            elif path.startswith("/api/whole/"):
                # /api/whole/<folder>?max=N&compact=1
                parts = path.split("/")
                if len(parts) >= 4:
                    folder_key = parts[3]
                    if folder_key in FOLDERS:
                        qs = parse_qs(parsed.query)
                        max_files = None
                        if "max" in qs:
                            try:
                                max_files = int(qs["max"][0])
                            except ValueError:
                                pass
                        compact = qs.get("compact", [""])[0] in ("1", "true", "yes")
                        data = get_whole_data(folder_key, max_files, compact)
                        if data is not None:
                            self._send_data(data, "application/octet-stream")
                            return
                        else:
                            # Still building
                            self.send_response(202)
                            self.send_header("Content-Length", "0")
                            self.send_header("Connection", "close")
                            self.end_headers()
                            return
                self._send_404()

            elif path.startswith("/api/file/"):
                parts = path.split("/")
                # /api/file/<folder>/<id>
                if len(parts) >= 5:
                    folder_key = parts[3]
                    sample_id = parts[4]
                    if folder_key in FOLDERS and re.match(r'^[\w\-]+$', sample_id):
                        folder, suffix = FOLDERS[folder_key]
                        fpath = os.path.join(folder, sample_id + suffix)
                        if os.path.isfile(fpath):
                            with open(fpath, "rb") as f:
                                data = f.read()
                            self._send_data(data, "text/plain; charset=utf-8")
                            return
                self._send_404()

            elif path.startswith("/lib/") or path.startswith("/static/"):
                # Serve static files from viewer directory
                safe_path = path.lstrip("/")
                fpath = os.path.join(ROOT, safe_path)
                # Prevent path traversal
                fpath = os.path.normpath(fpath)
                if os.path.isfile(fpath) and fpath.startswith(ROOT):
                    with open(fpath, "rb") as f:
                        data = f.read()
                    import mimetypes
                    mime, _ = mimetypes.guess_type(fpath)
                    mime = mime or "application/octet-stream"
                    self._send_data(data, mime)
                else:
                    self._send_404()

        except Exception as e:
            traceback.print_exc()
            sys.stderr.flush()

    def log_message(self, format, *args):
        pass


def main():
    port = 8766
    ids = get_id_list()
    print(f"Found {len(ids)} samples common to all folders.", flush=True)
    if ids:
        print(f"  First 5: {ids[:5]}", flush=True)
        print(f"  Last 5:  {ids[-5:]}", flush=True)
    os.chdir(ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    print(f"\nServer running at http://127.0.0.1:{port}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
