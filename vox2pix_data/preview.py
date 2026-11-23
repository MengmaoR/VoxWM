"""Read committed Vox2Pix scenes for the local Viewer."""

import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import zarr

from .protocol import VERSION, plan
from .storage import WHITE_FIELDS, committed_paths, read_json, read_vocabulary


VIEWER = Path(__file__).resolve().with_name("viewer")
VENDOR = VIEWER.parents[1] / "docs" / "vendor"
JSON = "application/json; charset=utf-8"
BINARY = "application/octet-stream"
STATIC = {
    "/": (VIEWER / "index.html", "text/html; charset=utf-8"),
    "/style.css": (VIEWER / "style.css", "text/css; charset=utf-8"),
    "/viewer.js": (VIEWER / "viewer.js", "text/javascript; charset=utf-8"),
    "/scene.js": (VIEWER / "scene.js", "text/javascript; charset=utf-8"),
    "/vendor/three.module.min.js": (VENDOR / "three.module.min.js", "text/javascript; charset=utf-8"),
    "/vendor/OrbitControls.js": (VENDOR / "OrbitControls.js", "text/javascript; charset=utf-8"),
}
SCENE_ROUTE = re.compile(r"/api/scenes/(\d+)/(?:(meta|voxels)|views/(\d+)/(rgb|white))")


def make_handler(dataset):
    dataset = Path(dataset)
    manifest = read_json(dataset / "manifest.json")
    if manifest["version"] != VERSION or manifest["axes"] != "ENU":
        raise ValueError("Unsupported Vox2Pix dataset format")
    config = manifest["config"]
    paths = committed_paths(dataset)
    if not paths:
        raise ValueError(f"No committed Vox2Pix scenes in {dataset}")
    vocabulary = read_vocabulary(dataset)
    scenes = [slot for slot in plan(config["seed"]) if slot["scene_id"] in paths]
    catalog = json.dumps(dict(config=config, vocabulary=vocabulary, scenes=scenes)).encode()

    class Handler(BaseHTTPRequestHandler):
        def send_data(self, payload, content_type, compress=False):
            compressed = compress and "gzip" in self.headers.get("Accept-Encoding", "")
            if compressed:
                payload = gzip.compress(payload, compresslevel=1)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            if compress:
                self.send_header("Vary", "Accept-Encoding")
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            route = urlsplit(self.path).path
            try:
                if route == "/api/catalog":
                    return self.send_data(catalog, JSON, compress=True)
                if route in STATIC:
                    path, content_type = STATIC[route]
                    return self.send_data(path.read_bytes(), content_type)
                match = SCENE_ROUTE.fullmatch(route)
                if match is None:
                    return self.send_error(404)
                scene_id, field, view, view_field = match.groups()
                scene = paths.get(int(scene_id))
                if scene is None:
                    return self.send_error(404, "Scene is not committed")
                if field == "meta":
                    return self.send_data((scene / "scene.json").read_bytes(), JSON)
                if view is not None:
                    view = int(view)
                    if not 0 <= view < config["views_per_scene"]:
                        return self.send_error(404, "View index is outside the scene")
                if view_field == "rgb":
                    return self.send_data((scene / "rgb" / f"{view:03d}.png").read_bytes(), "image/png")
                group = zarr.open_group(str(scene / "voxels.zarr"), mode="r")
                if field == "voxels":
                    payload = group["scene_node_id"][:].astype("<u2", copy=False).tobytes()
                else:
                    payload = b"".join(group[key][view].astype(dtype, copy=False).tobytes()
                                       for key, dtype in WHITE_FIELDS.items())
                return self.send_data(payload, BINARY, compress=True)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A view switch can cancel a pending request.
            except (KeyError, OSError, ValueError) as exc:
                self.log_error("%s", exc)
                self.send_error(500, "Could not read this Viewer asset")

        def log_request(self, code="-", size="-"):
            if isinstance(code, int) and code >= 400:
                super().log_request(code, size)

    return Handler


def serve(dataset, port):
    handler = make_handler(dataset)
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"Vox2Pix Viewer: http://127.0.0.1:{server.server_port}/ (Ctrl-C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
