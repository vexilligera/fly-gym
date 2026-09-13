"""Serve the self-contained browser apps on loopback only (no dependencies)."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class LocalHandler(SimpleHTTPRequestHandler):
    def send_json(self, data, status=200):
        body = json.dumps(data, allow_nan=False, separators=(',', ':')).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/api/brain/status':
            return self.send_json(self.server.brain.status())
        if self.path == '/api/vision/status':
            return self.send_json(self.server.brain.navigation_status('vision'))
        if self.path == '/api/maze/status':
            return self.send_json(self.server.brain.navigation_status('maze'))
        if self.path == '/api/maze/world':
            try:
                return self.send_json(self.server.brain.maze_geometry())
            except ValueError as error:
                return self.send_json({'error':str(error)},400)
        if self.path == '/api/brain/geometry' or self.path.startswith('/api/brain/neuron/'):
            try:
                if self.path == '/api/brain/geometry':
                    return self.send_json(self.server.brain.geometry())
                index = int(self.path.rsplit('/', 1)[-1])
                return self.send_json(self.server.brain.neuron_info(index))
            except ValueError as error:
                return self.send_json({'error': str(error)}, 400)
            except Exception as error:
                return self.send_json({'error': str(error)}, 500)
        return super().do_GET()

    def do_POST(self):
        if self.path not in ('/api/brain/step', '/api/brain/reset', '/api/vision/start', '/api/vision/pause', '/api/vision/reset', '/api/maze/start', '/api/maze/pause', '/api/maze/reset'):
            return self.send_error(404)
        origin = self.headers.get('Origin')
        port = self.server.server_address[1]
        if origin and origin not in {f'http://127.0.0.1:{port}', f'http://localhost:{port}', *self.server.allowed_origins}:
            return self.send_error(403)
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 4096 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                raise ValueError('Expected a small JSON object')
            arguments = json.loads(self.rfile.read(length))
            visual = self.path.startswith('/api/vision/')
            maze = self.path.startswith('/api/maze/')
            allowed = {'condition','heading_deg','seed','duration','food_odor'} if maze else {'condition', 'heading_deg', 'target_deg', 'seed', 'duration'} if visual else {'stimulus', 'rate_hz', 'odor', 'silence'}
            if not isinstance(arguments, dict) or set(arguments) - allowed:
                raise ValueError('Unknown parameters')
            action = self.path.rsplit('/',1)[-1]
            if maze or visual:
                result = self.server.brain.navigation_command('maze' if maze else 'vision',action,arguments)
            else:
                result = self.server.brain.execute(action,arguments)
            return self.send_json(result)
        except (ValueError, TypeError) as error:
            return self.send_json({'error': str(error)}, 400)
        except Exception as error:
            return self.send_json({'error': str(error)}, 500)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format, *args):
        # A model fetch loads dozens of meshes. Keep terminal output useful.
        if len(args) < 2 or str(args[1]) not in ("200", "304"):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", choices=('brian2', 'cuda'), default='brian2')
    parser.add_argument("--allowed-origin", action='append', default=[], help='Exact browser origin of a private reverse proxy')
    args = parser.parse_args()
    if sys.platform.startswith('linux') and args.backend == 'cuda':
        os.environ.setdefault('MUJOCO_GL', 'egl')
        os.environ.setdefault('NUMBA_NUM_THREADS', '4')
        # Some compute images have NVIDIA EGL but omit the GLVND loader.
        # Use the private, documented package extraction when present. An
        # exec is needed because the dynamic linker reads its path at startup.
        egl = Path(__file__).resolve().parents[1] / 'deploy/egl/usr/lib/x86_64-linux-gnu'
        library_path = os.environ.get('LD_LIBRARY_PATH', '')
        if egl.is_dir() and str(egl) not in library_path.split(':'):
            os.environ['LD_LIBRARY_PATH'] = str(egl) + (':' + library_path if library_path else '')
            os.execv(sys.executable, [sys.executable, *sys.argv])
    root = Path(__file__).resolve().parents[1] / "wasm"
    required = ["shared/vendor/mujoco/mujoco.wasm", "game/assets/model/fly.xml", "viewer/assets/model/fly.xml"]
    if any(not (root / path).is_file() for path in required):
        parser.error("Missing assets. Run .venv/bin/python scripts/dev/properdocs_hooks.py first.")
    handler = partial(LocalHandler, directory=str(root))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        from brain.service import BrainService
        server.allowed_origins = set(args.allowed_origin)
        server.brain = BrainService(backend=args.backend)
        print(f"NeuroMechFly local lab: http://127.0.0.1:{args.port}/", flush=True)
        print("Bound to this computer only. Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
