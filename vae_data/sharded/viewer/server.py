"""Serve the independent viewer and an explicit local trace directory. No write API."""
from pathlib import Path
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import unquote,urlsplit

HERE=Path(__file__).resolve().parent

def serve(dataset,host='127.0.0.1',port=8765):
    data=Path(dataset).resolve();vendor=HERE.parents[2]/'docs/vendor'
    if not(data/'viewer/manifest.json').exists():raise FileNotFoundError('Run export-traces first')
    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self,path):
            path=unquote(urlsplit(path).path)
            root,relative=(data,path[6:])if path.startswith('/data/')else(vendor,path[8:])if path.startswith('/vendor/')else(HERE,path.lstrip('/')or'index.html')
            candidate=(root/relative).resolve()
            return str(candidate)if candidate.is_relative_to(root)else str(HERE/'not-found')
    print(f'GT-only viewer: http://{host}:{port} (Ctrl-C to stop)',flush=True)
    try:ThreadingHTTPServer((host,port),Handler).serve_forever()
    except KeyboardInterrupt:pass
