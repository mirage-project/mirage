"""Schedule viewer: a page (plan_viewer.html) that opens the compiler's schedule files and the solver's result files.

  schedule_gpu<g>.json   (compiler.compile_plan) per-SM task lists of one GPU: nodes, tasks, dependencies. The page shows the
                         nodes and an estimated per-SM timeline: each SM runs its list in order, a task starts when its SM is
                         free and its dependencies have ended, with a duration per node the viewer sets (the file has no times)
  result.json            (save() below, from search.search_plan's summary) the candidates of every node and the solver's
                         predicted per-SM timeline

The page is one standalone HTML file: open more files with its "Open files" button or by dropping them; nothing is uploaded.

  python plan_viewer.py schedule_gpu0.json [more ...]           open the page with these files in the default browser
  python plan_viewer.py [files ...] --html page.html            only write the page (these files already opened) to one file
  python plan_viewer.py [files ...] --serve [--port 8905]       serve it on 127.0.0.1 (on a remote machine: forward the port)
  or open plan_viewer.html itself in a browser and use "Open files".
Standard library only: runs without Mirage.
"""
import argparse
import http.server
import json
import os
import socketserver
import tempfile
import webbrowser
from typing import List

HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan_viewer.html")


def save(path: str, plan: dict, summary: dict, title: str) -> None:
    """A result file for the viewer. plan: {"grids", "lists"} (compiler.compile_plan's format); summary: search.search_plan's."""
    doc = {"title": title, "summary": summary,
           "plan": {"grids": {str(k): list(v) for k, v in plan["grids"].items()},
                    "lists": [[[[node, list(pos)] for node, pos in sm] for sm in gpu] for gpu in plan["lists"]]}}
    with open(path, "w") as f:
        json.dump(doc, f)


def page(paths: List[str]) -> str:
    """plan_viewer.html with the files `paths` already opened."""
    with open(HTML) as f:
        html = f.read()
    preload = []
    for p in paths:
        with open(p) as f:
            preload.append({"name": os.path.basename(p), "data": json.load(f)})
    data = json.dumps(preload, separators=(",", ":")).replace("</", "<\\/")
    return html.replace('<script type="application/json" id="preload">[]</script>',
                        f'<script type="application/json" id="preload">{data}</script>')


def serve(html: str, port: int) -> None:
    body = html.encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as server:
        print(f"schedule viewer: http://127.0.0.1:{port}/  (remote machine: forward port {port}, then open "
              f"http://localhost:{port}/)", flush=True)
        server.serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="schedule_gpu<g>.json or result.json files to open with the page")
    p.add_argument("--html", default=None, help="only write the page to this file")
    p.add_argument("--serve", action="store_true", help="serve the page on 127.0.0.1 instead of opening a browser")
    p.add_argument("--port", type=int, default=8905)
    args = p.parse_args()
    html = page(args.files)
    if args.serve:
        serve(html, args.port)
        return
    path = os.path.abspath(args.html) if args.html else os.path.join(tempfile.mkdtemp(prefix="schedule_viewer_"), "viewer.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"page written to {path}")
    if not args.html and not webbrowser.open("file://" + path):
        print("no browser found here: open that file in a browser, or use --serve on a remote machine")


if __name__ == "__main__":
    main()
