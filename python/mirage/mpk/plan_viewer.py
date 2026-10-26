"""Plan viewer: one standalone page (plan_viewer.html) that shows how a static megakernel's graph is split into tasks, where each
task runs and when, from two kinds of files only:

  schedule_gpu<g>.json   (compiler.compile_plan) one GPU's plan: per node its grid, params, tensors (dims, which dimension each grid
                         axis cuts: the page derives every task's slice with compiler.box's rule), producers and cost key; every task
                         (node, grid position, the tasks it waits for); per SM the task ids in run order
  cost files             (cost_records.py, e.g. costs/b300_sxm6pc_cuda13.0.json) the measured task times: records per task type, cost
                         key and context (first on its SM or not)

The page times every task as the search's model does (search.Costs: a task behind another on its SM takes the median of its
records not first on the SM, the first task on an SM its first-on-SM records; a one-task-per-SM node is a pool whose measured work
the SMs share; the records' p75 by default, as the loop's model, median / p90 selectable) and lays the plan out by the solver's rules (solver.py: a task starts when its SM is free and the tasks it waits for
have ended; a concurrent group starts together; the pool opens when its producers' tasks have ended, each SM joins when free, it
ends when the SMs have done its work). A task with no record takes the time typed in for its node (drawn dashed).

  python plan_viewer.py DIR/schedules [costs.json ...]       open the page with these files (a directory: its schedule_gpu*.json)
  python plan_viewer.py A/schedules B/schedules costs.json   two plans: a compare view (one time axis)
  python plan_viewer.py ... --html page.html                 only write the page (the files already opened) to one file
  python plan_viewer.py ... --serve [--port 8905]            serve it on 127.0.0.1 (on a remote machine: forward the port)
or open plan_viewer.html in a browser and use "Open files" (schedule files opened together are one plan). Standard library only.
"""
import argparse
import glob
import http.server
import json
import os
import socketserver
import tempfile
import webbrowser
from typing import List

HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan_viewer.html")


def preload_items(paths: List[str]) -> List[dict]:
    """The files to open: a directory = its schedule_gpu*.json (one plan, named after the directory: its parent's name when it is
    called schedules); a file = a schedule file (one plan per directory) or a cost file."""
    items = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(p))
        files = sorted(glob.glob(os.path.join(p, "schedule_gpu*.json")), key=lambda f: int(f.rsplit("gpu", 1)[1].split(".")[0])) \
            if os.path.isdir(p) else [p]
        if not files:
            raise SystemExit(f"{p}: no schedule_gpu*.json")
        for f in files:
            d = os.path.dirname(f)
            plan = os.path.basename(os.path.dirname(d)) if os.path.basename(d) == "schedules" else os.path.basename(d)
            with open(f) as fh:
                items.append({"name": os.path.basename(f), "group": plan, "data": json.load(fh)})
    return items


def page(paths: List[str]) -> str:
    """plan_viewer.html with the files `paths` already opened."""
    with open(HTML) as f:
        html = f.read()
    data = json.dumps(preload_items(paths), separators=(",", ":")).replace("</", "<\\/")
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
        print(f"plan viewer: http://127.0.0.1:{port}/  (remote machine: forward port {port}, then open "
              f"http://localhost:{port}/)", flush=True)
        server.serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="schedule directories / schedule files / cost files to open with the page")
    p.add_argument("--html", default=None, help="only write the page to this file")
    p.add_argument("--serve", action="store_true", help="serve the page on 127.0.0.1 instead of opening a browser")
    p.add_argument("--port", type=int, default=8905)
    args = p.parse_args()
    html = page(args.files)
    if args.serve:
        serve(html, args.port)
        return
    path = os.path.abspath(args.html) if args.html else os.path.join(tempfile.mkdtemp(prefix="plan_viewer_"), "viewer.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"page written to {path}")
    if not args.html and not webbrowser.open("file://" + path):
        print("no browser found here: open that file in a browser, or use --serve on a remote machine")


if __name__ == "__main__":
    main()
