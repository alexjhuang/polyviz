"""Live polyhedral loop-nest visualizer.  python app.py  ->  http://localhost:8765

POST /run {code, schedule} -> {names, points, sched, faces, sched_faces, error}
The Python loop nest in `code` is instrumented via ast: the innermost For (descending
through If guards) gets its body replaced by a point-recording call, then exec'd.
"""
import ast, json, os, sys, textwrap
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import numpy as np
from scipy.spatial import ConvexHull

MAX_POINTS = 20000
HERE = os.path.dirname(os.path.abspath(__file__))


def instrument(code):
    tree = ast.parse(textwrap.dedent(code.expandtabs(4)))  # tolerate pasted indentation / tabs
    # outermost For in the module body; descend For/If chain to the innermost body
    outer = next((s for s in tree.body if isinstance(s, ast.For)), None)
    if outer is None:
        raise ValueError("no `for` loop found")
    names, node = [], outer
    while True:
        if isinstance(node, ast.For):
            if not isinstance(node.target, ast.Name):
                raise ValueError("loop target must be a plain name")
            names.append(node.target.id)
        inner = [s for s in node.body if isinstance(s, (ast.For, ast.If))]
        if not inner:
            break
        node = inner[0]
    node.body = [ast.parse(f"_rec(({', '.join(names)},))").body[0]]
    if isinstance(node, ast.If):
        node.orelse = []
    ast.fix_missing_locations(tree)
    return names, compile(tree, "<nest>", "exec")


def enumerate_points(code):
    names, co = instrument(code)
    pts = []

    def _rec(p):
        pts.append(tuple(int(x) for x in p))
        if len(pts) > MAX_POINTS:
            raise RuntimeError(f"more than {MAX_POINTS} iterations; shrink the bounds")

    exec(co, {"_rec": _rec, "range": range, "min": min, "max": max, "abs": abs})
    return names, pts


def pad3(pts):
    a = np.zeros((len(pts), 3))
    if len(pts):
        a[:, : min(3, len(pts[0]))] = np.array(pts)[:, :3]
    return a


def hull_faces(a):
    """Triangles (index triples) of the convex hull, handling flat/collinear sets."""
    if len(a) < 3:
        return []
    c = a - a.mean(0)
    rank = np.linalg.matrix_rank(c, tol=1e-9)
    if rank >= 3:
        return ConvexHull(a).simplices.tolist()
    if rank == 2:
        _, _, vt = np.linalg.svd(c)
        h = ConvexHull(c @ vt[:2].T).vertices.tolist()  # 2D hull, ordered
        return [[h[0], h[k], h[k + 1]] for k in range(1, len(h) - 1)]
    return []


def run(code, schedule):
    names, pts = enumerate_points(code)
    a = pad3(pts)
    out = {"names": names, "points": a.tolist(), "faces": hull_faces(a)}
    if schedule.strip():
        f = eval(f"lambda {', '.join(names)}: ({schedule},)", {})
        s = pad3([f(*p) for p in pts])
        out["sched"], out["sched_faces"] = s.tolist(), hull_faces(s)
    return out


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            out = run(body.get("code", ""), body.get("schedule", ""))
        except Exception as e:  # surfaced in the UI
            out = {"error": f"{type(e).__name__}: {e}"}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    if "--check" in sys.argv:
        r = run("N = 4\nfor i in range(N):\n    for j in range(i + 1):\n        pass\n", "i, i + j")
        assert r["names"] == ["i", "j"] and len(r["points"]) == 10, r
        assert r["faces"] and r["sched_faces"], r          # flat 2D domain still gets a hull
        assert r["sched"][-1][:2] == [3.0, 6.0], r["sched"][-1]
        r = run("for i in range(3):\n    for j in range(3):\n        if i != j:\n            pass\n", "")
        assert len(r["points"]) == 6, r
        r = run("    N = 3\n    for i in range(N):\n\t\tfor j in range(N):\n\t\t\tpass\n", "")
        assert len(r["points"]) == 9, r
        print("ok")
        sys.exit()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"http://localhost:{port}")
    ThreadingHTTPServer(("", port), H).serve_forever()
