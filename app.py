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
MAX_EDGES = 20000
HERE = os.path.dirname(os.path.abspath(globals().get("__file__", "app.py")))  # no __file__ under Pyodide


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
    reads, writes = accesses(node.body)
    node.body = [ast.parse(f"_rec(({', '.join(names)},))").body[0]]  # body is analyzed, never executed
    if isinstance(node, ast.If):
        node.orelse = []
    ast.fix_missing_locations(tree)
    return names, compile(tree, "<nest>", "exec"), reads, writes


def _flat(n):
    """A[i-1][j] / A[i, j]  ->  ('A', [index exprs]) ; None if the base isn't a plain name."""
    idx = []
    while isinstance(n, ast.Subscript):
        idx = (list(n.slice.elts) if isinstance(n.slice, ast.Tuple) else [n.slice]) + idx
        n = n.value
    return (n.id, idx) if isinstance(n, ast.Name) else None


def accesses(body):
    """Array reads/writes in the innermost body as (name, [index exprs]) — outermost subscripts only."""
    reads, writes = [], []

    class V(ast.NodeVisitor):
        def visit_Subscript(self, n):
            a = _flat(n)
            if a is None:
                return self.generic_visit(n)
            (writes if isinstance(n.ctx, ast.Store) else reads).append(a)
            for e in a[1]:
                self.visit(e)  # indirect indices: A[B[i]] reads B

        def visit_AugAssign(self, n):  # A[i] += x reads and writes A[i]
            self.generic_visit(n)
            a = _flat(n.target)
            if a is not None:
                reads.append(a)

    for st in body:
        V().visit(st)
    return reads, writes


def enumerate_points(code):
    """Integer points of the domain plus dynamically traced dependence edges (src, dst, kind)."""
    names, co, reads, writes = instrument(code)
    ns = {"_rec": None, "range": range, "min": min, "max": max, "abs": abs}
    # index lambdas share the exec namespace, so they see user parameters like N (late-bound)
    mk = lambda acc: [(nm, eval(f"lambda {', '.join(names)}: ({', '.join(map(ast.unparse, idx))},)", ns)) for nm, idx in acc]
    rd, wr = mk(reads), mk(writes)
    pts, edges, last_write, readers = [], set(), {}, {}

    def _rec(p):
        pts.append(tuple(int(x) for x in p))
        if len(pts) > MAX_POINTS:
            raise RuntimeError(f"more than {MAX_POINTS} iterations; shrink the bounds")
        cur = len(pts) - 1
        for nm, f in rd:  # RHS first
            cell = (nm, f(*p))
            if cell in last_write and last_write[cell] != cur:
                edges.add((last_write[cell], cur, "RAW"))
            readers.setdefault(cell, []).append(cur)
        for nm, f in wr:
            cell = (nm, f(*p))
            if cell in last_write and last_write[cell] != cur:
                edges.add((last_write[cell], cur, "WAW"))
            edges.update((r, cur, "WAR") for r in readers.pop(cell, ()) if r != cur)
            last_write[cell] = cur
        if len(edges) > MAX_EDGES:
            raise RuntimeError(f"more than {MAX_EDGES} dependence edges; shrink the bounds")

    ns["_rec"] = _rec
    exec(co, ns)
    return names, pts, sorted(edges)


def transitive(edges, n):
    """Mark edges implied by a longer path (transitive reduction over all kinds together).

    Edges always go forward in iteration order, so a reverse sweep with int bitsets is enough.
    """
    succ = {}
    for a, b, _ in edges:
        succ.setdefault(a, set()).add(b)
    reach = [0] * n  # reach[v]: bitset of nodes reachable from v (excluding v)
    for v in range(n - 1, -1, -1):
        for w in succ.get(v, ()):
            reach[v] |= reach[w] | (1 << w)
    red = {}
    for v, ws in succ.items():
        via = 0  # nodes reachable through at least one hop *then* more
        for w in ws:
            via |= reach[w]
        red[v] = via
    return [bool(red.get(a, 0) >> b & 1) for a, b, _ in edges]


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
    names, pts, edges = enumerate_points(code)
    a = pad3(pts)
    out = {"names": names, "points": a.tolist(), "faces": hull_faces(a),
           "edges": [[s, d, k, t] for (s, d, k), t in zip(edges, transitive(edges, len(pts)))]}
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
        assert len(r["points"]) == 6 and r["edges"] == [], r
        # 2-D stencil: 12 flow deps, none transitive
        r = run("N = 4\nfor i in range(1, N):\n    for j in range(1, N):\n        A[i][j] = A[i-1][j] + A[i][j-1]\n", "")
        kinds = [e[2] for e in r["edges"]]
        assert kinds.count("RAW") == 12 and len(kinds) == 12 and not any(e[3] for e in r["edges"]), r["edges"]
        # fib: the i-2 edge is implied by two i-1 hops
        r = run("for i in range(2, 5):\n    A[i] = A[i-1] + A[i-2]\n", "")
        assert [e[3] for e in r["edges"]] == [False, True, False], r["edges"]
        r = run("for i in range(3):\n    for j in range(3):\n        S[0] += B[i, j]\n", "")  # reduction: RAW+WAW chain
        assert [e[2] for e in r["edges"]].count("RAW") == 8 and len(r["edges"]) == 16, r["edges"]
        r = run("for i in range(4):\n    B[i] = A[i + 1]\n    A[i] = 0\n", "")  # anti deps only
        assert [e[2] for e in r["edges"]] == ["WAR"] * 3, r["edges"]
        r = run("    N = 3\n    for i in range(N):\n\t\tfor j in range(N):\n\t\t\tpass\n", "")
        assert len(r["points"]) == 9, r
        print("ok")
        sys.exit()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"http://localhost:{port}")
    ThreadingHTTPServer(("", port), H).serve_forever()
