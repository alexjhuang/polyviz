# polyviz

Live polyhedral visualizer for Python loop nests. Type a nest, see its iteration
domain as a 3D polyhedron, scrub/play execution order, and morph it through an
affine schedule.

```sh
pip install numpy scipy
python app.py          # http://localhost:8765
python app.py --check  # self-test
```

Also runs fully in-browser via Pyodide when served statically (GitHub Pages).
