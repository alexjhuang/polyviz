# polyviz

https://alexjhuang.github.io/polyviz/

Live polyhedral visualizer for Python loop nests. Type a nest, see its iteration
domain as a 3D polyhedron, scrub/play execution order, and morph it through an
affine schedule. Array accesses in the innermost body (`A[i][j] = A[i-1][j] + ...`)
are traced into RAW/WAR/WAW dependence edges, checked for legality under the schedule.

For more context you can read: https://libisl.sourceforge.io/tutorial.pdf

If you want to run locally: 
```sh
pip install numpy scipy
python app.py          # http://localhost:8765
python app.py --check  # self-test
```

