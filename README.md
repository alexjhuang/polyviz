# polyviz

https://alexjhuang.github.io/polyviz/

Live polyhedral visualizer for Python loop nests. Type a nest, see its iteration
domain as a 3D polyhedron, scrub/play execution order, and morph it through an
affine schedule.

For more context you can read: https://libisl.sourceforge.io/tutorial.pdf

If you want to run locally: 
```sh
pip install numpy scipy
python app.py          # http://localhost:8765
python app.py --check  # self-test
```

