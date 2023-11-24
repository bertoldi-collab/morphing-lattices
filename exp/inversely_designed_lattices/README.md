# Notes

Each folder contains the following files:

- labeled_material_distribution.png: A visualization of the material distribution in the lattice with numbered nodes and struts.
- 3d_printing_data.npz: A python dictionary with the following keys:
  - `'points'`: The coordinates of the nodes in the lattice.
  - `'connectivity'`: The connectivity of the lattice i.e. an array of shape (n_struts, 2) where each row contains the indices of the nodes connected by the strut.
  - `'phase'`: Array of 0s and 1s indicating the material of each strut: 0 → HTNI, 1 → LTNI.
