### PyDSCPACK

PyDSCPACK is a Python package to compute doubly connected Schwarz-Christoffel conformal mappings for polygonal domains, wrapping a modified version of the code originally written by Hu (1998):

```
Hu, Chenglie 1998 Algorithm 785: A software package for computing schwarz-christoffel conformal transformation for doubly connected polygonal regions. ACM Trans.Math.Softw.24(3), 317–333.
```

To get started, clone the repository and install with the package with `pip3 -e install .` The following packages are necessary to build and run this package on Ubuntu 20.04:

`python3-dev python3-pip python3-numpy python3-matplotlib gfortran`

For general usage, import `pydscpack.AnnulusMap` and supply the coordinates of the vertices of the inner and outer polygons bounding the doubly connected region. Note that in some cases, the mapping quality might be poor due to the limitations of the original algorithm.

NOTE: This project is licensed under the BSD 3-Clause license, except for pydscpack/Src/Dp/src.f, which is licensed under the [ACM Software License Agreement](https://www.acm.org/publications/policies/software-copyright-notice). This license permits redistribution, usage and modification free of charge ONLY FOR NONCOMMERCIAL PURPOSES. AUTHOR(S) ARE NOT LIABLE FOR ANY DAMAGES WHICH MAY RESULT FROM NONCOMPLIANCE WITH THE ACM SOFTWARE LICENSING AGREEMENT TERMS.
