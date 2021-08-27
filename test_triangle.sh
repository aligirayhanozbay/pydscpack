rm /tmp/annulus.png /tmp/z.png
python3 -m numpy.f2py --f77flags='-fopenmp' --f90flags='-fopenmp' --opt='-mcpu=native' --opt='-O3' -c Src/Dp/src.f -m dsc -lgomp
python3 testdsc.py 
xdg-open /tmp/annulus.png 
xdg-open /tmp/z.png
