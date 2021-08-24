rm /tmp/annulus.png /tmp/z.png
python3 -m numpy.f2py --opt="-O3" --arch="-march=native" --opt="-fopenmp" -c Src/Dp/src.f -m dsc -lgomp
python3 testdsc_airfoil.py 
python3 -m numpy.f2py --f77exec=flang --f90exec=flang --opt="-O3" --arch="-march=native" --opt="-fopenmp=libgomp" -c Src/Dp/src.f -m dsc -lgomp
python3 testdsc_airfoil.py 
xdg-open /tmp/annulus.png 
xdg-open /tmp/z.png
