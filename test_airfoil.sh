rm /tmp/annulus.png /tmp/z.png
python3 -m numpy.f2py -c Src/Dp/src.f -m dsc
python3 testdsc_airfoil.py 
xdg-open /tmp/annulus.png 
xdg-open /tmp/z.png
