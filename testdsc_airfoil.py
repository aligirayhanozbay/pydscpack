import pydsc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from testdsc import unstructured_plot, complexify, map_zdsc

def read_airfoil(path, complex = True):
    with open(path, 'r') as f:
        lines = f.readlines()
        pts = []
        for line in lines:
            vals = line.split()
            if len(vals) == 2:
                pts.append([float(v) for v in vals])
    #if not ((pts[0][0] == 0.0) and (pts[0][1] == 0.0)):
    #    pts = pts[1:]
    pts = np.array(pts)
    if complex:
        pts = pts[:,0] + 1j * pts[:,1]
    return pts



def sort_ccw(pts):
    arguments = np.angle(pts)
    new_indices = np.argsort(arguments)
    return pts[new_indices]
    

if __name__ == '__main__':
    af_coords = read_airfoil('n0012.dat')
    centroid = np.mean(af_coords)
    af_coords = af_coords - centroid
    #af_coords = sort_ccw(af_coords)

    nptq = 8 #No of Gauss-Jacobi integration pts
    tol = 1e-10 #tolerance for iterative process
    iguess = 1 #initial guess type. see line 770 is src.f - not sure what this does yet.
    ishape = 0 #0 for no vertices at infinity
    linearc = 1

    outer_coords = 1.5*complexify(['1+j', '-1+j', '-1-j', '1-j']) #coordinates of the outer polygon
    inner_coords = af_coords[::4]
    print(len(inner_coords))
    
    amap = pydsc.AnnulusMap(outer_coords, inner_coords)
    
    n_plotpts = (50,200)
    r = np.linspace(amap.mapping_params['inner_radius'],1.0-(1e-5),n_plotpts[0]) #important to not evaluate map at r=1.0 (outer annulus ring)
    theta = np.linspace(0,2*np.pi,n_plotpts[1])
    a = np.exp(theta*1j)
    #import pdb; pdb.set_trace()
    wplot = np.einsum('i,j->ij', r, a)
    wnorm = np.real(wplot * np.conj(wplot))
    wangle = np.angle(wplot)

    zplot = amap.forward_map(wplot)
    unstructured_plot(wplot, f=wnorm, arg = wangle, plotname='/tmp/annulus.png')
    unstructured_plot(zplot, outer_coords, inner_coords, f=wnorm, arg=wangle, plotname='/tmp/z.png')
    
    zplot = np.array([0.65+0.10*1j,0.75+0.1*1j,0.65-0.10*1j,0.75-0.1*1j])
    wplot = amap.backward_map(zplot)
    amap.plot_map(np.ones(zplot.shape),z=zplot, plot_type='scatter', save_path='/tmp/af_sensors_z.png')
    amap.plot_map(np.ones(zplot.shape),w=wplot, plot_type='scatter', save_path='/tmp/af_sensors_w.png')
    
