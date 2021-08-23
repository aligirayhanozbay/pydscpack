import dsc
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
    print(arguments)
    new_indices = np.argsort(arguments)
    return pts[new_indices]
    

if __name__ == '__main__':
    af_coords = read_airfoil('n0012.dat')
    centroid = np.mean(af_coords)
    af_coords = af_coords - centroid

    nptq = 8 #No of Gauss-Jacobi integration pts
    tol = 1e-10 #tolerance for iterative process
    iguess = 1 #initial guess type. see line 770 is src.f - not sure what this does yet.
    ishape = 0 #0 for no vertices at infinity
    linearc = 1

    outer_coords = 1.5*complexify(['1+j', '-1+j', '-1-j', '1-j']) #coordinates of the outer polygon
    inner_coords = af_coords[::4]
    print(len(inner_coords))
    
    alfa0 = dsc.angles(outer_coords, 0) #turning angles for the outer and inner polygon.
    alfa1 = dsc.angles(inner_coords, 1)
    qwork = dsc.qinit(alfa0, alfa1, nptq) # quadrature nodes
    print(qwork.shape)
    dsc.check(alfa0, alfa1, ishape)

    u,c,w0,w1,phi0,phi1,uary,vary,dlam,iu,isprt,icount = dsc.dscsolv(tol, iguess, outer_coords, inner_coords, alfa0, alfa1, nptq, qwork, ishape, linearc) #mapping parameters
    dsc.thdata(uary,vary,dlam,iu,u)
    dsc.dsctest(uary,vary,dlam,iu,u,c,w0,w1,outer_coords,inner_coords,alfa0,alfa1,nptq,qwork)

    test_pt = complex('1.50+0.0j')
    ww = dsc.wdsc(test_pt, u, c, w0, w1, outer_coords, inner_coords, alfa0, alfa1, phi0, phi1, nptq, qwork, 1e-8, 1, uary, vary, dlam, iu)
    print(ww)
    zz = dsc.zdsc(ww, 0, 2, u, c, w0, w1, outer_coords, inner_coords, alfa0, alfa1, phi0, phi1, nptq, qwork, 1, uary, vary, dlam, iu)
    print(zz)

    n_plotpts = (50,200)
    r = np.linspace(u,1.0-(1e-5),n_plotpts[0]) #important to not evaluate map at r=1.0 (outer annulus ring)
    theta = np.linspace(0,2*np.pi,n_plotpts[1])
    a = np.exp(theta*1j)
    
    wplot = np.einsum('i,j->ij', r, a)
    wnorm = np.real(wplot * np.conj(wplot))
    wangle = np.angle(wplot)

    zplot = map_zdsc(wplot.reshape(-1), 0, 2, u, c, w0, w1, outer_coords, inner_coords, alfa0, alfa1, phi0, phi1, nptq, qwork, 1, uary, vary, dlam, iu).reshape(wplot.shape)
    
    
    np.save('wplot.npy',wplot)
    np.save('zplot.npy',zplot)
    unstructured_plot(wplot, f=wnorm, arg = wangle, plotname='/tmp/annulus.png')
    unstructured_plot(zplot, outer_coords, inner_coords, f=wnorm, arg = wangle, plotname='/tmp/z.png')

    #solve laplace eq with Dirichlet BCs - 0 on inner annulus ring and 1 on outer.
    #Analytical soln is u(r,theta) = u(r) = 1-ln(r)/ln(u)
    A0 = 1.0
    B0 = -A0/np.log(u)
    uplot = 1-(1/np.log(u))*np.log(wnorm**0.5)
    unstructured_plot(wplot, f=uplot, circle=u, plotname='/tmp/laplace_annulus.png')
    unstructured_plot(zplot, outer_coords, inner_coords, f=uplot, plotname='/tmp/laplace_z.png')
