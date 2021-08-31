import pydsc
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
#python3 -m numpy.f2py -c Src/Dp/src.f -m dsc

def complexify(x):
    return np.array([complex(zs) for zs in x])

def check_laplacian(c, spacing, u, *wdsc_args):
    spacing = np.real(spacing)
    if not isinstance(c, np.ndarray):
        c = np.array(c)
    if not (c.dtype == np.complex128 or c.dtype==np.complex64):
        c = np.complex(*c)
    zcoords = np.array([c, c+spacing, c-spacing, c+1j*spacing, c-1j*spacing], dtype = c.dtype)
    wcoords = map_wdsc(zcoords,u,*wdsc_args)
    wnorm = np.real(wcoords * np.conj(wcoords))
    soln = 1-(1/np.log(u))*np.log(wnorm**0.5)
    laplacian = (np.sum(soln[1:]) - 4*soln[0])/(spacing**2)
    return laplacian
    

    
if __name__ == '__main__':

    nptq = 64 #No of Gauss-Jacobi integration pts
    tol = 1e-10 #tolerance for iterative process
    iguess = 1 #initial guess type. see line 770 is src.f - not sure what this does yet.
    ishape = 0 #0 for no vertices at infinity
    linearc = 1

    outer_coords = 5*complexify(['1.5+1.5j', '-1.5+1.5j', '-1.5-1.5j', '1.5-1.5j']) #coordinates of the outer polygon
    inner_coords = -1.0*complexify(['0.5', '-0.5+0.5j', '-0.5-0.5j']) #coordinates of the inner polygon

    #q = np.sqrt(2)
    #q = 0.25
    #outer_coords = 1.5*complexify(['1+j', '-1+j', '-1-j', '1-j']) #coordinates of the outer polygon
    #inner_coords = complexify([f'{q}+0.0j', f'0.0+{q}j', f'-{q}+0.0j', f'0.0-{q}j']) #coordinates of the inner polygon
    amap = pydsc.AnnulusMap(outer_coords, inner_coords)
    
    n_plotpts = (50,200)
    r = np.linspace(amap.mapping_params['inner_radius'],1.0-(1e-5),n_plotpts[0]) #important to not evaluate map at r=1.0 (outer annulus ring)
    theta = np.linspace(0,2*np.pi,n_plotpts[1])
    a = np.exp(theta*1j)
    #import pdb; pdb.set_trace()
    wplot = np.einsum('i,j->ij', r, a)
    wnorm = np.real(wplot * np.conj(wplot))
    wangle = np.angle(wplot)
    amap.plot_map('norm', 'argument', w=wplot, save_path='/tmp/radius_and_argument.png')

    zplot = np.array([0.65+0.10*1j,0.75+0.1*1j,0.65-0.10*1j,0.75-0.1*1j])
    wplot = amap.backward_map(zplot)
    amap.plot_map(np.ones(zplot.shape),z=zplot, plot_type='scatter', save_path='/tmp/triangle_sensors_z.png')
    



