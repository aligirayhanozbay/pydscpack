import dsc
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
#python3 -m numpy.f2py -c Src/Dp/src.f -m dsc

def complexify(x):
    return np.array([complex(zs) for zs in x])


def unstructured_plot(w, *obj_boundaries, f=None, arg = None, plotname = './plot.png'):
    x = np.real(w).reshape(-1)
    y = np.imag(w).reshape(-1)
    plt.figure()

    if arg is None and f is not None:
        f = np.real(f * np.conj(f)).reshape(-1)
        plt.tricontourf(x,y,f)
    else:
        f = np.real(f * np.conj(f)).reshape(-1)
        plt.tricontour(x,y,f)
    if arg is not None:
        arg = arg.reshape(-1)
        plt.tricontour(x,y,arg)

    for obj_boundary in obj_boundaries:
        for start, end in zip(obj_boundary, np.roll(obj_boundary,-1)):
            s_real, s_imag = np.real(start), np.imag(start)
            e_real, e_imag = np.real(end), np.imag(end)
            plt.plot([s_real, e_real], [s_imag,e_imag])
    plt.axis('equal')
    plt.colorbar()
    plt.savefig(plotname)
    plt.close()

def wrap_zdsc(indices, queue, ww, kww, ic, u, c, w0, w1, z0, z1, alfa0, alfa1, phi0, phi1, nptq, qwork, iopt):
    for idx, w in zip(indices,ww): 
        z = dsc.zdsc(w, kww, ic, u, c, w0, w1, z0, z1, alfa0, alfa1, phi0, phi1, nptq, qwork, iopt)
        queue.put((idx, z))

def wrap_wdsc(indices, queue, zz, u, c, w0, w1, z0, z1, alfa0, alfa1, phi0, phi1, nptq, qwork, eps, iopt):
    for idx, z in zip(indices,zz):
        w = dsc.wdsc(z, u, c, w0, w1, z0, z1, alfa0, alfa1, phi0, phi1, nptq, qwork, eps, iopt)
        queue.put((idx, w))

def map_transform(transform, coords, *other_args):
    processes = []
    queue = multiprocessing.Queue()
    indices = np.arange(coords.shape[0])
    nprocs = min(multiprocessing.cpu_count(), coords.shape[0])
    coords_split = np.array_split(coords,nprocs)
    indices_split = np.array_split(indices,nprocs)
    result = np.zeros(coords.shape, dtype=coords.dtype)
    for slice_idx,coords_slice in zip(indices_split,coords_split):
        p = multiprocessing.Process(target=transform, args=[slice_idx, queue, coords_slice] + list(other_args))
        processes.append(p)
        p.start()
    for k in range(result.shape[0]):
        idx, val = queue.get()
        result[idx] = val
    for p in processes:
        p.join()
        p.close()
    return result

def map_zdsc(ww, kww, ic, u, c, w0, w1, z0, z1, alfa0, alfa1, phi0, phi1, nptq, qwork, iopt):
    return map_transform(wrap_zdsc, ww, kww, ic, u, c, w0, w1, z0, z1, alfa0, alfa1, phi0, phi1, nptq, qwork, iopt)

def map_wdsc(zz, u, c, w0, w1, z0, z1, alfa0, alfa1, phi0, phi1, nptq, qwork, eps, iopt):
    return map_transform(wrap_wdsc, zz, u, c, w0, w1, z0, z1, alfa0, alfa1, phi0, phi1, nptq, qwork, eps, iopt)

# def map_zdsc(ww, kww, ic, u, c, w0, w1, z0, z1, alfa0, alfa1, phi0, phi1, nptq, qwork, iopt):
#     processes = []
#     queue = multiprocessing.Queue()
#     indices = np.arange(ww.shape[0])
#     nprocs = min(multiprocessing.cpu_count(), ww.shape[0])
#     ww_split = np.array_split(ww,nprocs)
#     indices_split = np.array_split(indices,nprocs)
#     zz = np.zeros(ww.shape, dtype=ww.dtype)
#     for wwidx,w in zip(indices_split,ww_split):
#         p = multiprocessing.Process(target=wrap_zdsc, args=(wwidx, queue, w, kww, ic, u, c, w0, w1, z0, z1, alfa0, alfa1, phi0, phi1, nptq, qwork, iopt))
#         processes.append(p)
#         p.start()
#     for k in range(zz.shape[0]):
#         wwidx, z = queue.get()
#         zz[wwidx] = z
#     for p in processes:
#         p.join()
#         p.close()
#     return zz

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

    outer_coords = 5*complexify(['2.5+1.5j', '-1.5+1.5j', '-1.5-1.5j', '2.5-1.5j']) #coordinates of the outer polygon
    inner_coords = -1.0*complexify(['0.5', '-0.5+0.5j', '-0.5-0.5j']) #coordinates of the inner polygon

    #q = np.sqrt(2)
    #outer_coords = (1+q) * complexify(['1+j', '-1+j', '-1-j', '1-j']) #coordinates of the outer polygon
    #inner_coords = complexify([f'{q}+0.0j', f'0.0+{q}j', f'-{q}+0.0j', f'0.0-{q}j']) #coordinates of the inner polygon

    alfa0 = dsc.angles(outer_coords, 0) #turning angles for the outer and inner polygon.
    alfa1 = dsc.angles(inner_coords, 1)
    qwork = dsc.qinit(alfa0, alfa1, nptq) # quadrature nodes
    dsc.check(alfa0, alfa1, ishape)

    u,c,w0,w1,phi0,phi1 = dsc.dscsolv(tol, iguess, outer_coords, inner_coords, alfa0, alfa1, nptq, qwork, ishape, linearc) #mapping parameters
    dsc.thdata(u)
    dsc.dsctest(u,c,w0,w1,outer_coords,inner_coords,alfa0,alfa1,nptq,qwork)

    test_pt = complex('1.50+0.0j')
    ww = dsc.wdsc(test_pt, u, c, w0, w1, outer_coords, inner_coords, alfa0, alfa1, phi0, phi1, nptq, qwork, 1e-6, 1)
    print(ww)
    zz = dsc.zdsc(ww, 0, 2, u, c, w0, w1, outer_coords, inner_coords, alfa0, alfa1, phi0, phi1, nptq, qwork, 1)
    print(zz)

    n_plotpts = (50,200)
    r = np.linspace(u,1.0-(1e-5),n_plotpts[0]) #important to not evaluate map at r=1.0 (outer annulus ring)
    theta = np.linspace(0,2*np.pi,n_plotpts[1])
    a = np.exp(theta*1j)
    #import pdb; pdb.set_trace()
    wplot = np.einsum('i,j->ij', r, a)
    wnorm = np.real(wplot * np.conj(wplot))
    wangle = np.angle(wplot)
    
    zplot = map_zdsc(wplot.reshape(-1), 0, 2, u, c, w0, w1, outer_coords, inner_coords, alfa0, alfa1, phi0, phi1, nptq, qwork, 1).reshape(wplot.shape)

    np.save('wplot.npy',wplot)
    np.save('zplot.npy',zplot)
    unstructured_plot(wplot, f=wnorm, arg = wangle, plotname='/tmp/annulus.png')
    unstructured_plot(zplot, outer_coords, inner_coords, f=wnorm, arg=wangle, plotname='/tmp/z.png')

    #solve laplace eq with Dirichlet BCs - 0 on inner annulus ring and 1 on outer.
    #Analytical soln is u(r,theta) = u(r) = 1-ln(r)/ln(u)
    A0 = 1.0
    B0 = -A0/np.log(u)
    uplot = 1-(1/np.log(u))*np.log(wnorm**0.5)
    unstructured_plot(wplot, f=uplot, plotname='/tmp/laplace_annulus.png')
    unstructured_plot(zplot, outer_coords, inner_coords, f=uplot, plotname='/tmp/laplace_z.png')

    spacings = 5*np.array([10**(-n) for n in range(5)])
    laplacians = []
    for spacing in spacings:
        laplacian = check_laplacian(1.75+1.75*1j, spacing, u, c, w0, w1, outer_coords, inner_coords, alfa0, alfa1, phi0, phi1, nptq, qwork, 1e-6, 1)
        laplacians.append(laplacian)

    plt.figure()
    plt.loglog(1/spacings**2, laplacians)
    plt.xlabel('1/dx^2')
    plt.ylabel('error @ 1.75+1.75i')
    plt.savefig('/tmp/laplacian_convergence.png')
    plt.close()
    #import pdb; pdb.set_trace()
    #print(alfa0)
    #print(alfa1)
    #print(qwork)



