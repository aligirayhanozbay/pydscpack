import dsc
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
#python3 -m numpy.f2py -c Src/Dp/src.f -m dsc

def complexify(x):
    return np.array([complex(zs) for zs in x])


def unstructured_plot(w, *obj_boundaries, f=None, plotname = './plot.png'):
    x = np.real(w).reshape(-1)
    y = np.imag(w).reshape(-1)
    if f is not None:
        f = np.real(f * np.conj(f)).reshape(-1)
    plt.figure()
    plt.scatter(x,y,c=f)
    #plt.tricontour(x,y,f)

    for obj_boundary in obj_boundaries:
        for start, end in zip(obj_boundary, np.roll(obj_boundary,-1)):
            s_real, s_imag = np.real(start), np.imag(start)
            e_real, e_imag = np.real(end), np.imag(end)
            plt.plot([s_real, e_real], [s_imag,e_imag])
    
    plt.colorbar()
    plt.savefig(plotname)
    plt.close()

def wrap_zdsc(indices, queue, ww, kww, ic, u, c, w0, w1, z0, z1, alfa0, alfa1, phi0, phi1, nptq, qwork, iopt):
    for idx, w in zip(indices,ww): 
        z = dsc.zdsc(w, kww, ic, u, c, w0, w1, z0, z1, alfa0, alfa1, phi0, phi1, nptq, qwork, iopt)
        queue.put((idx, z))

def map_zdsc(ww, kww, ic, u, c, w0, w1, z0, z1, alfa0, alfa1, phi0, phi1, nptq, qwork, iopt):
    processes = []
    queue = multiprocessing.Queue()
    indices = np.arange(ww.shape[0])
    nprocs = min(multiprocessing.cpu_count(), ww.shape[0])
    ww_split = np.array_split(ww,nproc)
    indices_split = np.array_split(indices,nproc)
    zz = np.zeros(ww.shape, dtype=ww.dtype)
    for wwidx,w in zip(indices_split,ww_split):
        p = multiprocessing.Process(target=wrap_zdsc, args=(wwidx, queue, w, kww, ic, u, c, w0, w1, z0, z1, alfa0, alfa1, phi0, phi1, nptq, qwork, iopt))
        processes.append(p)
        p.start()
    for k in range(zz.shape[0]):
        wwidx, z = queue.get()
        zz[wwidx] = z
    for p in processes:
        p.join()
        p.close()
    return zz

if __name__ == '__main__':

    nptq = 64 #No of Gauss-Jacobi integration pts
    tol = 1e-10 #tolerance for iterative process
    iguess = 1 #initial guess type. see line 770 is src.f - not sure what this does yet.
    ishape = 0 #0 for no vertices at infinity
    linearc = 1

    outer_coords = complexify(['1+j', '-1+j', '-1-j', '1-j']) #coordinates of the outer polygon
    inner_coords = complexify(['0.5', '-0.5+0.5j', '-0.5-0.5j']) #coordinates of the inner polygon

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
    
    zplot = map_zdsc(wplot.reshape(-1), 0, 2, u, c, w0, w1, outer_coords, inner_coords, alfa0, alfa1, phi0, phi1, nptq, qwork, 1).reshape(wplot.shape)

    np.save('wplot.npy',wplot)
    np.save('zplot.npy',zplot)
    unstructured_plot(wplot, f=wnorm, plotname='/tmp/annulus.png')
    unstructured_plot(zplot, outer_coords, inner_coords, f=wnorm, plotname='/tmp/z.png')

    #import pdb; pdb.set_trace()
    #print(alfa0)
    #print(alfa1)
    #print(qwork)



