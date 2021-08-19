import dsc
import numpy as np
import matplotlib.pyplot as plt
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

if __name__ == '__main__':

    nptq = 64 #No of Gauss-Jacobi integration pts
    tol = 1e-10 #tolerance for iterative process
    iguess = 1 #initial guess type. see line 770 is src.f - not sure what this does yet.
    ishape = 0 #0 for no vertices at infinity
    linearc = 0
    u = 0.0 #some sort of parameter computed by the program
    c = complex('0.0') #some sort of parameter computed by the program

    # outer_coords = complexify(['1+j', '-1+j', '-1-j', '1-j']) #coordinates of the outer polygon
    # inner_coords = complexify(['0.5', '-0.5+0.5j', '-0.5-0.5j']) #coordinates of the inner polygon

    q = np.sqrt(2)
    outer_coords = (1+q) * complexify(['1+j', '-1+j', '-1-j', '1-j']) #coordinates of the outer polygon
    inner_coords = complexify([f'{q}+0.0j', f'0.0+{q}j', f'-{q}+0.0j', f'0.0-{q}j']) #coordinates of the inner polygon

    test_pt = complex('1.50+0.0j')

    alfa0 = np.zeros(outer_coords.shape, dtype = np.float64) #turning angles for the outer polygon. computed.
    alfa1 = np.zeros(inner_coords.shape, dtype = np.float64) #turning angles for the inner polygon. computed.
    w0 = np.zeros(outer_coords.shape, dtype = np.complex128) #w0, w1, phi0, phi1 are mapping params. computed.
    w1 = np.zeros(inner_coords.shape, dtype = np.complex128)
    phi0 = np.zeros(outer_coords.shape, dtype = np.float64)
    phi1 = np.zeros(inner_coords.shape, dtype = np.float64)
    qwork = np.zeros([nptq*(3+2*(outer_coords.shape[0] + inner_coords.shape[0]))], dtype = np.float64) #quadrature nodes
    #qwork = np.zeros((1660,), dtype=np.float64)

    dsc.angles(outer_coords, alfa0, 0)
    dsc.angles(inner_coords, alfa1, 1)
    dsc.qinit(alfa0, alfa1, nptq, qwork)
    dsc.check(alfa0, alfa1, ishape)

    u,c,w0,w1,phi0,phi1 = dsc.dscsolv(tol, iguess, outer_coords, inner_coords, alfa0, alfa1, nptq, qwork, ishape, linearc)

    # print('---------')
    # print(tol)
    # print('---------')
    # print(iguess)
    # print('---------')
    # print(u)
    # print('---------')
    # print(c)
    # print('---------')
    # print(w0)
    # print('---------')
    # print(w1)
    # print('---------')
    # print(phi0)
    # print('---------')
    # print(phi1)
    # print('---------')
    # print(outer_coords)
    # print('---------')
    # print(inner_coords)
    # print('---------')
    # print(alfa0)
    # print('---------')
    # print(alfa1)
    # print('---------')
    # print(nptq)
    # print('---------')
    # print(qwork)
    # print('---------')
    # print(ishape)
    # print('---------')
    # print(linearc)

    dsc.thdata(u)
    #print(u)
    dsc.dsctest(u,c,w0,w1,outer_coords,inner_coords,alfa0,alfa1,nptq,qwork)
    ww = dsc.wdsc(test_pt, u, c, w0, w1, outer_coords, inner_coords, alfa0, alfa1, phi0, phi1, nptq, qwork, 1e-6, 1)
    print(ww)
    zz = dsc.zdsc(ww, 0, 2, u, c, w0, w1, outer_coords, inner_coords, alfa0, alfa1, phi0, phi1, nptq, qwork, 1)
    print(zz)

    n_plotpts = (50,200)
    r = np.linspace(u,1.0-(1e-5),n_plotpts[0]) #important to not evaluate map at r=1.0 (outer annulus ring)
    theta = np.linspace(0,2*np.pi,n_plotpts[1])
    c = np.exp(theta*1j)
    #import pdb; pdb.set_trace()
    wplot = np.einsum('i,j->ij', r, c)
    wnorm = np.real(wplot * np.conj(wplot))

    zplot = np.zeros(wplot.shape, dtype=wplot.dtype)
    for i in range(wplot.shape[0]):
        for j in range(wplot.shape[1]):
            zplot[i,j] = dsc.zdsc(wplot[i,j], 0, 2, u, c, w0, w1, outer_coords, inner_coords, alfa0, alfa1, phi0, phi1, nptq, qwork, 1)

    unstructured_plot(wplot, f=wnorm, plotname='/tmp/annulus.png')
    unstructured_plot(zplot, outer_coords, inner_coords, f=wnorm, plotname='/tmp/z.png')

    #print(alfa0)
    #print(alfa1)
    #print(qwork)



