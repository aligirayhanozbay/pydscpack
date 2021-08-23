import matplotlib.pyplot as plt
import numpy as np

from testdsc import unstructured_plot

# def ring_map(r_inner_orig,r_inner_target,r_outer_orig,r_outer_target):
#     #try to fit mapping such that r'=a*r^2+b*r
#     lhs_matrix = np.array([[r_inner_orig**2,r_inner_orig],[r_outer_orig**2,r_outer_orig]])
#     rhs = np.array([r_inner_target, r_outer_target])
#     soln = np.linalg.solve(lhs_matrix,rhs)

#     def map_z(z):
#         return soln[0]*z*(z*np.conj(z))**0.5 + soln[1]*z
    
#     return soln, map_z

def ring_map(r_inner_orig,r_inner_target,r_outer_orig,r_outer_target):
    #try to fit mapping such that r'=a*exp(b*r)
    #r=(z*conj(z))**0.5, r'=(z'*conj(z'))**0.5
    #forward map: z'=a*exp(i*arg(z) + b*r)
    #backward map: z=(ln(r'/a)/b)*exp(i*arg(z'))
    lhs_matrix = np.array([[1.0,r_inner_orig],[1.0,r_outer_orig]])
    rhs = np.log(np.array([r_inner_target, r_outer_target]))
    soln = np.linalg.solve(lhs_matrix,rhs)
    soln[0] = np.exp(soln[0])

    def forward_map(z):
        zr = (z*np.conj(z))**0.5
        za = np.angle(z)
        return soln[0]*np.exp(1j*za + soln[1]*zr)

    def backward_map(w):
        wr = (w*np.conj(w))**0.5
        wa = np.angle(w)
        return (1/soln[1])*np.log(wr/soln[0])*np.exp(1j*wa)
    
    return soln, forward_map, backward_map

if __name__ == '__main__':

    ri,ro = 0.435,1.0
    riprime,roprime = 0.35,1.0

    n_plotpts = (50,200)
    r = np.linspace(ri,ro,n_plotpts[0]) #important to not evaluate map at r=1.0 (outer annulus ring)
    theta = np.linspace(0,2*np.pi,n_plotpts[1])
    a = np.exp(theta*1j)
    
    wplot = np.einsum('i,j->ij', r, a)
    wnorm = np.real(wplot * np.conj(wplot))
    wangle = np.angle(wplot)

    c,umap,wmap = ring_map(ri,riprime,ro,roprime)
    print(c)
    uplot = umap(wplot)
    unorm = np.real(uplot*np.conj(uplot))**0.5
    uangle = np.angle(uplot)
    print(np.min(unorm),np.max(unorm))

    unstructured_plot(wplot,f=wnorm, arg=wangle, plotname='/tmp/wplot.png')
    unstructured_plot(uplot,f=unorm, arg=uangle, plotname='/tmp/uplot.png')
