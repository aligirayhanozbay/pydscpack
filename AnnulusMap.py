import dsc
import numpy as np

class AnnulusMap:
    initial_guess_types = {
        'equispaced': 0,
        'non-equispaced': 1
    }
    integration_path_types = {
        'line': 0,
        'circle': 1
    }
    def __init__(self, outer_polygon = None, inner_polygon = None, nptq = 8, tol = 1e-10, initial_guess = None, vertices_at_infinity = False, integration_path = None, mapping_params = None):
        '''
        Computes forward and backward maps between a doubly connected region with polygonal boundaries and an annulus
        
        Init arguments:
        -outer_polygon: np.array[np.complex64] or np.array[np.complex128]. Vertices of the outer polygon.
        -inner_polygon: np.array[np.complex64] or np.array[np.complex128]. Vertices of the inner polygon.
        -nptq: int. No of quadrature points for Gauss-Jacobi quadrature.
        -tol: float. Tolerance for the iterative solver to compute the mapping parameters.
        -initial_guess: str or Tuple[np.array[np.float64], np.array[np.float64]]. Initial guesses for the prevertex arguments (angles). If str, 'equispaced' and 'non-equispaced' are the options.
        -vertices_at_infinity: bool. Set to true if some vertices are at infinity.
        -integration_path: str. 'line' or 'circle'
        -mapping_params: dict. If this is provided, all mapping params are taken from this dict instead, and all other inputs are ignored
        '''

        if mapping_params is not None:
            self.mapping_params = mapping_params
        else:
            self.mapping_params = self._calculate_map(outer_polygon, inner_polygon, nptq, tol, initial_guess, vertices_at_infinity, integration_path)

        self.check_map()
        
    def _calculate_map(self, outer_polygon, inner_polygon, nptq, tol, initial_guess, vertices_at_infinity, integration_path):
    
        vertices_at_infinity = int(vertices_at_infinity)
        
        #handle initial guess method. TODO: handle user-supplied guesses
        if initial_guess is None:
            initial_guess = 1
        elif isinstance(initial_guess, str):
            initial_guess = self.initial_guess_types[initial_guess]

        #handle complex integral path
        if integration_path is None:
            integration_path = self.integration_path_types['circle']
        elif isinstance(integration_path, str):
            integration_path = self.integration_path_types[integration_path]
            

        #compute turning angles
        turning_angles_outer = dsc.angles(outer_coords, 0)
        turning_angles_inner = dsc.angles(inner_coords, 1)

        #compute Gauss-Jacobi quadrature points
        gj_quadrature_params = dsc.qinit(turning_angles_outer, turning_angles_inner, nptq)

        #check inputs
        dsc.check(turning_angles_outer, turning_angles_inner, vertices_at_infinity)

        #compute mapping
        u,c,w0,w1,phi0,phi1,uary,vary,dlam,iu,isprt,icount = dsc.dscsolv(tol, initial_guess, outer_polygon, inner_polygon, turning_angles_outer, turning_angles_inner, nptq, gj_quadrature_params, vertices_at_infinity, integration_path)

        #adjustments for inner radius of annulus
        dsc.thdata(uary,vary,dlam,iu,u)

        return self._pack_mapping_params(u,c,outer_polygon,inner_polygon,w0,w1,turning_angles_outer,turning_angles_inner,nptq,gj_quadrature_params,phi0,phi1,uary,vary,dlam,iu)

    @staticmethod
    def _pack_mapping_params(u,c,z0,z1,w0,w1,alfa0,alfa1,nptq,qwork,phi0,phi1,uary,vary,dlam,iu):
        mapping_params = {
            'inner_radius': u,
            'scaling': c, #see Eq 3.1 in Hu 1998
            'outer_polygon_vertices': z0,
            'inner_polygon_vertices': z1,
            'outer_polygon_prevertices':w0,
            'inner_polygon_prevertices':w1,
            'outer_polygon_turning_angles': alfa0,
            'inner_polygon_turning_angles': alfa1,
            'gj_quadrature_points': nptq,
            'gj_quadrature_params': qwork,
            'outer_prevertex_arguments': phi0,
            'inner_prevertex_arguments': phi1,
            'theta_mu': uary,
            'theta_v': vary,
            'theta_dlam': dlam,
            'theta_iu': iu
        }
        return mapping_params

    def check_map(self):
        #test mapping
        dsc.dsctest(self.mapping_params['theta_mu'], \
            self.mapping_params['theta_v'],
            self.mapping_params['theta_dlam'],
            self.mapping_params['theta_iu'],
            self.mapping_params['inner_radius'],
            self.mapping_params['scaling'],
            self.mapping_params['outer_polygon_prevertices'],
            self.mapping_params['inner_polygon_prevertices'],
            self.mapping_params['outer_polygon_vertices'],
            self.mapping_params['inner_polygon_vertices'],
            self.mapping_params['outer_polygon_turning_angles'],
            self.mapping_params['inner_polygon_turning_angles'],
            self.mapping_params['gj_quadrature_points'],
            self.mapping_params['gj_quadrature_params'])
            

    def forward_map(self, w, kww=0, ic=2, line_segment_only=False):
        '''
        Computes the forward map (annulus->polygonal domain).

        Inputs:
        -w: np.array[np.complex128]. Coordinates in the annular domain
        
        Outputs:
        -z: np.array[np.complex128]. Coordinates corresponding to each element of w in the polygonal domain.
        '''
        orig_shape = w.shape
        z = dsc.forward_map(w.reshape(-1), kww, ic,
                            self.mapping_params['inner_radius'],
                            self.mapping_params['scaling'],
                            self.mapping_params['outer_polygon_prevertices'],
                            self.mapping_params['inner_polygon_prevertices'],
                            self.mapping_params['outer_polygon_vertices'],
                            self.mapping_params['inner_polygon_vertices'],
                            self.mapping_params['outer_polygon_turning_angles'],
                            self.mapping_params['inner_polygon_turning_angles'],
                            self.mapping_params['outer_prevertex_arguments'],
                            self.mapping_params['inner_prevertex_arguments'],
                            self.mapping_params['gj_quadrature_points'],
                            self.mapping_params['gj_quadrature_params'],
                            int(not line_segment_only),
                            self.mapping_params['theta_mu'],
                            self.mapping_params['theta_v'],
                            self.mapping_params['theta_dlam'],
                            self.mapping_params['theta_iu']).reshape(orig_shape)
        return z

    def backward_map(self, z, eps=1e-8, line_segment_only=False):
        '''
        Computes the backward map (polygonal domain->annulus).

        Inputs:
        -z: np.array[np.complex128]. Coordinates in the polygonal domain
        
        Outputs:
        -w: np.array[np.complex128]. Coordinates corresponding to each element of z in the annular domain.
        '''
        orig_shape = z.shape
        w = dsc.backward_map(z.reshape(-1),
                            self.mapping_params['inner_radius'],
                            self.mapping_params['scaling'],
                            self.mapping_params['outer_polygon_prevertices'],
                            self.mapping_params['inner_polygon_prevertices'],
                            self.mapping_params['outer_polygon_vertices'],
                            self.mapping_params['inner_polygon_vertices'],
                            self.mapping_params['outer_polygon_turning_angles'],
                            self.mapping_params['inner_polygon_turning_angles'],
                            self.mapping_params['outer_prevertex_arguments'],
                            self.mapping_params['inner_prevertex_arguments'],
                            self.mapping_params['gj_quadrature_points'],
                            self.mapping_params['gj_quadrature_params'],
                            eps, int(not line_segment_only),
                            self.mapping_params['theta_mu'],
                            self.mapping_params['theta_v'],
                            self.mapping_params['theta_dlam'],
                            self.mapping_params['theta_iu']).reshape(orig_shape)
        return w

if __name__ == '__main__':

    from testdsc import unstructured_plot
    
    outer_coords = 5*np.array([2.5+1.5*1j, -1.5+1.5*1j, -1.5-1.5*1j, 2.5-1.5*1j]) #coordinates of the outer polygon
    inner_coords = -1.0*np.array([0.5, -0.5+0.5*1j, -0.5-0.5*1j]) #coordinates of the inner polygon

    amap = AnnulusMap(outer_coords, inner_coords, nptq = 64)

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
