import dsc
import numpy as np
import matplotlib.pyplot as plt
import itertools

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
        turning_angles_outer = dsc.angles(outer_polygon, 0)
        turning_angles_inner = dsc.angles(inner_polygon, 1)

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

    def plot_map(self, *fields, w = None, z = None, xlim = None, ylim = None, draw_boundaries = True, save_path = None, n_pts = None, plot_type = None, **map_params):
        '''
        Plot quantities in both the annular (w-) and original (z-) coordinates.

        Inputs:
        -fields: List[Union[np.array, str]]. Fields to plot. If str, 
        -w: np.array. Locations of the data points in fields in w-coordinates.
        -z: np.array. Locations of the data points in fields in z-coordinates. Do not specify w and z simultaneously.
        -draw_boundaries: bool. True to enable drawing of the vertices.
        -save_path: str. Directory to save the output in.
        -n_pts: Tuple[int]. In case w and z are not supplied, this argument may be used to specify the number of gridpoints in radial and angular dimensions in the w-plane.
        -map_params: Dict. Additional params to supply to self.forward_map/self.backward_map.
        '''
        #import pdb; pdb.set_trace()
        if plot_type is None:
            plot_type = ['contour' for _ in fields]
        elif isinstance(plot_type, str):
            plot_type = [plot_type for _ in fields]
        plot_type_map = {'contour': lambda ax,x,y,f: ax.tricontour(x,y,f), 'contourf': lambda ax,x,y,f: ax.tricontourf(x,y,f), 'scatter': lambda ax,x,y,f: ax.scatter(x,y,c=f)}
        
        if len(fields) == 0:
            fields = ['norm', 'argument']
        default_npts = (50,200) #(radial resolution, angular resolution)
        if n_pts is None:
            for field in fields:
                try:
                    n_pts = field.shape
                except:
                    n_pts = default_npts
        
        w_reals = []
        w_imags = []
        z_reals = []
        z_imags = []
        if w is None and z is None:
            r = np.linspace(self.mapping_params['inner_radius'],1.0-(1e-5),n_pts[0]) #important to not evaluate map at r=1.0 (outer annulus ring)
            theta = np.linspace(0,2*np.pi,n_pts[1],endpoint=False)
            a = np.exp(theta*1j)
            w = np.einsum('i,j->ij', r, a)
            z = self.forward_map(w, **map_params)
            w = itertools.repeat(w,len(fields))
            z = itertools.repeat(z,len(fields))
        elif w is not None and z is None:
            if isinstance(w,np.ndarray):
                z = self.forward_map(w, **map_params)
                w = itertools.repeat(w,len(fields))
                z = itertools.repeat(z,len(fields))
            else:
                z = [self.forward_map(ws, **map_params) for ws in w]
        elif z is not None and w is None:
            if isinstance(z,np.ndarray):
                w = self.backward_map(z, **map_params)
                z = itertools.repeat(z,len(fields))
                w = itertools.repeat(w,len(fields))
            else:
                w = [self.backward_map(zs, **map_params) for zs in z]
        else:
            raise(ValueError('Supply w or z, but not both.'))
        for ws, zs in zip(w,z): 
            wreal, wimag = np.real(ws).reshape(-1), np.imag(ws).reshape(-1)
            zreal, zimag = np.real(zs).reshape(-1), np.imag(zs).reshape(-1)
            w_reals.append(wreal)
            w_imags.append(wimag)
            z_reals.append(zreal)
            z_imags.append(zimag)
        
        plt.figure()
        fig, (z_ax,w_ax) = plt.subplots(2)
        
        if draw_boundaries:
            for obj_boundary in [self.mapping_params['outer_polygon_vertices'], self.mapping_params['inner_polygon_vertices']]:
                for start, end in zip(obj_boundary, np.roll(obj_boundary,-1)):
                    s_real, s_imag = np.real(start), np.imag(start)
                    e_real, e_imag = np.real(end), np.imag(end)
                    z_ax.plot([s_real, e_real], [s_imag,e_imag])
                    
        for field,ptype,zreal,zimag,wreal,wimag in zip(fields, plot_type, z_reals, z_imags, w_reals, w_imags):
            if isinstance(field, str):
                if field == 'norm':
                    field = (wreal**2 + wimag**2)**0.5
                elif field == 'argument':
                    field = np.angle(wreal + 1j*wimag)
            plot_type_map[ptype](z_ax, zreal.reshape(-1), zimag.reshape(-1), field.reshape(-1))
            plot_type_map[ptype](w_ax, wreal.reshape(-1), wimag.reshape(-1), field.reshape(-1))
            
        w_ax.add_patch(plt.Circle((0,0), self.mapping_params['inner_radius'], edgecolor='k', fill=False))
        w_ax.add_patch(plt.Circle((0,0), 1.0, edgecolor='k', fill=False))

        if xlim is not None:
            z_ax.set_xlim(*xlim)
        if ylim is not None:
            z_ax.set_ylim(*ylim)

        w_ax.set_aspect('equal')
        z_ax.set_aspect('equal')

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
            
        plt.close()
        
        

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
