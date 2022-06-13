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
        #import pdb; pdb.set_trace()
        outer_polygon = self._sort_ccw(outer_polygon)
        inner_polygon = self._sort_ccw(inner_polygon)
    
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
        u,c,w0,w1,phi0,phi1,uary,vary,dlam,iu,isprt,icount = dsc.dscsolv(tol, initial_guess, outer_polygon, inner_polygon, turning_angles_outer, turning_angles_inner, gj_quadrature_params, vertices_at_infinity, integration_path, nptq=nptq)

        #adjustments for inner radius of annulus
        dsc.thdata(uary,vary,dlam,iu,u)

        return self._pack_mapping_params(u,c,outer_polygon,inner_polygon,w0,w1,turning_angles_outer,turning_angles_inner,nptq,gj_quadrature_params,phi0,phi1,uary,vary,dlam,iu)

    @staticmethod
    def _sort_ccw(z):
        angles = np.angle(z)
        sort_indices = np.argsort(angles)
        return z[sort_indices]
    
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
            self.mapping_params['gj_quadrature_params'],
            nptq=self.mapping_params['gj_quadrature_points'])
            

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
                            self.mapping_params['gj_quadrature_params'],
                            int(not line_segment_only),
                            self.mapping_params['theta_mu'],
                            self.mapping_params['theta_v'],
                            self.mapping_params['theta_dlam'],
                            self.mapping_params['theta_iu'],
                            nptq=self.mapping_params['gj_quadrature_points']).reshape(orig_shape)
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
                            self.mapping_params['gj_quadrature_params'],
                            eps, int(not line_segment_only),
                            self.mapping_params['theta_mu'],
                            self.mapping_params['theta_v'],
                            self.mapping_params['theta_dlam'],
                            self.mapping_params['theta_iu'],
                            nptq=self.mapping_params['gj_quadrature_points']).reshape(orig_shape)
        return w

    def _generate_annular_grid(self, n_pts=None, return_polar=False, **map_params):
        '''
        Generates a grid of points, equispaced in the radial and angular directions on the annular domain (w-domain). Returns the coordinates in w and z domains.

        Inputs:
        -npts: Tuple[int]. npts[0] is the # of points in the radial direction. npts[1] is the # of points in the angular direction.
        -**map_params: kwargs for the optional args of self.forward_map.
        
        Outputs:
        Tuple[np.array] of dtype np.complex128. First element contains complex coordinates of the gridpoints in the w-domain and the second element in the z domain.
        '''
        if n_pts is None:
            n_pts = (50,200)
        r = np.linspace(self.mapping_params['inner_radius'],1.0-(1e-5),n_pts[0]) #important to not evaluate map at r=1.0 (outer annulus ring)
        theta = np.linspace(0,2*np.pi,n_pts[1],endpoint=False)
        if return_polar:
            return np.stack(np.meshgrid(r,theta,indexing='ij'),-1)
        else:
            a = np.exp(theta*1j)
            w = np.einsum('i,j->ij', r, a)
            z = self.forward_map(w, **map_params)
            return w,z

    def dzdw(self, w_coord):
        #dz/dw = C * wprod(w) - read area around Eq 2.6 in Hu 1998
        orig_shape = w_coord.shape
        C = self.mapping_params['scaling']
        wprod = dsc.map_wprod(self.mapping_params['theta_mu'], self.mapping_params['theta_v'], self.mapping_params['theta_dlam'], self.mapping_params['theta_iu'], w_coord.reshape(-1), self.mapping_params['inner_radius'], self.mapping_params['outer_polygon_prevertices'], self.mapping_params['inner_polygon_prevertices'], self.mapping_params['outer_polygon_turning_angles'], self.mapping_params['inner_polygon_turning_angles'])
        return C*(wprod.reshape(orig_shape))

    def dwdz(self, w_coord):
        return 1/self.dzdw(w_coord)

    def _dzdw_wproddebug(self, w_coord):
        #this should yield the same result as self.dzdw!
        orig_shape = list(w_coord.shape)
        w_coord = w_coord.reshape(-1)
        
        sigma_outer, _ = self._sigma_outer(w_coord, 8)
        
        sigma_inner, _ = self._sigma_inner(w_coord, 8)
        
        res = self.mapping_params['scaling'] * np.prod(sigma_outer, axis=0) * np.prod(sigma_inner, axis=0)

        return res.reshape(orig_shape)

    def _dwdz_wproddebug(self, w_coord):
        return 1/self._dzdw_wproddebug(w_coord)

    def _sigma_outer(self, w_coord, series_terms = 8):
        #assume w_coord is 1 dimensional
        npts = w_coord.shape[0]
        j = np.arange(1,series_terms+1)
        mu = self.mapping_params['inner_radius']
        M = len(self.mapping_params['outer_polygon_prevertices'])
        a0 = self.mapping_params['outer_polygon_turning_angles']
        wk = self.mapping_params['outer_polygon_prevertices']

        w = np.tile(w_coord.reshape(1,1,-1),(M,series_terms,1))
        j = np.tile(j.reshape(1,-1,1),(M,1,npts))
        wk = np.tile(wk.reshape(-1,1,1),(1,series_terms,npts))
        a0 = np.tile(a0.reshape(-1,1),(1,npts))

        sigma_outer_term_sum = np.sum(mu**(j**2)*((-w/(mu*wk))**j + (-w/(mu*wk))**(-j)), axis=1)
        sigma_outer_prime_term_sum = np.sum(mu**(j**2)*(j*(-w/(mu*wk))**j/w - j*(-w/(mu*wk))**(-j)/w), axis=1)
        
        sigma_outer = 1+sigma_outer_term_sum
        sigma_outer_derivs = (a0-1)*(sigma_outer**(a0-2))*(sigma_outer_prime_term_sum)
        sigma_outer = sigma_outer**(a0-1)

        return sigma_outer, sigma_outer_derivs

    def _sigma_inner(self, w_coord, series_terms = 8):
        #assume w_coord is 1 dimensional
        npts = w_coord.shape[0]
        j = np.arange(1,series_terms+1)
        mu = self.mapping_params['inner_radius']
        N = len(self.mapping_params['inner_polygon_prevertices'])
        a1 = self.mapping_params['inner_polygon_turning_angles']
        wk = self.mapping_params['inner_polygon_prevertices']

        w = np.tile(w_coord.reshape(1,1,-1),(N,series_terms,1))
        j = np.tile(j.reshape(1,-1,1),(N,1,npts))
        wk = np.tile(wk.reshape(-1,1,1),(1,series_terms,npts))
        a1 = np.tile(a1.reshape(-1,1),(1,npts))

        sigma_inner_term_sum = np.sum(mu**(j**2)*((-mu*w/wk)**j + (-mu*w/wk)**(-j)),axis=1)
        sigma_inner_prime_term_sum = np.sum(mu**(j**2)*(j*(-mu*w/wk)**j/w - j*(-mu*w/wk)**(-j)/w),axis=1)

        sigma_inner = 1+sigma_inner_term_sum
        sigma_inner_derivs = (a1-1)*(sigma_inner**(a1-2))*(sigma_inner_prime_term_sum)
        sigma_inner = sigma_inner**(a1-1)

        return sigma_inner, sigma_inner_derivs

    def _wprod_derivative(self, w_coord, series_terms = 8):

        orig_shape = list(w_coord.shape)
        w_coord = w_coord.reshape(-1)

        sigma_outer = np.stack(self._sigma_outer(w_coord, series_terms),0)
        outer_derivative_mask = np.eye(sigma_outer.shape[1], dtype=np.int64)
        outer_derivative_mask = np.stack([1-outer_derivative_mask,outer_derivative_mask],0)
        
        sigma_inner = np.stack(self._sigma_inner(w_coord, series_terms),0)
        inner_derivative_mask = np.eye(sigma_inner.shape[1], dtype=np.int64)
        inner_derivative_mask = np.stack([1-inner_derivative_mask,inner_derivative_mask],0)

        term1 = np.sum(np.prod(np.einsum('bjk, bkl-> jkl', outer_derivative_mask, sigma_outer), axis=1), axis=0) * np.prod(sigma_inner[0], axis=0)
        term2 = np.sum(np.prod(np.einsum('bjk, bkl-> jkl', inner_derivative_mask, sigma_inner), axis=1), axis=0) * np.prod(sigma_outer[0], axis=0)

        return term1 + term2
        
    
    def d2zdw2(self, w_coord, series_terms = 8):
        orig_shape = list(w_coord.shape)
        C = self.mapping_params['scaling']
        wprod_prime = self._wprod_derivative(w_coord.reshape(-1), series_terms).reshape(orig_shape)
        return C * wprod_prime

    def d2wdz2(self, w_coord, series_terms = 8):
        d2zdw2 = self.d2zdw2(w_coord, series_terms)
        dzdw = self.dzdw(w_coord)
        return -d2zdw2/(dzdw**3)
        

    def test_map(self):
        return dsc.dsctest(
            self.mapping_params['theta_mu'],
            self.mapping_params['theta_v'],
            self.mapping_params['theta_dlam'],
            self.mapping_params['theta_iu'],
            self.mapping_params['inner_radius'], self.mapping_params['scaling'],
            self.mapping_params['outer_polygon_prevertices'],
            self.mapping_params['inner_polygon_prevertices'],
            self.mapping_params['outer_polygon_vertices'],
            self.mapping_params['inner_polygon_vertices'],
            self.mapping_params['outer_polygon_turning_angles'],
            self.mapping_params['inner_polygon_turning_angles'],
            self.mapping_params['gj_quadrature_params'],
            nptq=self.mapping_params['gj_quadrature_points'])

    def plot_map(self, *fields,
                 w = None,
                 z = None,
                 xlim = None,
                 ylim = None,
                 draw_boundaries = True,
                 save_path = None,
                 n_pts = None,
                 plot_type = None,
                 cmaps = None,
                 plot_alignment=None, **map_params):
        '''
        Plot quantities in both the annular (w-) and original (z-) coordinates.

        Inputs:
        -fields: List[Union[np.array, str]]. Fields to plot. If str, 
        -w: np.array. Locations of the data points in fields in w-coordinates.
        -z: np.array. Locations of the data points in fields in z-coordinates. Do not specify w and z simultaneously.
        -draw_boundaries: bool. True to enable drawing of the bounding polygons.
        -save_path: str. Directory to save the output in.
        -n_pts: Tuple[int]. In case w and z are not supplied, this argument may be used to specify the number of gridpoints in radial and angular dimensions in the w-plane.
        -plot_type: str. 'contour', 'contourf' or 'scatter'.
        -cmaps: str or List[str]. matplotlib colormaps.
        -plot_alignment: str. 'horizontal' or 'vertical'. How the output subplots are aligned.
        -map_params: Dict. Additional params to supply to self.forward_map/self.backward_map.
        '''
        
        if plot_type is None:
            plot_type = ['contour' for _ in fields]
        elif isinstance(plot_type, str):
            plot_type = [plot_type for _ in fields]
        plot_type_map = {'contour': lambda ax,x,y,f,**kwargs: ax.tricontour(x,y,f,**kwargs), 'contourf': lambda ax,x,y,f,**kwargs: ax.tricontourf(x,y,f,**kwargs), 'scatter': lambda ax,x,y,f,**kwargs: ax.scatter(x,y,c=f,**kwargs)}
        
        if len(fields) == 0:
            fields = ['norm', 'argument']
        default_npts = (50,200) #(radial resolution, angular resolution)
        if n_pts is None:
            for field in fields:
                try:
                    n_pts = field.shape
                except:
                    n_pts = default_npts

        if cmaps is None:
            cmaps = [None for _ in fields]
        
        w_reals = []
        w_imags = []
        z_reals = []
        z_imags = []
        if w is None and z is None:
            w,z = self._generate_annular_grid(n_pts=n_pts, **map_params)
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
        if ((plot_alignment is None) or (plot_alignment=='horizontal')):
            subplot_partitioning = [1,2]
        elif (plot_alignment=='vertical'):
            subplot_partitioning = [2]
        else:
            raise(ValueError(f'Invalid plot alignment {plot_alignment}'))
        fig, (z_ax,w_ax) = plt.subplots(*subplot_partitioning)
        w_ax.set_xlabel('Re(w)')
        w_ax.set_ylabel('Im(w)')
        z_ax.set_xlabel('Re(z)')
        z_ax.set_ylabel('Im(z)')
        fig.tight_layout()

        for field,ptype,zreal,zimag,wreal,wimag,cmap in zip(fields, plot_type, z_reals, z_imags, w_reals, w_imags,cmaps):
            if isinstance(field, str):
                if field == 'norm':
                    field = (wreal**2 + wimag**2)**0.5
                elif field == 'argument':
                    field = np.angle(wreal + 1j*wimag)
            plot_type_map[ptype](z_ax, zreal.reshape(-1), zimag.reshape(-1), field.reshape(-1),cmap=cmap)
            plot_type_map[ptype](w_ax, wreal.reshape(-1), wimag.reshape(-1), field.reshape(-1),cmap=cmap)
            
        w_ax.add_patch(plt.Circle((0,0), self.mapping_params['inner_radius'], edgecolor='k', fill=True, facecolor= 'purple', zorder = 999))
        w_ax.add_patch(plt.Circle((0,0), 1.0, edgecolor='k', fill=False))

        if draw_boundaries:
            opv = self.mapping_params['outer_polygon_vertices']
            ipv = self.mapping_params['inner_polygon_vertices']
            opv_real = np.stack([np.real(opv), np.imag(opv)], -1)
            ipv_real = np.stack([np.real(ipv), np.imag(ipv)], -1)
            z_ax.add_patch(plt.Polygon(opv_real, edgecolor='k', fill=False, zorder = 999))
            z_ax.add_patch(plt.Polygon(ipv_real, edgecolor='k', facecolor = 'purple', fill=True, zorder=999))

        if xlim is not None:
            z_ax.set_xlim(*xlim)
        if ylim is not None:
            z_ax.set_ylim(*ylim)

        w_ax.set_aspect('equal')
        z_ax.set_aspect('equal')
        

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, bbox_inches='tight')
            
        plt.close()

    def export_mapping_params_h5(self, dataset):
        import h5py
        if isinstance(dataset, str):
            dataset = h5py.File(dataset,'w')

        for val_name in self.mapping_params:
            dataset.create_dataset(val_name, data = self.mapping_params[val_name])
        
        

if __name__ == '__main__':
    
    outer_coords = 5*np.array([2.5+1.5*1j, -1.5+1.5*1j, -1.5-1.5*1j, 2.5-1.5*1j]) #coordinates of the outer polygon
    inner_coords = -1.0*np.array([0.5, -0.5+0.5*1j, -0.5-0.5*1j]) #coordinates of the inner polygon
    
    amap = AnnulusMap(outer_coords, inner_coords, nptq = 64)

    print(amap.test_map())
    amap.plot_map('norm', 'argument', save_path='/tmp/norm_and_argument.png')

    import h5py
    f = h5py.File('/tmp/test_amap.h5','w')
    grp = f.create_group('test')
    amap.export_mapping_params_h5(grp)
