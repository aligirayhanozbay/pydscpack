from scipy.io import FortranFile
import matplotlib.pyplot as plt
import numpy as np

from testdsc import unstructured_plot

def read_ffile(path, dtype=np.complex128):
    vals = []
    with FortranFile(path,'r') as f:
        while True:
            try:
                vals.append(f.read_reals(dtype))
            except:
                break
    return np.array(vals, dtype=dtype)


'''
Use this script to visualize the mapping generated by the dscpack binary
'''

if __name__ == '__main__':

    fnames = ['wplot.bin', 'wnorm.bin', 'zplot.bin', 'z0.bin', 'z1.bin']
    wplot, wnorm, zplot, z0, z1 = list(map(read_ffile, fnames, [np.complex128, np.float64, np.complex128, np.complex128,np.complex128]))

    unstructured_plot(wplot, f=wnorm, plotname='/tmp/annulus_fort.png')
    unstructured_plot(zplot, z0, z1, f = wnorm, plotname='/tmp/z_fort.png')

