from plot_unf import read_ffile
import numpy as np

fnames = ['wplot.bin', 'zplot.bin']
wplot_f, zplot_f = list(map(read_ffile, fnames))
wplot_n, zplot_n = np.load('wplot.npy').reshape(wplot_f.shape, order='c'), np.load('zplot.npy').reshape(zplot_f.shape, order='c')
wdiff = wplot_f - wplot_n
zdiff = zplot_f - zplot_n
print(np.mean(np.real(wdiff * np.conj(wdiff))))
print(np.mean(np.real(zdiff * np.conj(zdiff))))
