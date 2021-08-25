#import os
#import glob
from numpy.distutils.core import Extension

dsc = Extension(name='dsc', sources=['pydsc/Src/Dp/src.f'], extra_f77_compile_args=['-O3','-march=native','-fopenmp'], libraries=['gomp'])

if __name__ == '__main__':
	pkg_name = 'pydsc'
	from numpy.distutils.core import setup
	setup(
		name=pkg_name,
		description='A python toolkit to compute doubly connected Schwarz-Christoffel mappings',
		ext_modules=[dsc]
	)
	#library = glob.glob('dsc*.so')[0]
	#os.rename(library, pkg_name + '/' + library)
