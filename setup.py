import platform
from numpy.distutils.core import Extension

isa = platform.machine()
if isa == 'x86_64' or isa == 'i386':
	arch_flag = '-march=native'
else:
	arch_flag = '-mcpu=native'

compiler_opts = ['-O3',arch_flag,'-fopenmp']
dsc = Extension(name='dsc', sources=['pydsc/Src/Dp/src.f'], extra_f77_compile_args=compiler_opts, extra_f90_compile_args=compiler_opts, libraries=['gomp'])

if __name__ == '__main__':
	pkg_name = 'pydsc'
	from numpy.distutils.core import setup
	setup(
		name=pkg_name,
		description='A python toolkit to compute doubly connected Schwarz-Christoffel mappings',
		ext_modules=[dsc]
	)
