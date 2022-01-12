import os
import platform
from numpy.distutils.core import Extension

isa = platform.machine()
if isa == 'x86_64' or isa == 'i386':
	arch_flag = '-march=native'
else:
	arch_flag = '-mcpu=native'
compiler_opts = ['-fopenmp']
debug_compiler_opts = ['-O0', '-g', '-fbacktrace']
release_compiler_opts = ['-O3', arch_flag]
        
if __name__ == '__main__':

        debug_build = os.environ.get('PYDSCPACK_DEBUG_BUILD', '0')
        
        if bool(int(debug_build)):
                compiler_opts = compiler_opts + debug_compiler_opts
        else:
                compiler_opts = compiler_opts + release_compiler_opts
        
        dsc = Extension(name='dsc', sources=['pydscpack/Src/Dp/src.f'], extra_f77_compile_args=compiler_opts, extra_f90_compile_args=compiler_opts, libraries=['gomp'])
        
        pkg_name = 'pydscpack'
        from numpy.distutils.core import setup
        setup(
                name=pkg_name,
                version='0.0.1',
                author='Ali Girayhan Ozbay',
                url='https://github.com/aligirayhanozbay/pydscpack',
                classifiers=[
                        'Programming Language :: Python :: 3',
                        'Operating System :: Unix',
                        'Programming Language :: Fortran', 
                        'Topic :: Scientific/Engineering :: Mathematics'
                ],
                description='A python toolkit to compute doubly connected Schwarz-Christoffel mappings',
                ext_modules=[dsc],
                install_requires=['numpy', 'matplotlib']
        )
