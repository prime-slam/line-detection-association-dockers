from setuptools import find_packages, setup
from setuptools.extension import Extension
import os, sys, sysconfig


_DEBUG = False
_DEBUG_LEVEL = 0

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ['-std=c++11', '-Wall', '-Wextra']
if _DEBUG:
    extra_compile_args += ['-g', '-O0', '-DDEBUG=%s' % _DEBUG_LEVEL, '-UNDEBUG']
else:
    extra_compile_args += ['-DNDEBUG', '-O3']


setup(
    name='kht',
    version='2.0.0',
    description='Kernel-Based Hough Transform (KHT) for Python',
    author='Leandro A. F. Fernandes',
    author_email='laffernandes@ic.uff.br',
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[
        Extension(
            'kht',
            sources=[os.path.join('source', basename) for basename in os.listdir('source') if basename.endswith('.cpp')],
            include_dirs=['/usr/local/include'],
            library_dirs=['/usr/local/lib'],
            runtime_library_dirs=['/usr/local/lib'],
            libraries=[f'boost_python{sys.version_info.major}{sys.version_info.minor}'],
            extra_compile_args=extra_compile_args,
            language='c++11',
        )
    ],
    install_requires=[
        'numpy',
    ],
)

# See https://www.boost.org/doc/libs/1_77_0/libs/python/doc/html/building/installing_boost_python_on_your_.html
