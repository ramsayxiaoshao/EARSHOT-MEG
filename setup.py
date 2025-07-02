from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extension_args = {
    'define_macros': [("NPY_NO_DEPRECATED_API", "NPY_1_11_API_VERSION")],
    'include_dirs': [numpy.get_include()],
}

setup(
    name='earshot',
    version='1.0',
    packages=['earshot'],
    entry_points={
        'console_scripts': [
                    'earshot-train = earshot.train_earshot:main',
                    'earshot-train-decoder = earshot.train_decoder:main',
                    'earshot-evaluate = earshot.evaluate:main',
                ]
    },
    ext_modules=cythonize([
        Extension("earshot._op", ["earshot/_op.pyx"], **extension_args),
    ]),
    zip_safe=False,
)
