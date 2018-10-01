#!/usr/bin/env python

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='bit2byte',
      ext_modules=[CppExtension('bit2byte', ['bit2byte.cpp'])],
      cmdclass={'build_ext': BuildExtension})