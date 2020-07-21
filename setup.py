#!/usr/bin/env python
import os
import setuptools
from setuptools import setup, Command
import unittest

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')
 
def sickle_tests():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('test', pattern='test_*.py')
    return test_suite

setup(name='sickle',
      version='0.1',
      description='Data aggregator synthesis',
      author='Xiangyu Zhou, Chenglong Wang',
      author_email='',
      url='https://github.com/KevinXiangyuZhou/Sickle',
      packages=setuptools.find_packages(),
      test_suite='setup.sickle_tests',
      cmdclass={'clean': CleanCommand,})