#!/usr/bin/env python

from distutils.core import setup

setup(name='orbit',
      version='1.0',
      description='Python Keplerian Orbit Library',
      author='Michael Kessler',
      author_email='mikepkes@gmail.com',
      packages=['orbit'],
      package_dir={'orbit':'src/orbit'},
      scripts=['scripts/vieworbits'],
     )
