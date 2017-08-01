#!/usr/bin/env python

from setuptools import setup


setup_requires = []

scripts = []


CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)
Operating System :: MacOS
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Topic :: Scientific/Engineering

"""
setup(
    name="lab",
    version="0.0.1",
    packages=['lab',
              'lab.analysis',
              'lab.classes',
              'lab.figures',
              'lab.misc',
              'lab.plotting',
              ],
    install_requires=[
        'numpy>=1.8',
        'scipy>=0.13.0',
        'scikit-image>=0.9.3',
        'shapely>=1.2.14',
        'scikit-learn>=0.11',
        'pillow>=2.6.1',
        'future>=0.14',
    ],
    scripts=scripts,
    author="Losonczy Lab",
    author_email="software@losonczylab.org",
    description="Losonczy Lab Analysis Bundle",
    license="GNU GPLv2",
    keywords="imaging microscopy neuroscience behavior",
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    setup_requires=setup_requires,
    url="http://www.losonczylab.org/",
    platforms=["Linux", "Mac OS-X", "Windows"],
)
