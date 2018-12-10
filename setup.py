import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="xBOUT",
    version="0.1",
    url="https://github.com/TomNicholas/xBOUT",
    author="Thomas Nicholas",
    author_email="thomas.nicholas@york.ac.uk",
    description='Collect data from BOUT++ runs in python using xarray',
    license="Apache",
    python_requires='>=3.5',
    install_requires=['xarray>=v0.10.0',
                      'dask[array]>=1.0.0',
                      'natsort>=5.5.0',
                      'matplotlib>=2.2',
                      'animatplot>=0.3'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
    packages=find_packages(),
    include_package_data=True)
