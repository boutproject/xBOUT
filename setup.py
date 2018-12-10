import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()
version_dict = {}
with open("xbout/_version.py") as f:
    exec(f.read(), version_dict)

name = 'xBOUT'
version = version_dict['__version__']
release = version

setup(
    name=name,
    version=version,
    url="https://github.com/boutproject/xBOUT",
    author="Thomas Nicholas",
    author_email="thomas.nicholas@york.ac.uk",
    description='Collect data from BOUT++ runs in python using xarray',
    license="Apache",
    python_requires='>=3.5',
    install_requires=['xarray>=v0.11.0',
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
