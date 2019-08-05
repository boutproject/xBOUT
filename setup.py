from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

version_dict = {}
with open("xbout/_version.py") as f:
    exec(f.read(), version_dict)

name = 'xBOUT'
version = version_dict['__version__']
release = version

extras_require = {
  'calc': ['numpy >= 1.13.0', 'scipy >= 1.3.0', 'dask >= 2.2.0',
           'statsmodels >= 0.10.1', 'xrft', 'xhistogram'],
  'docs': ['sphinx >= 1.4'],
}

packages = ['xbout', 'xbout.calc']

tests = [p + '.tests' for p in packages]

setup(
    name=name,
    version=version,
    url="https://github.com/boutproject/xBOUT",
    author="Thomas Nicholas",
    author_email="thomas.nicholas@york.ac.uk",
    description='Collect data from BOUT++ runs in python using xarray',
    license="Apache",
    python_requires='>=3.5',
    install_requires=[
        'xarray>=v0.12.2',
        'dask[array]>=1.0.0',
        'natsort>=5.5.0',
        'matplotlib>=2.2',
        'animatplot>=0.3',
        'netcdf4>=1.4.0',
    ],
    extras_require=extras_require,
    tests_require=['pytest >= 3.3.0'],
    long_description=long_description,
    long_description_content_type="text/markdown",
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
    packages=packages + tests,
    include_package_data=True,
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release),
            'source_dir': ('setup.py', 'docs'),
        }
    },
)
