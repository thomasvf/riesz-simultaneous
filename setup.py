#!/usr/bin/env python
from setuptools import setup, find_packages


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('requirements_dev.txt') as f:
    requirements_dev = f.read().splitlines()

setup(
    author="Thomas Vaitses Fontanari",
    author_email='tvfontanari@gmail.com',
    python_requires='~=3.10',
    description="Implementation of simultaneous motion and amplitude magnification with Riesz pyramids",
    entry_points={
        'console_scripts': [
            'riesz_simul_gui=rieszsimultaneous.scripts.riesz_simul_gui:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    package_data={
        "rieszsimultaneous": ["resources/*"],
    },
    name='rieszsimultaneous',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    test_suite='src/tests',
    extras_require={
        "dev": requirements_dev
    },
    version='0.1.0',
    zip_safe=False,
)
