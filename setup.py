#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

# with open('README.md') as readme_file:
#     readme = readme_file.read()
readme = ""

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

test_requirements = ['pytest>=3', ]

setup(
    author="Thomas Vaitses Fontanari",
    author_email='tvfontanari@gmail.com',
    python_requires='>=3.6',
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
    name='rieszsimultaneous',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    test_suite='src/tests',
    tests_require=test_requirements,
    version='0.1.0',
    zip_safe=False,
)
