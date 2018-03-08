"""Setup for the models monitoring package."""

import sys
import warnings
import setuptools
import versioneer

# Require Python 3.5 or higher
if sys.version_info.major < 3 or sys.version_info.minor < 5:
    warnings.warn("The models monitoring requires Python 3.5 or higher!")
    sys.exit(1)

with open('README.rst') as f:
    README = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    author="Nathan Tedgui",
    author_email="nathan.tedgui@gmail.com",
    name="natools",
    long_description=README,
    url="https://github.com/nted92/natools.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
