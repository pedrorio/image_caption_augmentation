from setuptools import setup, find_packages
import os
# import sys

try:  # for pip >=12
    from pip._internal.req import parse_requirements
    try:
        from pip._internal import download
    except ImportError:  # for pip >= 20
        from pip._internal.network import session as download
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
    from pip import downloads

VERSION = '0.0.4'

# parse_requirements() returns generator of pip.req.InstallRequirement
# objects
install_reqs = parse_requirements(
    "requirements.txt", session=download.PipSession()
)
# install_requires is a list of requirement
try:
    install_requires = [str(ir.req) for ir in install_reqs]
except AttributeError:  # for pip >= 20
    install_requires = [str(ir.requirement) for ir in install_reqs]


def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()

setup(
    name="ica",
    version=VERSION,
    author="Pedro Rio",
    author_email="pedrocacaisrio@gmail.com",
    description="A package to augment image captions",
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/pedrorio/image_caption_augmentation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9.0',
    license='MIT',
    install_requires=install_requires,
)
