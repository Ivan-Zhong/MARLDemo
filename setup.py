import os
from setuptools import find_packages, setup

setup(
    name="marldemo",
    version="1.0.0",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.7",
    install_requires=[
        "pettingzoo==1.22.2",
        "supersuit==3.7.0",
        "gym==0.21.0",
        "pyglet==1.5.0",
        "importlib-metadata==4.13.0",
        "torch>=1.8.0",
        "pyyaml>=5.3.1",
        "tensorboard>=2.2.1",
        "tensorboardX",
        "setproctitle"
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
)
