# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="spatracker",
    version="1.0",
    install_requires=[],
    package_dir={'': 'models'},  
    packages=find_packages(where='models')  
)