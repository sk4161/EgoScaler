from setuptools import find_packages, setup

setup(
    name="depth_anything",
    version="2.0",
    install_requires=[],
    package_dir={'': 'metric_depth'},  
    packages=find_packages(where='metric_depth')  
)