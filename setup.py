import os
from setuptools import setup, find_packages

# Set the base path relative to this file's location
base_dir = os.path.abspath(os.path.dirname(__file__))

# Read requirements.txt with absolute path
requirements_path = os.path.join(base_dir, 'requirements.txt')
requirements = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        requirements = f.read().splitlines()

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

__version__ = "1.0.1"

packages = find_packages(exclude=("deeva.tests",))
setup(
    name="deeva",
    version=__version__,
    author = read("AUTHORS.txt").replace('\n', ', ').replace('-', ''),
    author_email="vahram.babadjanyan@gmail.com",
    description="Object Detection Data Analysis Toolbox",
    long_description=read('README.md'),
    long_description_content_type = 'text/markdown',
    keywords = ["deeva", "object-detection", "analytics", "visualization",
                "datasets", "toolkit", "statistics", "detection", "streamlit",
                "plotly"],
    url = "https://github.com/vbyan/DEEVA",
    packages=packages,
    package_dir={"DEEVA": "deeva"},
    install_requires=requirements,
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'deeva=deeva.start:main',
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
    ],
    license="Apache License 2.0",
    python_requires='>=3.9',

)