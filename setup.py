from setuptools import find_packages
from setuptools import setup

required_packages = [
    "tensorflow",
    "numpy",
    "matplotlib",
    "pandas",
    "sklearn",
    "imageio"]

setup(name='HumanProteinAtlas',
      version='0.1',
      packages=find_packages(),
      description="Human Protein Atlas Project",
      author="Jon Deaton",
      author_email="jdeaton@stanford.edu",
      license='MIT',
      zip_safe=False,
      install_requires=required_packages)
