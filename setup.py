import os
from setuptools import setup, find_packages

path = os.path.abspath(os.path.dirname(__file__))

readme = open(path + "/docs/README.md")

setup(
  name="stgraph_trainer",
  version="1.0.0",
  description="Experments for Spatio-Temporal graph neural networks",
  url="https://github.com/KienMN/STGNN-for-Covid-in-Korea",
  author="Kien MN",
  author_email="kienmn97@gmail.com",
  license="MIT",
  packages=find_packages(exclude=["tests", "docs", ".gitignore"]),
  install_requires=[""],
  dependency_links=[""],
  include_package_data=True
)