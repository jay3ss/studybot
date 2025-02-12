from setuptools import find_packages, setup

setup(
    name="StudyBot",
    version="0.1.0",
    packages=find_packages(where="app"),
    package_dir={"": "app"},
)
