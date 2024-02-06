from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()


setup(
    name="bioagent",
    version="0.0.4",
    description="",
    author="Daniel",
    license="Apache License 2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
