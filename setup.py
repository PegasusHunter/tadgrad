from setuptools import setup, find_packages

VERSION = '0.1.0'

URL = 'https://github.com/cospectrum/tadgrad.git'

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='tadgrad',
    version=VERSION,
    license='MIT',
    url=URL,
    description='Machine Learning from scratch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Alexey Severin',
    install_requires=requirements,
    packages=find_packages(),
    package_data={'tadgrad': ['py.typed']},
    keywords=['python'],
)

