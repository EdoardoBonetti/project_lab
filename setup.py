from setuptools import find_packages, setup

setup(
    name='lab',
    author='Edoardo Bonetti',
    # the directory is contained in the "lab_module" folder
    version='0.0.1',
    packages=find_packages(),

    # add polar to the list of dependencies
    # in the future pandas --> polars
    install_requires=['matplotlib', 'numpy', 'pandas', 'scipy', 'scikit-learn']
)
