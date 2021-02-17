from setuptools import setup
from myresources.crocodile import __version__

with open("README.md", "r") as file:
    long_desc = file.read()  # to help adding README to PyPi website not only Github

setup(
    name='crocodile',
    version=__version__,
    packages=['crocodile'],
    package_dir={'': 'myresources'},
    py_modules=['toolbox', "deeplearning", "deeplearning_torch"],
    url='https://github.com/thisismygitrepo/crocodile',
    license='Apache 2.0',
    author='Alex Al-Saffar',
    author_email='programmer@usa.com',
    description='Making Python even more convenient by extending list and dict and pathlib and more.',
    long_description=long_desc,
    long_description_content_type="text/markdown",
)

# useful webiste: gitignore.io
# choosealicense.com

# steps:
# git push origin
# python setup.py sdist bdist_wheel
# twine upload dist/*
#
