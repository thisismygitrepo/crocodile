from setuptools import setup

with open("README.md", "r") as file:
    long_desc = file.read()  # to help adding README to PyPi website not only Github

setup(
    name='alexlib',
    version='0.0.4',
    packages=['alexlib'],
    package_dir={'': 'myresources'},
    py_modules=['toolbox'],
    url='https://github.com/thisismygitrepo/alexlib',
    license='Apache 2.0',
    author='Alex Al-Saffar',
    author_email='programmer@usa.com',
    description='Making Python even more convenient by extending list and dict and pathlib and more.',
    long_description=long_desc,
    long_description_content_type="text/markdown",
)

# useful webiste: gitignore.io
# choosealicense.com
