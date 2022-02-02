from setuptools import setup
from myresources.crocodile import __version__

with open("README.md", "r") as file:
    long_desc = file.read()  # to help adding README to PyPi website not only Github


setup(
    name='crocodile',
    version=__version__,
    packages=['crocodile'],
    # packages=setuptools.find_packages(where="myresources"),
    package_dir={'': 'myresources'},
    py_modules=['toolbox', 'core', 'file_management', 'meta', "deeplearning", "deeplearning_torch"],
    url='https://github.com/thisismygitrepo/crocodile',
    project_urls={
        "Bug Tracker": "https://github.com/thisismygitrepo/crocodile/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # 3.9
    license='Apache 2.0',
    author='Alex Al-Saffar',
    author_email='programmer@usa.com',
    description='Making Python even more convenient by extending list and dict and pathlib and more.',
    long_description=long_desc,
    long_description_content_type="text/markdown",
    install_requires=[
        # CORE:
        "numpy",
        "scipy",  # heavy-weight.
        "pandas",
        "matplotlib",

        # Accessories
        # h5py
        "joblib",  # multitasking
        "pip",
        "setuptools",  # for packaging  # correct place is in .toml
        "wheel",
        "twine",  # for pushing package to pypi.org
        "pytest",  # popular alternative to builtint unittest
        "ipython",  # interactive python
        "fire",  # for automatic CLI interface
        "tqdm",  # for progress bar
        "send2trash",  # move to recycle bin
        "clipboard",
        "dill",  # extends pickle
        "cryptography",  # for encoding
        "paramiko",  # for SSH
        "requests",  # interacting with web
        "pyyaml",  # storing yaml files.
        "cycler",

        "scikit-image",  # image processing. Heavy-weight.

        # torch
        # tensorflow
    ]

)


with open("./myresources/crocodile/art/happy_croco", "r") as file:
    croco = file.read()  # search ascii art or characters art.
print(croco)


# useful webiste: gitignore.io
# choosealicense.com

# steps:
# git push origin
# populates build, dist and .egg directories:
# python setup.py sdist bdist_wheel
# twine upload dist/*
# Locally: (only once)
# pip install -e . # add --user if there is .toml file
# The files backed up here OneDrive/AppData/home/.pypirc saves the credentials needed by Twine to uploade to pypi
# Note: the tag v.54 or whatever convention is followed must be consistent as pypi decides on which version is the latest
# based on that. E.g. if the version was named 5.6 as opposed tp v5.6 then 5.6 will appear as old release.
# similarly if v5.6 is used instead of v5.60, then 5.6 will not be the latest
