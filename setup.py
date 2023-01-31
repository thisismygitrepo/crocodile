from setuptools import setup
from myresources.crocodile import __version__
import setuptools

with open("README.md", "r", encoding="utf-8") as file:
    long_desc = file.read()  # to help adding README to PyPi website not only Github

# get python modules:
# tb.P("./myresources").search("*.py", r=True).apply(lambda x: x.split(at="myresources", sep=-1)[1].as_posix().replace("/", ".")[:-3]).filter(lambda x: "__init__" not in x).list

setup(
    name='crocodile',
    version=__version__,
    packages=['crocodile'],
    # packages=setuptools.find_packages(where="myresources"),
    package_dir={'': 'myresources'},
    # py_modules=setuptools.find_packages(where="myresources") + ['crocodile.msc.ascii_art'],
    py_modules=['crocodile.core',
                'crocodile.croshell',
                'crocodile.database',
                'crocodile.deeplearning',
                'crocodile.deeplearning_torch',
                'crocodile.environment',
                'crocodile.file_management',
                'crocodile.matplotlib_management',
                'crocodile.meta',
                'crocodile.plotly_management',
                'crocodile.run',
                'crocodile.toolbox',
                'crocodile.cluster.data_transfer',
                'crocodile.cluster.distribute',
                'crocodile.cluster.meta_handling',
                'crocodile.cluster.remote_machine',
                'crocodile.cluster.script_execution',
                'crocodile.cluster.script_notify_upon_completion',
                'crocodile.cluster.trial_file',
                'crocodile.comms.gdrive',
                'crocodile.comms.helper_funcs',
                'crocodile.comms.notification',
                'crocodile.comms.onedrive',
                'crocodile.msc.ascii_art',
                'crocodile.msc.dl_template',
                'crocodile.msc.numerical',
                'crocodile.msc.odds']
    ,
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
        # "scipy",  # heavy-weight.
        "pandas",
        "matplotlib",
        "tabulate",  # pretty printing of tables

        # Accessories
        # h5py
        "joblib",  # multitasking
        "ipython",  # interactive python
        "fire",  # for automatic CLI interface
        "tqdm",  # for progress bar
        "dill",  # extends pickle
        "cryptography",  # for encoding
        "paramiko",  # for SSH
        "requests",  # interacting with web
        # "pyyaml",  # storing yaml files.
        "cycler",

        # Developer Tools
        "setuptools",  # for packaging  # correct place is in .toml
        "wheel",
        "twine",  # for pushing package to pypi.org
        "pytest",
        # popular alternative to builtint unittest  # consider splitting requirements to development and production versions.
        # "scikit-image",  # image processing. Heavy-weight.

        # torch
        # tensorflow
    ]

)

with open("./myresources/crocodile/art/happy_croco", "r") as file:
    croco = file.read()  # search ascii art or characters art.
print(croco)
