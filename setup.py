
"""
PKG installer

"""

from setuptools import setup
from myresources.crocodile import __version__
# import setuptools
import platform


with open("README.md", "r", encoding="utf-8") as file:
    long_desc = file.read()  # to help adding README to PyPi website not only Github

# get python modules:
# P("./myresources").search("*.py", r=True).apply(lambda x: x.split(at="myresources", sep=-1)[1].as_posix().replace("/", ".")[:-3]).filter(lambda x: "__init__" not in x).list

install_requires = [
        # CORE:
        "numpy",  # number crunching
        "pandas",  # number crunching
        "tabulate",  # pandas optional for pretty printing of tables
        "matplotlib",  # viz

        # the following is required to serialize dataframes
        "pyarrow",
        "fastparquet",

        # Accessories
        "rich",  # for rich text
        "tabulate",  # for pretty printing (required by rich to print tables)
        "randomname",  # for generating random names
        "psutil",  # monitor processes.
        "joblib",  # multitasking
        "ipython",  # interactive python
        "fire",  # for automatic CLI interface
        "tqdm",  # for progress bar
        "tomli",  # for TOML config files
        "pyyaml",  # for YAML config files
        "pyjson5", # for JSON files
        "dill",  # extends pickle
        "cryptography",  # for encoding
        "paramiko",  # for SSH
        "requests",  # interacting with web
        "colorlog",  # for colored logging
        "sqlalchemy",  # for database

        "bcrypt",  # for hashing
        "distro",  # for getting OS info
        "send2trash",  # for moving files to trash
        "py7zr",  # for 7z files
        "gitpython",  # for git
        "clipboard",  # for clipboard
    ]


if platform.system() == "Windows":
    # this is crucial for windows to pop up the concent window in case python was not run as admin.
    install_requires.append("pywin32")


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
                'crocodile.deeplearning_template',
                'crocodile.deeplearning_df',
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
                'crocodile.comms.helper_funcs',
                'crocodile.comms.notification',
                'crocodile.comms.onedrive',
                'crocodile.msc.ascii_art',
                'crocodile.msc.dl_template',
                'crocodile.msc.numerical',
                'crocodile.msc.odds'
                ],
    url='https://github.com/thisismygitrepo/crocodile',
    project_urls={
        "Bug Tracker": "https://github.com/thisismygitrepo/crocodile/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  # 3.9
    license='Apache 2.0',
    author='Alex Al-Saffar',
    author_email='programmer@usa.com',
    description='Deep Learning Framework & Workload Management For On-premise Personal Machines.',
    long_description=long_desc,
    long_description_content_type="text/markdown",
    package_data={'crocodile': ['art/*']},
    # include_package_data=True,
    install_requires=install_requires,

    extras_require={
            'full': [
                'tensorflow',
                'torch',
                "keras",
                     'scikit-learn',
                       # 'dash', 'dash_daq', 'dash_bootstrap_components',
                     'click',
                     'types-requests', 'types-paramiko', 'types-tqdm',
                     'setuptools', 'wheel', 'twine']
                },

)

with open("./myresources/crocodile/art/happy_croco", "r", encoding="utf-8") as file:
    croco = file.read()  # search ascii art or characters art.
print(croco)
