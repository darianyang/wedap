from setuptools import setup, find_packages

with open('./README.md') as f:
    readme = f.read()

with open('./LICENSE') as f:
    lic = f.read()

packages = find_packages()

setup(
    name='wedap',
    version='0.0.0',
    description='weighted ensemble data analysis and plotting',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Darian T. Yang',
    author_email='dty7@pitt.edu',
    install_requires=['numpy', 'matplotlib', 'h5py', 'scipy', 'moviepy', 'pandas', 'gooey'],
    url='https://github.com/darianyang/wedap',
    license=lic,
    #packages = find_packages(where = 'src'),
    Packages=find_packages(exclude="docs"),
    #package_data={"wedap": ["data/*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3 License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry"
    ],
    entry_points={"console_scripts" : ["wedap=wedap.wedap:main"],
                  "gui_scripts" : ["wedap-gui=wedap.wedap:main"]
                  }
)
