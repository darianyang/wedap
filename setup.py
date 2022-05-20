from setuptools import setup, find_packages

with open('./README.md') as f:
    readme = f.read()

with open('./LICENSE') as f:
    lic = f.read()

packages = find_packages()

setup(
    name='wedap',
    version='0.0.1',
    description='weighted ensemble data analysis and plotting',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Darian T. Yang',
    author_email='dty7@pitt.edu',
    #install_requires=['numpy', 'matplotlib', 'h5py', 'scipy', 'moviepy', 'tqdm', 'gooey'],
    install_requires=['numpy', 'matplotlib', 'h5py', 'scipy', 'moviepy', 'tqdm'],
    url='https://github.com/darianyang/wedap',
    project_urls={'Documentation' : 'https://darianyang.github.io/wedap'},
    license=lic,
    #packages = find_packages(where = 'src'),
    #Packages=find_packages(exclude="docs"),
    packages=find_packages(exclude="docs"),
    package_data={"wedap": ["styles/*"]},
    #py_modules=["wedap"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry"
    ],
    entry_points={"console_scripts" : ["wedap=wedap.__main__:main"]}
)
