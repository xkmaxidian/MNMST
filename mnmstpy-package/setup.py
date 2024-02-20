from setuptools import Command, find_packages, setup

__lib_name__ = "mnmstpy"
__lib_version__ = "1.1.0"
__description__ = "MNMST: Topology of cell networks leverages identification of spatial domains from spatial transcriptomics data"
__url__ = "https://github.com/xkmaxidian/MNMST"
__author__ = "Yu Wang"
__author_email__ = "qq352542417@gmail.com"
__license__ = "MIT"
__keywords__ = ["Spatial transcriptomics", "Spatial domain", "Network model", "Joint learning", "Topological structure",
                "Integrative analysis"]
__requires__ = [
    "scanpy>=1.9.3",
    "numpy>=1.24.0",
    "pandas>=1.5.0",
    "anndata>=0.9.0",
    "matplotlib>=3.7.1",
    "scipy>=1.11.2",
    "scikit-learn>=1.2.2",
    "psutil>=5.9.0",
    "tqdm>=4.65.0",
    "leidenalg>=0.10.1",
    "torch>=2.0.1",
    "torchvision>=0.15.2",
]

setup(
    name=__lib_name__,
    version=__lib_version__,
    description=__description__,
    url=__url__,
    author=__author__,
    author_email=__author_email__,
    license=__license__,
    packages=find_packages(),
    install_requires=__requires__,
    zip_safe=False,
    include_package_data=True,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]

)
