from setuptools import setup, find_packages

setup(
    name="slm-head-pruning",
    version="1.0.0",
    description="Layer-wise attention head redundancy and sensitivity geometry in sub-billion language models",
    author="Md. Mushfiqur Rahim",
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "transformers>=4.40",
        "datasets>=2.19",
        "pandas>=2.0",
        "numpy>=1.26",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "scipy>=1.11",
        "pyyaml>=6.0",
        "tqdm>=4.66",
    ],
)
