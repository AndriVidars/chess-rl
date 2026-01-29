from setuptools import setup, find_packages

setup(
    name="chess_rl",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "python-chess",
        "stockfish",
        "matplotlib",
        "tqdm"
    ]
)
