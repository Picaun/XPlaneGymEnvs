from setuptools import setup, find_packages

setup(
    name="XPlaneGym",
    version="0.1.0",
    description="OpenAI Gym compatible X-Plane reinforcement learning environment",
    author="Picaun",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.28.0",
        "numpy>=1.20.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Framework :: Robot Framework :: Library",
    ],
    python_requires=">=3.8",
) 