from setuptools import setup, find_packages

setup(
    name="mujoco_lidar",
    version="0.1.0",
    author="Yufei Jia",
    author_email="jyf23@mails.tsinghua.edu.cn",
    description="A high-performance LiDAR simulation tool designed for MuJoCo, powered by Taichi programming language",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TATP-233/MuJoCo-LiDAR.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=open("requirements.txt", encoding="utf-8").read().splitlines(),
)
