import platform
import sys
from pathlib import Path

import pkg_resources
from setuptools import find_packages, setup

requirements = []
if sys.platform.startswith("linux") and platform.machine() == "x86_64":
    requirements.append("triton>=2.0.0,<3")

setup(
    name="whisper-at",
    py_modules=["whisper_at"],
    version=0.5,
    description="Joint speech recognition and audio tagging model.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    author="Yuan Gong",
    url="https://github.com/YuanGongND/whisper-at",
    license="BSD",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            Path(__file__).with_name("requirements.txt").open()
        )
    ],
    entry_points={
        "console_scripts": ["whisper=whisper.transcribe:cli"],
    },
    include_package_data=True,
    extras_require={"dev": ["pytest", "scipy", "black", "flake8", "isort"]},
)
