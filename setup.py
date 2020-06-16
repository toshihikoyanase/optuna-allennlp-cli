from setuptools import setup, find_packages

try:
    long_description = open("README.rst", "rt").read()
except IOError:
    long_description = ""


PROJECT = "allenopt"

VERSION = "0.0.1"

setup(
    name=PROJECT,
    version=VERSION,
    description="Hyperparameter tuning for AllenNLP using Optuna.",
    long_description=long_description,
    author="Toshihiko Yanase",
    author_email="toshihiko.yanase@gmail.com",
    url="",
    download_url="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Developers",
        "Environment :: Console",
    ],
    platforms=["Any"],
    scripts=[],
    provides=[],
    install_requires=["allennlp", "cliff", "optuna"],
    namespace_packages=[],
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": ["allenopt = cli:main"],
        "allenopt.command": ["search = cli:Search",],
    },
    zip_safe=False,
)
