from setuptools import setup

setup(
    name="mlflow_dai_flavor",
    version="0.1",
    description="DAI Flavor",
    author="H2O.ai",
    packages=["h2o_mlflow_flavors"],
    install_requires=["mlflow"],
    # Require 3.9 before Apple M1 has trouble installing numpy for versions <= 3.8
    python_requires=">=3.8",
    zip_safe=False,
)