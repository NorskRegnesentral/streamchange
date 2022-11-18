from setuptools import setup

setup(
    name="streamchange",
    version="0.1",
    author="Martin Tveten",
    author_email="tveten@nr.no",
    description=(
        "A package for segmenting streaming time series data into homogenous segments."
        " The segmentation is based on statistical change-point detection (aka"
        " online/sequential/iterative change-point detection)."
    ),
    long_description="",
    long_description_content_type="text/markdown",
    packages=["streamchange"],
    install_requires=["pandas", "numpy", "numba", "scipy", "copy", "plotly"],
)