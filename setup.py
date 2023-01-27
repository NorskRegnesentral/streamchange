import setuptools

setuptools.setup(
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
    packages=setuptools.find_packages(),
    install_requires=["pandas", "numpy", "scipy", "river", "numba", "plotly"],
    license="BSD-3",
)
