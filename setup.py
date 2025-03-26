from setuptools import setup, find_packages

setup(
    name="german-energy-forecast-api",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-dotenv>=1.0.0",
        "pandas>=2.1.4",
        "numpy>=1.26.0",
        "entsoe-py>=0.5.10",
        "scikit-learn>=1.6.1",
        "lightgbm>=4.5.0",
    ]
)