from setuptools import setup, find_packages

setup(
    name="german-energy-forecast-api",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Core API
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-dotenv>=1.0.0",

        # Data Processing
        "pandas>=2.1.4",
        "numpy>=1.26.0",
        "entsoe-py>=0.5.10",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",  
        "tqdm>=4.65.0",

        # Machine Learning
        "scikit-learn>=1.6.1",
        "lightgbm>=4.5.0",

        # Testing
        "pytest>=8.3.0",
        "pytest-cov>=6.0.0",
        "pytest-mock>=3.10.0",
        "pytest-asyncio>=0.22.0",

    ]
)