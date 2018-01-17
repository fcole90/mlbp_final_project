from setuptools import setup
setup(
    name="mlbp-final-project",
    packages=["mlbp_final_project"],
    version="0.0.1.dev1",
    description="MLBP Final Project",
    author="Anonimous",
    author_email="anon.mous@aalto.fi",
    url="https://www.kaggle.com/c/mlbp-2017-da-challenge-accuracy",
    download_url="https://www.kaggle.com/c/mlbp-2017-da-challenge-logloss",
    keywords=["mlbp", "final", "project"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research"
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3"
        ],
    long_description="""\
Final project of the course of Machine Learning: Basic Principles 2017.
""",
    install_requires=['numpy'],
    python_requires='>=3.4'
)
