import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Unfairness optimiser", # Replace with your own username
    version="0.9.0",
    author="Emil Dudev, Nils Lehmann, Sietze Kuilman, Thomas van Zwol",
    author_email="emil.dudev@student.uva.nl, nils.lehmann@student.uva.nl, "
                 "Skkuilman@gmail.com, T.j.vanzwol@gmail.com",
    description="An algorithm to reduce unfairness in rankings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TvanZ/FACT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.18.1,<1.23',
        'pandas>=0.25.3,<0.26',
        'PuLP>=2.0,<2.1',
    ],
)
