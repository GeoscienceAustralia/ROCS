from setuptools import setup, find_packages

# Read the contents of your README file (optional)
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("LICENSE", "r", encoding="utf-8") as fh:
    license = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements_lines = fh.read().splitlines()
requirements = []
for item in requirements_lines:
    requirements.append(item)

setup(
    name="rocs",
    version="1.0",
    license=license,
    author="Geoscience Australia",
    author_email="GNSSAnalysis@ga.gov.au",
    description="Orbit combination software",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/your-package",
    packages=find_packages(),  # Automatically find all packages
    classifiers=[
        "Programming Language :: Python :: 3",
        f"License :: OSI Approved :: {license}",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Python version compatibility
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rocs =rocs.__main__:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml"],
    },
)

