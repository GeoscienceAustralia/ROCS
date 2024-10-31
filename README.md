# ROCS: Robust Orbit Combination Software
#### ROCS version 1.0

## Overview

The *Robust Orbit Combination Software - ROCS* is a GNSS processing package developed in Python 3 to combine orbits (and in future clocks) from different International GNSS Service (IGS) analysis centres [https://igs.org/acc/]. The algorithm used by the software is mainly based on the legacy version of the IGS Analysis Centre Coordinator (ACC) combination software which was developed in Fortran with Perl and Shell Scripts by the Astronomical Institute, University of Bern, Delft University of Technology, and Geodetic Survey of Canada, NRCan (see Beutler, Kouba and Springer: Combining the orbits of the IGS Analysis Centers; Bulletin Geodesique; 1995).

This modern implementation of the combination software enables the combination of multi-GNSS orbits, and incorporates more complex techniques for weighting of the Analysis Centre solutions for different constellations/blocks/satellites. For details of the processing algorithms, please refer to: Zajdel, R., Masoumi, S., Sośnica, K. et al. Combination and SLR validation of IGS Repro3 orbits for ITRF2020. J Geod 97, 87 (2023). https://doi.org/10.1007/s00190-023-01777-3.


## How to run
Most of the configurations can be controlled by a YAML file. Examples of config.yaml files are provided in the ```inp/``` directory. 

## Directory structure

The tree structure of the modules is as below:

```
+-- rocs
|   +-- __main__.py --> main module executed when package is run
|   +-- combine_orbits.py --> main module to execute orbit combination
|   +-- orbits.py --> orbit comibnation module
|   +-- helmert.py --> forward and inverse helmert transformation
|   +-- checkutils.py --> consistency checks for sizes and types
|   +-- coordinates.py --> coordinate transformations
|   +-- eclipse.py --> satellite eclipse calculations
|   +-- formatters.py --> cutsom formatters for logging
|   +-- gpscal.py --> time conversions
|   +-- iau.py --> International Astronomical Union models
|   +-- io_data.py --> input/output of data files
|   +-- planets.py --> planetary position calculations
|   +-- report.py --> reporting and creating summary files
|   +-- rotation.py --> calculation of rotation matrix and its derivative
|   +-- settings.py --> settings and configurations
|   +-- setup_logging.py --> setup logging information
+-- inp --> directory including examples of configuration YAML files
+-- README.md --> this readme file
+-- requirements.txt --> list of software dependencies
+-- setup.py --> package and distribution management
+-- logging.yaml --> YAML file for logging
+-- LICENSE --> software license
```

## How to install
To install the software in Linux, simply navigate to where you would like to install the software on your machine, and clone the repository:

```
git clone https://github.com/yourusername/rocs.git
cd ROCS
```

Then install the required packages:

```
pip3 install -r requirements.txt
```

To be able to run the software from anywhere on your system, add the package to your ```$PYTHONPATH```:

```
export PYTHONPATH=/path/to/your/ROCS/:$PYTHONPATH
```

It is recommended that you add the above line to your shell initialization file (```~/.bashrc```, ```~/.cshrc```, etc.) so you do not need to run this every time you start up the sell.


## How to run

After installing the software, you need to prepare your campaign before running the combination. To do this, follow the below steps:

- Create a configuration YAML file. You can choose the name and location of this file whatever you would like as you will need to point to it in the command that you will run. Examples of the configuration YAML file are given in the ```inp``` directory. Use one of these examples, and modify it to suit your preferences and environment (i.e. directory and file locations).
- Create a directory for the submissions, and copy all the individual (Analysis Centre) solutions there. The path to the directory should be indicated in the configuration YAML file created above in the ```subm_rootdir``` entry.
- Ensure the metadata and other required files indicated in the configuration YAML file exist in the indicated locations. These may include, e.g., satellite metadata file, EOP file, and NANU summary file
- Ensure that the ```prod_rootdir``` entry given in the configuration file exists and is where you would like the products to be written to. Previous file in that location may be overwritten by the output files from running the software.

Once you have prepared the campaign, you can run the program by simply executing the command below:

```
python3 -m rocs <WWWW> <D> <H> -c </path/to/your/config.yaml>
```

where ```<WWWW>``` is GPS week, ```<D>``` is day of week (for the starting epoch of the combination), ```<H>``` is hour of the day (optional, only required for ultra-rapid combination), and ```</path/to/your/config.yaml>``` is the full path to your configuration YAML file.

In the ```examples``` folder, there are two different scenario examples, one for final orbits and one for ultra-rapid orbits, along with the required input data, as well as the expected outputs in the ```products``` folder. You may want to use these examples to ensure you retrieve the expected outcomes.

To run the example of a final orbit combination, run the below commands:

```
cd examples/final/
python3 -m rocs 2332 4 -c ./inp/config_example_final.yaml
```

Similarly, to run the ultra-rapid example:

```
cd examples/ultra/
python3 -m rocs 2332 4 18 -c ./inp/config_example_ultra.yaml
```

The combined orbits and summary files in the ```products``` directory of the above examples, should be identical to the pre-combined files in the ```products_expected``` directory.
Note the above are just example scenarios, and that you may want to change the configurations based on your particular needs.

Running the package with the ```--help``` argument will show a simple help option:

```
python3 -m rocs --help

usage: rocs [-h] [-c CONFIG_YAML] gpsweek dow [hr]

Robust combination of orbit solutions

positional arguments:
  gpsweek               GPS week for processing
  dow                   Day of week for processing
  hr                    Starting hour for ultr-rapid combination (default:
                        None)

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_YAML, --config CONFIG_YAML
                        YAML file containing configurations for combination
                        (default: None)
```
