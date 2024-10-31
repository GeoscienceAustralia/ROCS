# Module for setting up the configurations

import logging
import yaml
import collections.abc


logger = logging.getLogger(__name__)

# function to recursively update a nested dictionary with another dictionary
def update(d, u):
    """
    Recursively update a nested dictionary with another dictionary

    Keyword arguments:
        d [dict] : dictionary to be updated
        u [dict] : dictionary used to update the items in d

    Returns:
        d [dict] : updated dictionary d

    """
    if u is not None:
        for k, v in u.items():
            if isinstance(d, collections.abc.Mapping):
                if isinstance(v, collections.abc.Mapping):
                    r = update(d.get(k, {}), v)
                    d[k] = r
                else:
                    d[k] = u[k]
            else:
                d = {k: u[k]}
    return d


class Config:

    def __init__(self,config_yaml=None):
        """
        Setup the configurations for combination

        Keyword arguments:
            config_yaml [str], optional  : YAML configuration file

        Updates:
            self.config [dict]           : configurations for combination
        """
        # Set the defaults
        config = {
                'process': {
                    'verbose': 'INFO'
                    },
                'campaign': {
                    'author': '',
                    'contact': '',
                    'sol_id' : 'FIN',
                    'camp_id': 'TST',
                    'cmb_name': 'IGS',
                    'vid': 0,
                    'cut_start': 0,
                    'cut_end': 0,
                    'subm_rootdir': './ac_subm',
                    'prod_rootdir': './products',
                    'sat_metadata_file': None,
                    'eop_format': None,
                    'eop_file': None,
                    'rf_rootdir': './sinex',
                    'rf_name': 'IGS0OPSSNX',
                    'nanu_sumfile':'./metadata/nanus_sum.2024',
                    'ac_acronyms': {}
                    },
                'orbits': {
                    'ac_contribs': {
                        'weighted': {
                            'systems': None,
                            'prns': None,
                            'svns': None
                            },
                        'unweighted': {
                            'systems': None,
                            'prns': None,
                            'svns': None
                            },
                        'excluded': {
                            'systems': None,
                            'prns': None,
                            'svns': None
                            }
                        },
                    'sampling': None,
                    'cen_wht_method': 'global',
                    'sat_wht_method': 'RMS_L1',
                    'rf_align': [False,False,False],
                    'ut1_rot': None,
                    'ut1_eop_format' : 'IGS_ERP2',
                    'rm_dv':False,
                    'no_rm_dv':[],
                    'assess': {
                        'sat_rms_tst': None,
                        'sat_rms_tst_unweighted': None,
                        'coef_sat': 470.0,
                        'thresh_sat': None,
                        'max_high_satrms': 5,
                        'trn_tst': None,
                        'thresh_trn': [None,None,None],
                        'numcen_tst': None,
                        'min_numcen': None,
                        'max_iter' : 100
                        },
                    'sp3_header': {
                        'coord_sys': 'IGS20',
                        'cmb_type': 'FINAL',
                        'clk_src': 'CMB',
                        'antex': 'IGS20',
                        'oload': 'FES2014b'
                        }
                    },
                'clocks': {
                    'ac_contribs': {
                        'weighted': {
                            'systems': None,
                            'prns': None,
                            'svns': None
                            },
                        'unweighted': {
                            'systems': None,
                            'prns': None,
                            'svns': None
                            },
                        'excluded': {
                            'systems': None,
                            'prns': None,
                            'svns': None
                            }
                        }
                    }
                }

        # If a configuration yaml file is given, update the configurations
        if config_yaml is not None:

            # Check the given config_yaml filename
            if not isinstance(config_yaml,str):
                logger.error("The given yaml config file must be a string",
                                stack_info=True)
                raise TypeError("The given yaml config file name must be a "
                                "string")

            # Try to open the yaml file
            try:
                with open(config_yaml, 'r') as stream:
                    yaml_parsed = yaml.load(stream, Loader=yaml.SafeLoader)
            except IOError:
                logger.error(f"The configuratoin yaml file {config_yaml} is"
                            f" not accessible!",stack_info=True)
                raise IOError(f"File {config_yaml} not accessible!")
            except yaml.YAMLError as e:
                logger.error(f"Error loading YAML file {config_yaml}: {e}", stack_info=True)
                raise

            # update the configurations
            config = update(config,yaml_parsed)

        # Update the attribute
        self.config = config
