import argparse
import logging
import time
from rocs.combine_orbits import combine_orbits
from rocs.setup_logging import setup_logging
import rocs.settings as settings


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(prog='rocs', description =
                        "Robust combination of orbit solutions",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('gpsweek',metavar='gpsweek',nargs=1,
                        help='GPS week for processing')
    parser.add_argument('dow',metavar='dow',nargs=1,
                        help='Day of week for processing')
    parser.add_argument('hr',metavar='hr',nargs='?',
                        help='Starting hour for ultr-rapid combination')
    parser.add_argument('-c','--config',nargs=1, dest = 'config_yaml',
                help = ('YAML file containing configurations for combination '))
    args = parser.parse_args()

    # GPS week, day of week and hour
    gpsweek = int(args.gpsweek[0])
    dow = int(args.dow[0])
    if args.hr is not None:
        hr = int(args.hr)
    else:
        hr = 0

    # Setup the configurations for combination by reading the yaml file
    if args.config_yaml is not None:
        config_yaml = str(args.config_yaml[0])
        config = settings.Config(config_yaml).config
    else:
        # if no config file specified, use the defaults
        config_yaml = None
        config = settings.Config().config

    # verbose mode
    verbose = config['process']['verbose']
    allowed_verbose = ['INFO','DEBUG']
    if verbose not in allowed_verbose:
        logger.error(f"\nVerbose mode must be one of {allowed_verbose}\n", stack_info=True)
        raise ValueError(f"Verbose mode {verbose} not recognized!")

    # Setup the logging using logging.yaml
    setup_logging()
    logger = logging.getLogger()
    logger.setLevel(verbose)

    logger.debug(f"\nthis is debug from main")
    logger.info(f"\nthis is info from main")

    # time measurements
    pc0 = time.perf_counter()
    pt0 = time.process_time()

    # call orbit combination
    combine_orbits(gpsweek,dow,hr,config)

    # time measurements
    pc1 = time.perf_counter() - pc0
    pt1 = time.process_time() - pt0
    logger.info(f"Performance counter spent: {pc1}")
    logger.info(f"Process time spent: {pt1}")



if __name__ == '__main__':
    main()


