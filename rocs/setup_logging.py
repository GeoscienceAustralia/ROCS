import os
import logging.config
import yaml

def setup_logging(
    default_path=None,
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration
    """

    # If no default path is provided, use project-relative logging.yaml
    if not default_path:
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of setup_logging.py
        project_dir = os.path.abspath(os.path.join(current_dir, '..'))  # Go one level up to project root
        default_path = os.path.join(project_dir, 'logging.yaml')  # Assuming logging.yaml is in the project root

    # Check for the environment variable or use the default path
    path = os.getenv(env_key, default_path)

    # Try to load the logging config
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
