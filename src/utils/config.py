import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

# Load Configurations
def load_config(config_name: str = "config.yaml") -> dict:

    root = get_project_root()
    file_path = root / config_name

    if not file_path.exists():
        logger.error(f"Configuration file not found at: {file_path}")
        raise FileNotFoundError(f"Expected config.yaml at {file_path.absolute()}")

    try:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Configuration loaded successfully from {file_path}")
        
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        raise