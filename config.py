import yaml
from typing import Any, Dict

import os
ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

class ConfigParser:

    def __init__(self, config_path=None):
        if config_path == None:
            config_path = os.path.join(ROOT_DIR, 'resources', 'configurations.yml')
        with open(config_path, 'r') as yamlfile:
            self._config = yaml.unsafe_load(yamlfile)
            self._define_constants()
    
    def __getitem__(self, name) -> Any:
        return self.config[name]
    
    @property
    def config(self) -> Dict:
        return self._config
    
    @property
    def consts(self) -> Dict:
        return self._consts
    
    def _define_constants(self) -> None:
        self._consts = dict()

        self._consts['RESOURCE_PATH'] = os.path.join(ROOT_DIR, self.config['resouces']['path']) if self.config['resouces']['path'] is not None else os.path.join(ROOT_DIR, "resources")
        self._consts['DATASET_PATH'] = os.path.join(ROOT_DIR, self.config['dataset']['path'])
        self._consts['FOLDER_NAME_PREFIX'] = self.config['dataset']['folder_name_prefix'] if self.config['dataset']['folder_name_prefix'] is not None else 'board_'
        self._consts['SENSOR_NAME_PREFIX'] = self.config['dataset']['sensor_name_prefix'] if self.config['dataset']['sensor_name_prefix'] is not None else 's'

        self._consts['BOARD_CONFIG_FILE'] = os.path.join(self._consts['RESOURCE_PATH'], self.config['dataset']['board_config_name']) if self.config['dataset']['board_config_name'] is not None else os.path.join(self._consts['RESOURCE_PATH'], 'board.json')

        self._consts['DOWNLOAD_DATASET_MAP'] = os.path.join(ROOT_DIR, self.config['dataset']['download_repositories']) if self.config['dataset']['download_repositories'] is not None else os.path.join(self._consts['RESOURCE_PATH'], "download_dataset_map.json")

        self._consts['UPLOAD_CHECKPOINT_PATH'] = os.path.join(ROOT_DIR, self.config['dataset']['checkpoint_path']) if self.config['dataset']['checkpoint_path'] is not None else self._consts['RESOURCE_PATH']