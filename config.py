import yaml

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
    
    def __getitem__(self, name):
        return self.config[name]
    
    @property
    def config(self):
        return self._config