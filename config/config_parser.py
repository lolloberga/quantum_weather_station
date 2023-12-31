import yaml
from typing import Any, Dict

import os

ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
).replace('/config', '')


class ConfigParser:

    def __init__(self, config_path=None):
        self._consts = dict()
        if config_path is None:
            config_path = os.path.join(ROOT_DIR, 'resources', 'configurations.yml')
        with open(config_path, 'r') as yamlfile:
            self._config = yaml.unsafe_load(yamlfile)
            self._define_constants()

    def __getitem__(self, name) -> Any:
        return self._config[name]

    def __contains__(self, item) -> bool:
        return item in self._config

    @property
    def consts(self) -> Dict:
        return self._consts

    @property
    def is_model_notification_configured(self) -> bool:
        return ((self.consts['MODEL_EMAIL_NOTIFY_ENABLE'] is not None or self.consts['MODEL_SMS_NOTIFY_ENABLE'])
                and self.consts['MODEL_NOTIFICATIONS'] is not None)

    def _define_constants(self) -> None:
        self._consts['MAIN_PATH'] = ROOT_DIR
        self._consts['RESOURCE_PATH'] = os.path.join(ROOT_DIR, self._config['resouces']['path']) if \
            self._config['resouces']['path'] is not None else os.path.join(ROOT_DIR, "resources")
        self._consts['DATASET_PATH'] = os.path.join(ROOT_DIR, self._config['dataset']['path']) if \
            self._config['dataset']['path'] is not None else os.path.join(self._consts['RESOURCE_PATH'], "dataset")
        self._consts['FOLDER_NAME_PREFIX'] = self._config['dataset']['folder_name_prefix'] if self._config['dataset'][
                                                                                                 'folder_name_prefix'] is not None else 'board_'
        self._consts['SENSOR_NAME_PREFIX'] = self._config['dataset']['sensor_name_prefix'] if self._config['dataset'][
                                                                                                 'sensor_name_prefix'] is not None else 's'

        self._consts['BOARD_CONFIG_FILE'] = os.path.join(self._consts['RESOURCE_PATH'],
                                                         self._config['dataset']['board_config_name']) if \
            self._config['dataset']['board_config_name'] is not None else os.path.join(self._consts['RESOURCE_PATH'],
                                                                                      'board.json')

        self._consts['DOWNLOAD_DATASET_MAP'] = os.path.join(ROOT_DIR,
                                                            self._config['dataset']['download_repositories']) if \
            self._config['dataset']['download_repositories'] is not None else os.path.join(self._consts['RESOURCE_PATH'],
                                                                                          "download_dataset_map.json")

        self._consts['UPLOAD_CHECKPOINT_PATH'] = os.path.join(ROOT_DIR, self._config['dataset']['checkpoint_path']) if \
            self._config['dataset']['checkpoint_path'] is not None else self._consts['RESOURCE_PATH']

        self._consts['MODEL_DRAWS_PATH'] = os.path.join(ROOT_DIR, self._config['model']['draws_path']) if \
            self._config['model']['draws_path'] is not None else os.path.join(ROOT_DIR, "../model", "draws")
        self._consts['MODEL_CHECKPOINT_PATH'] = os.path.join(ROOT_DIR, self._config['model']['checkpoint_path']) if \
            self._config['model']['checkpoint_path'] is not None else os.path.join(ROOT_DIR, "../model", "checkpoints")
        self._consts['MODEL_NOTIFICATIONS'] = self._config['model']['notifications'] \
            if 'notifications' in self._config['model'] and self._config['model']['notifications'] is not None else None
        self._consts['MODEL_EMAIL_NOTIFY_ENABLE'] = self._config['model']['email_notification'] \
            if 'email_notification' in self._config['model'] and self._config['model']['email_notification'] is not None else False
        self._consts['MODEL_EMAIL_RECEIVERS'] = self._config['model']['email_receivers'] \
            if 'email_receivers' in self._config['model'] and self._config['model']['email_receivers'] is not None else None
        self._consts['MODEL_SMS_NOTIFY_ENABLE'] = self._config['model']['sms_notification'] \
            if 'sms_notification' in self._config['model'] and self._config['model']['sms_notification'] is not None else False
        self._consts['MODEL_SMS_RECEIVERS'] = self._config['model']['sms_receivers'] \
            if 'sms_receivers' in self._config['model'] and self._config['model']['sms_receivers'] is not None else None
