from abc import abstractmethod
from typing import Dict, List

from config.config_parser import ConfigParser


class Notify:

    def __init__(self, cfg: ConfigParser):
        super().__init__()
        self._email_config = cfg['notification']['email'] if 'notification' in cfg and 'email' in cfg['notification'] else None
        self._sms_config = cfg['notification']['sms'] if 'notification' in cfg and 'sms' in cfg['notification'] else None

    @property
    def email_config(self) -> Dict:
        return self._email_config

    @property
    def sms_config(self) -> Dict:
        return self._sms_config

    @abstractmethod
    def send(self, message: str, receivers: List[str]) -> bool:
        pass

    @staticmethod
    def replace_placeholder(original_text: str, placeholders: Dict) -> str:
        if placeholders is None:
            return original_text
        for key, value in placeholders.items():
            if isinstance(key, str):
                if not isinstance(value, str):
                    value = str(value)
                original_text = original_text.replace(key, value)
        return original_text

