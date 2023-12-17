from abc import abstractmethod
from typing import Dict, List

from config.config_parser import ConfigParser


class Notify:

    def __init__(self, cfg: ConfigParser):
        super().__init__()
        self._email_config = cfg['email'] if 'email' in cfg else None
        self._sms_config = cfg['sms'] if 'sms' in cfg else None

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
            if isinstance(key, str) and isinstance(value, str):
                original_text = original_text.replace(key, value)
        return original_text

