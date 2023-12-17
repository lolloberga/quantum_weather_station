import smtplib
import ssl
from typing import List, Dict

from config.config_parser import ConfigParser
from notification.base.notify import Notify


class GmailProvider(Notify):

    def __init__(self, cfg: ConfigParser, placeholders: Dict = None):
        super().__init__(cfg)

        self._config = self.email_config['gmail']
        if self._config is None:
            raise ValueError('No gmail configuration found in yaml file')
        self._address = self._config['address']
        self._port = int(self._config['port'])
        self._use_ssl = self._config['use_ssl']
        self._username = self._config['username']
        self._password = self._config['password']
        self._placeholders = placeholders

    def send(self, message: str, receivers: List[str]) -> bool:
        if len(receivers) <= 0:
            raise ValueError('No receivers set to send the notification')

        if self._placeholders is not None:
            self._placeholders['#RECEIVER#'] = receivers[0]

        message = Notify.replace_placeholder(message, self._placeholders)

        if self._use_ssl:
            try:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(self._address, self._port, context=context) as server:
                    server.login(self._username, self._password)
                    server.sendmail(self._username, receivers, message)
            except smtplib.SMTPException:
                return False
            return True

        return False
