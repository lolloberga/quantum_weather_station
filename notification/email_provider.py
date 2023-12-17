import os
import smtplib
import ssl
from typing import List, Dict

from config.config_parser import ConfigParser
from notification.base.notify import Notify


class EmailProvider(Notify):

    def __init__(self, cfg: ConfigParser, placeholders: Dict = None):
        super().__init__(cfg)

        self._config = self.email_config
        if self._config is None:
            raise ValueError('No email configuration found in yaml file')
        self._address = self._config['address']
        self._port = int(self._config['port'])
        self._use_ssl = self._config['use_ssl'] if 'use_ssl' in self._config else True
        self._placeholders = placeholders

        # Get credentials
        self._username = self._config['username'] if 'username' in self._config else None
        self._password = self._config['password'] if 'password' in self._config else None
        if ('get_credentials_from_file' in self._config and self._config['get_credentials_from_file']
                and self._username is not None and self._password is not None):
            username_file = os.path.join(cfg.consts['MAIN_PATH'], self._username)
            password_file = os.path.join(cfg.consts['MAIN_PATH'], self._password)
            with open(username_file, 'r') as f:
                self._username = f.readline().strip('\n')
            with open(password_file, 'r') as f:
                self._password = f.readline().strip('\n')

    def send(self, message: str, receivers: List[str]) -> bool:
        if len(receivers) <= 0:
            raise ValueError('No receivers set to send the notification')

        if self._use_ssl:
            try:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(self._address, self._port, context=context) as server:
                    if self._username is not None and self._password is not None:
                        server.login(self._username, self._password)

                    for receiver in receivers:
                        self._placeholders['#RECEIVER#'] = receiver
                        message = Notify.replace_placeholder(message, self._placeholders)
                        server.sendmail(self._username, receiver, message)

            except smtplib.SMTPException:
                return False
            return True

        return False
