from config.config_parser import ConfigParser
import requests
import json
import shutil
from tqdm import tqdm

import os
ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

class DownloadDataset:

    def __init__(self, path:str = None) -> None:
        # get configurations
        cfg = ConfigParser()
        if path is None:
            path = cfg.consts['DATASET_PATH']
        self._default_path = path
        self._map = cfg.consts['DOWNLOAD_DATASET_MAP']

        if not os.path.exists(self._map) or not os.path.isfile(self._map):
            raise FileNotFoundError('No download map file found')
        if not os.path.exists(self._default_path) or os.path.isfile(self._default_path):
            raise FileNotFoundError('No downloads folder found')

    def download_with_map_file(self) -> None:
        with open(self._map) as f:
            map_file = json.load(f)
            for download_conf in map_file:
                if 'id' in download_conf and 'repository' in download_conf and isinstance(download_conf['repository'], dict) and download_conf['enable'] == True:
                    # create a folder if not exists
                    download_folder = os.path.join(self._default_path, download_conf['id'])
                    if not os.path.exists(download_folder):
                        os.makedirs(download_folder)
                    # loop for each repo
                    for repo_name in tqdm(download_conf['repository'].keys(), desc='Downloading dataset'):
                        file_path = os.path.join(download_folder, repo_name)
                        if not os.path.exists(file_path):
                            self._make_http_request(download_conf['repository'][repo_name], file_path)
    
    
    def _make_http_request(self, url:str, file_path:str) -> None:
        r = requests.get(url, verify=False, stream=True)
        r.raw.decode_content = True
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    
    def _download(self) -> None:
        # TBD
        pass