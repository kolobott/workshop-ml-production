import logging
import os
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def download_file_url(url, filepath):
    resp = requests.get(url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


def download_and_unzip(url, extract_to):
    """
    Downloads a zip file from the specified URL and extracts it to the given folder.

    Parameters:
        url (str): The URL of the zip file to download.
        extract_to (str): The directory path where the zip file should be extracted.

    Returns:
        None
    """
    # Check if the directory exists, if not, create it
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    zipf = os.path.join("/tmp", str(uuid.uuid4()) + ".zip")
    try:
        logging.info("Downloading data")
        download_file_url(url, zipf)

        with zipfile.ZipFile(zipf, 'r') as file:
            file.extractall(extract_to)
    except Exception as e:
        raise e
    finally:
        if os.path.exists(zipf):
            os.remove(zipf)
