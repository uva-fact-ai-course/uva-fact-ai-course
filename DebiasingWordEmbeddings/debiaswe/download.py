""" This file is used to download different word embeddings.

DISCLAIMER: This code is largely copied from this stack overflow answer:
https://stackoverflow.com/a/39225272

The following word embeddings are available:
- word2vec_large
- word2vec_small
- glove_small
- fasttext_large
- fasttext_small

TODO: Embedding descriptions (e.g. embedding dimensions, training data)

"""

import copy
import os

import requests
from tqdm import tqdm

from .embeddings_config import ID


def download(embedding):
    """
    Downloads and saves embedding file.


    :param string embedding: Name of the desired embedding.
    :returns: None
    """
    assert embedding in ID.keys(), "Unknown embedding."

    URL = "https://docs.google.com/uc?export=download"
    id = ID[embedding]["id"]
    extension = ID[embedding]["extension"]

    # Destination is current file destination, one directory up, then the
    # "embeddings" directory.
    destination = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "embeddings", embedding + extension
    )
    print(f"Downloading {embedding} to {os.path.abspath(destination)}")

    session = requests.Session()
    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    """
    Filters response for confirm token.


    :param Response response: Response object to filter through.
    :returns: token or None
    """
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    """
    Saves downloaded content.


    :param Response respinse: Response object to extract content from.
    :param string destination: Location to save the content to.
    :returns: None
    """
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        download_size = len([1 for _ in copy.copy(response).iter_content(CHUNK_SIZE)])
        response = response.iter_content(CHUNK_SIZE)
        for _ in tqdm(range(download_size)):
            chunk = next(response)
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
