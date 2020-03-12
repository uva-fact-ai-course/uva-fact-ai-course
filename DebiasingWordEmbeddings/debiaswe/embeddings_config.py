''' Configuration file for downloading available pretrained word embeddings.

ID is a dictionary with word embeddings as keys, and dictionaries containing a
Google Drive download id and file extension as values.

Google Drive folder:
https://drive.google.com/drive/folders/1AY6IwIoJqepxw3s6wK-Fl6udMkVmMjFO

Note that the embedding names in de Google Drive folder do not exactly match
the embedding names used in the ID dictionary.
'''

ID = {
    # Word2Vec, trained on Google News
    # 300d embeddings 3M words
    "word2vec_large": {
        "id": "1ep-6TjdfG86EbdcypvrHg7oxnc_3aD02",
        "extension": ".bin"
    },
    # 300d embeddings, 26423 words
    "word2vec_small": {
        "id": "1Lk-jzberOG9F9lA0pnEyzlF9mGEyk8OZ",
        "extension": ".txt"
    },
    # 300d embeddings, 26423 words, hard debiased
    "word2vec_small_hard_debiased": {
        "id": "1B2DN7I-QVHi67qf04GG2bq5qnOAI0vVN",
        "extension": ".txt"
    },
    # 300d embeddings, 26423 words, soft debiased
    # Params: epochs=2000, lr=0.01, gamma=0.1, decrease_times=[1000,1500,1800]
    "word2vec_small_soft_debiased": {
        "id": "1JXHYPFRmpNvvYXvH7wUMYv5qhrgT-9-9",
        "extension": ".txt"
    },

    # GloVe, trained on Common Crawl (42B tokens)
    # 300d embeddings, 1.9M words
    "glove_large": {
        "id": "1H1Ed1hZBQPymQ3O3yOltOG-Pi3eOy-Ac",
        "extension": ".txt"
    },
    "glove_small": {
        "id": "1cZ5UG5LmjCM5vNczLHFnDb7zP7tgce0r",
        "extension": ".txt"
    },
    # 300d embeddings, 42982 words, hard debiased
    "glove_small_hard_debiased": {
        "id": "1mPVF3NXfNCRt3QrJ8ovti1GNKbQ-d06Z",
        "extension": ".txt"
    },
    # 300d embeddings, 42982 words, soft debiased
    # Params: epochs=3500, lr=0.01, gamma=0.1, decrease_times=[2300,2800,3000]
    "glove_small_soft_debiased": {
        "id": "1oqVcOpMskmbNG-Sh2iQamDf49Zk7yf1t",
        "extension": ".txt"
    },

    # fastText, trained Wikipedia 2017, UMBC webbase corpus and statmt.org news
    # 300d embeddings, 999994 words
    "fasttext_large": {
        "id": "1G23SP2D7qKISGNKH5HK6v9Su3wo-jOxH",
        "extension": ".vec"
    },
    # 300d embeddings, 27014 words
    "fasttext_small": {
        "id": "1rLnn9MtIHwvCcRo9JUbaDvEZb-maygX_",
        "extension": ".txt"
    },
    # 300d embeddings, 27014 words, hard debiased
    "fasttext_small_hard_debiased": {
        "id": "1RNaUBYLRb99LZUA4LxoenkEMPNBfC4Hl",
        "extension": ".txt"
    },
    # 300d embeddings, 27014 words, soft debiased
    # Params: epochs=7000, lr=0.01, gamma=0.1, decrease_times=[5000],
    "fasttext_small_soft_debiased": {
        "id": "1UwRmovvv_FNClZ6jArZZ2U3_6WOR0zHx",
        "extension": ".txt"
    }
}
