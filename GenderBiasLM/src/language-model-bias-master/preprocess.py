# coding: utf-8
import os
import multiprocessing as mp
import re
import ctypes
import argparse
import struct
import gzip
import spacy
import _pickle as pk
from unidecode import unidecode
import logging
from log import init_console_logger


LOGGER = logging.getLogger('preprocess')
LOGGER.setLevel(logging.DEBUG)

en = spacy.load('en_core_web_sm')
en.max_length = 10000000


def is_valid_token(w):
    """
    Returns True if a token is valid
    """
    return bool(re.search('[a-zA-Z0-9]+', w))


def transform_token(w):
    """
    Transforms a token by making lowercase, and for numeric tokens replaces
    digits with placeholders
    """
    return re.sub(r'[.\-\']+$', '',
      re.sub(r'[.\-\']+', '',
        re.sub(r'[^A-Za-z<>$.\-\']', '',
            re.sub(r'\d+', '<NUM>',
                unidecode(w).lower()))))


def preprocess_file(filepath, output_path):
    """
    Preprocesses a file by splitting it into sentences and tokenizing it
    """
    # Open file
    try:
        with open(filepath, 'r') as f:
            text = f.read()
    except UnicodeDecodeError as e:
        try:
            # Account for some files that may be encoded with ISO-8859-1
            with open(filepath, 'r', encoding='iso-8859-1') as f:
                text = f.read()
        except UnicodeDecodeError as e:
            msg = "Could not open {}: {}".format(filepath, str(e))
            raise Exception(msg)


    # Remove any additional information e.g. "@highlights"
    main_text_body = text.split('\n@')[0]

    # Split up lines, and then break up lines into sentences
    sentences = []
    for line in main_text_body.split('\n\n'):
        sentences += list(en(line.strip('\n')).sents)

    tmp_output_path = output_path[:-3] + 'tmp'

    if not os.path.exists(tmp_output_path):
        # Get tokens for each sentence
        tokens = set()
        sentence_tokens = []
        for sent in sentences:
            sent_tokens = []
            for w in sent:
                if not is_valid_token(w.text):
                    continue
                w = transform_token(w.text)
                sent_tokens.append(w)
                tokens.add(w)
            if len(sent_tokens) > 1:
                sentence_tokens.append(sent_tokens)

        with gzip.open(tmp_output_path, 'wb') as f:
            pk.dump(sentence_tokens, f)
    else:
        # If tmp file already exists, just load file to get tokens
        with gzip.open(tmp_output_path, 'rb') as f:
            try:
                sentence_tokens = pk.load(f)
            except EOFError:
                sentence_tokens = []

        tokens = set()
        for sent in sentence_tokens:
            tokens.update(set(sent))

    return tokens


def write_preprocessed_file(encoded_sentences, output_path):
    """
    Write encoded sentences to a binary file
    """
    num_sentences = len(encoded_sentences)
    offset = 0
    uint_size = ctypes.sizeof(ctypes.c_uint)

    # Get total file size
    total_size = sum([(len(sent)+1) * uint_size for sent in encoded_sentences]) - uint_size

    buf = ctypes.create_string_buffer(total_size)
    for sent_idx, sent in enumerate(encoded_sentences):
        # Encode words as unsigned ints
        for idx in sent:
            struct.pack_into('I', buf, offset, idx + 1)
            offset += uint_size

        # Add a sentence delimter
        if sent_idx != (num_sentences - 1):
            struct.pack_into('I', buf, offset, 0)
            offset += uint_size

    # Gzip and save
    with gzip.open(output_path, 'wb') as f:
        f.write(buf)


def encode_sentences(sentences, word_to_idx):
    """
    Encode tokens in sentences by vocab indices
    """
    return [[word_to_idx[w] for w in sent] for sent in sentences]


def read_preprocessed_file(filepath, vocab):
    """
    Reads a preprocessed text file. Returns a list of sentences, where
    each sentence is a list of tokens.
    """
    # Get binary string
    with gzip.open(filepath, 'rb') as f:
        buf = f.read()

    sentences = []
    sent = []
    for (val,) in struct.iter_unpack('I', buf):
        if val > 0:
            # Get words for the current sentence
            sent.append(vocab[val-1])
        else:
            # We've reached the end of the sentence
            sentences.append(sent)
            sent = []

    return sentences


def read_preprocessed_file_as_str(filepath, vocab, sent_delim='<eos>'):
    """
    Reads a preprocessed text file. Returns a list of sentences, where
    each sentence is a list of tokens.
    """
    # Get binary string
    with gzip.open(filepath, 'rb') as f:
        buf = f.read()

    res = ""
    first_word = True
    for (val,) in struct.iter_unpack('I', buf):
        if not first_word:
            res += ' '
        else:
            first_word = False

        if val > 0:

            # Get words for the current sentence
            res += vocab[val-1]
        else:
            # We've reached the end of the sentence
            res += sent_delim
    res += ' ' + sent_delim

    return res


def load_preprocesed_dataset(pp_dataset_dir, sent_delim='<eos>', vocab_path=None):
    if not vocab_path:
        vocab_path = os.path.join(pp_dataset_dir, 'VOCAB.txt')

    vocab = read_vocab(vocab_path)

    data_dir = os.path.join(pp_dataset_dir, 'data')

    res = ""
    for idx, fname in enumerate(os.listdir(data_dir)):
        if idx != 0:
            res += ' '
        filepath = os.path.join(data_dir, fname)
        res += read_preprocessed_file_as_str(filepath, vocab, sent_delim='<eos>')

    return res


def read_vocab(vocab_path):
    """
    Read a vocabulary file. Returns a list of words
    """
    vocab = []
    with open(vocab_path, 'r') as f:
        for line in f:
            vocab.append(line.strip('\n'))

    return vocab


def preprocess_worker(args):
    """
    Multiprocessing worker for preprocessing a text file
    """
    txt_path, dataset_dir, output_data_dir = args
    basename, ext = os.path.splitext(os.path.basename(txt_path))
    out_prefix = os.path.dirname(txt_path).replace(dataset_dir, '').replace('/', '_')
    if out_prefix:
        out_prefix += '_'
    out_path = os.path.join(output_data_dir, '{}{}.bin'.format(out_prefix, basename))

    tokens = preprocess_file(txt_path, out_path)

    return out_path, tokens


def save_worker(args):
    """
    Multiprocessing worker for saving a preprocessed file
    """
    output_path, vocab_path = args
    tmp_output_path = output_path[:-3] + 'tmp'

    word_to_idx = {w: idx for (idx, w) in enumerate(read_vocab(vocab_path))}

    # Read tmp file
    if os.path.exists(tmp_output_path):
        with gzip.open(tmp_output_path, 'rb') as f:
            try:
                sentences = encode_sentences(pk.load(f), word_to_idx)
            except EOFError:
                sentences = []
    if os.path.exists(tmp_output_path):
    # Remove tmp file
        os.remove(tmp_output_path)

        write_preprocessed_file(sentences, output_path)

def mem_clr_list_iterator(lst):
    while lst:
        yield lst.pop()


def preprocess_dataset(dataset_dir, output_dir, target_ext='.txt', num_workers=1):
    """
    Preprocesses a dataset by splitting each file into sentences, tokenizing
    each sentence, encoding the files, and saving them.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    output_dir = os.path.abspath(output_dir)
    output_data_dir = os.path.join(output_dir, 'data')

    if not os.path.isdir(dataset_dir):
        raise ValueError('Dataset directory {} does not exist'.format(dataset_dir))

    if not os.path.isdir(output_data_dir):
        os.makedirs(output_data_dir)

    vocab = set()
    worker_args = []

    LOGGER.info("Getting list of files...")
    # Get list of txt files
    for root, dirs, files in os.walk(dataset_dir):
        root = os.path.abspath(root)
        for fname in files:
            basename, ext = os.path.splitext(fname)
            if ext.lower() != target_ext.lower():
                continue

            if basename.lower() == 'readme':
                continue

            txt_path = os.path.join(root, fname)

            worker_args.append((txt_path, dataset_dir, output_data_dir))

    pool = mp.Pool(num_workers)

    LOGGER.info("Preprocessing files...")
    output_paths = []
    num_files = len(worker_args)
    # Preprocess each file and get the tokens in each file
    for idx, (out_path, tokens) in enumerate(pool.imap_unordered(preprocess_worker, worker_args)):
        output_paths.append(out_path)
        vocab.update(tokens)

        if ((idx+1) % 1000) == 0:
            LOGGER.info("Preprocessed {}/{} files".format(idx+1, num_files))

    pool.close()
    pool.join()


    # Sort vocab and make into a list
    LOGGER.info("Saving vocab...")
    vocab = list(sorted(vocab))

    # Write vocab to disk
    vocab_path = os.path.join(output_dir, 'VOCAB.txt')
    with open(vocab_path, 'w') as f:
        f.write('\n'.join(vocab))

    # Delete vocab from memory to preserve space
    del vocab

    # Encode preprocessed files and write them to disk
    # Wrap list in an iterator that removes elements in the list (thus freeing
    # the memory) as elements are yielded. This will prevent the memory usage
    # from doubling when data is copied to multiprocessing workers
    worker_args = mem_clr_list_iterator([(output_path, vocab_path)
                                         for output_path in output_paths])

    LOGGER.info("Saving files...")
    pool = mp.Pool(num_workers)
    for idx, _ in enumerate(pool.imap_unordered(save_worker, worker_args)):
        if ((idx+1) % 1000) == 0:
            LOGGER.info("Saved {}/{} files".format(idx+1, num_files))
    pool.close()
    pool.join()

    LOGGER.info("Done.")


def generate_input_for_glove(input_files,input_vocab, output_file):
    """Generates input for the gloves ./demo.sh.  The output_file or the input for glove is text file of words separated by            whitespaces
    """
    bin_files = glob.glob(input_files)

    words =''

    for f in bin_files:
        sentences = read_preprocessed_file(f, read_vocab(input_vocab))
        words += ' '.join([' '.join(sent) for sent in sentences])

    with open(output_file, "w") as outfile:
        outfile.write(words)


def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description='Preprocess text data into lists of tokens')
    parser.add_argument('dataset_dir', help='Path to directory containing text files', type=str)
    parser.add_argument('output_dir', help='Path to output directory', type=str)
    parser.add_argument('target_ext', help='Extension of relevant text files', type=str)
    parser.add_argument('-n', '--num-workers', dest='num_workers', type=int, default=1, help='Number of workers')
    return vars(parser.parse_args())


if __name__ == '__main__':
    init_console_logger(LOGGER)
    preprocess_dataset(**(parse_arguments()))
