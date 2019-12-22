import os
import torch
import torch.utils.data
import argparse
from tqdm import tqdm

from dataloader import src_collate_fn, TextSummarizationDataset
from transformer.Translator import Summarizer
from flask import Flask, jsonify, request
from tokenizer import FullTokenizer
import transformer.Constants as Constants

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

parser = argparse.ArgumentParser(description='server.py')
parser.add_argument('-trained_model',
                    default=os.path.dirname(os.path.abspath(__file__)) + '/data/checkpoint/trained/trained_20191004.chkpt',
                    help='Path to model .pt file')
parser.add_argument('-src', default=os.path.dirname(os.path.abspath(__file__)) + '/data/preprocessed_data.data',
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-vocab', default=os.path.dirname(os.path.abspath(__file__)) + '/data/checkpoint/vocab.txt',
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-beam_size', type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best decoded sentences""")
parser.add_argument('-no_cuda', action='store_true')

opt = parser.parse_args()
opt.cuda = not opt.no_cuda

# Prepare DataLoader
data = torch.load(opt.src)
data['settings'].cuda = opt.cuda

# Create Translator Model
translator = Summarizer(opt)

# Create Tokenizer
tokenizer = FullTokenizer(opt.vocab)


@app.route('/', methods=['POST'])
def summarization():
    json_data = request.get_json()

    data_loader = preprocess(json_data)
    summaries = summarize(data_loader)
    summaries = remove_symbol(summaries)

    return jsonify({
        'summaries': summaries,
    })


def preprocess(json_data):
    """
     Preprocess input data.
     1. Extract text list from JSON data.
     2. Divide text list into the words.
     3. Convert text list to vocabulary id.
     4. Convert id list to DataLoader.

    Args:
        json_data (object): received JSON data.

    Returns:
        data_loader (object): This data has been processed with tokenize, token2id, and toDataloader.

    """
    # 1. Extract text list from JSON data.
    texts = get_text_from_json(json_data)

    # 2. Divide text list into the words.
    tokenized_texts = tokenize(texts)

    # 3. Convert text list to vocabulary id.
    tokens = text2token(tokenized_texts)

    # 4. Convert id list to DataLoader.
    data_loader = toDataLoader(tokens)

    return data_loader


def get_text_from_json(json_data):
    """
     Extract text list from JSON data.

    Args:
        json_data (object): input data.

    Returns:
        texts (list): text list

    """
    return json_data['source_texts']


def tokenize(texts):
    """
     Tokenize row text list. We use MeCab to tokenize sentences.

    Args:
        texts (list): The text list to tokenize.

    Returns:
        tokenized_texts (list): The tokenized text list.

    """
    max_len = 512
    splited_texts = []

    for text in texts:
        splited_text = tokenizer.tokenize(_convert_num_half_to_full(text.replace('。\n', '\n').replace('\n', '。\n')))
        if len(splited_text) > (max_len - 2):
            splited_text = splited_text[:max_len - 2]
        splited_texts.append(splited_text)

    return splited_texts


def _convert_num_half_to_full(text):
    table = str.maketrans({
        '0': '０',
        '1': '１',
        '2': '２',
        '3': '３',
        '4': '４',
        '5': '５',
        '6': '６',
        '7': '７',
        '8': '８',
        '9': '９',
    })
    return text.translate(table)


def text2token(texts):
    """
     Convert input text list to vocabulary id (token) list.

    Args:
        texts (list): input text list.

    Returns:
        tokens (list): vocabulary id list.

    """
    tokens = []

    for text in texts:
        token = [Constants.BOS] + \
                [data['dict']['src'].get(i, Constants.UNK) for i in text] + \
                [Constants.EOS]
        tokens.append(token)

    return tokens


def toDataLoader(tokens):
    """
     Create DataLoader object from input vocabulary id list.

    Args:
        tokens (list): vocabulary id list.

    Returns:
        data_loader (object): DataLoader created from ids.

    """
    return torch.utils.data.DataLoader(
        TextSummarizationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=tokens),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=src_collate_fn)


def summarize(data_loader):
    """
     Summarize text in DataLoader with trained Deep Learning Model.

    Args:
        data_loader (DataLoader): inputted DataLoader.

    Returns:
        summarized_texts (list): summarized test list.

    """
    pred_lines = []

    # Prediction
    for batch in tqdm(data_loader, mininterval=2, desc='  - (Test)', leave=False):
        all_hyp, all_scores = translator.translate_batch(*batch)
        for idx_seqs in all_hyp:
            for idx_seq in idx_seqs:
                pred_line = ''.join([data_loader.dataset.tgt_idx2word[idx] for idx in idx_seq])
                pred_lines.append(pred_line)

    return pred_lines


def remove_symbol(texts):
    """
     Remove symbol "##", "[SEP]".

    Args:
        texts(list) : input text list

    Returns:
        removed_text(list): text list that removed symbol "##", "[SEP]"

    """
    removed_texts = []
    for text in texts:
        removed_texts.append(text.replace('##', '').replace('[SEP]', ''))

    return removed_texts


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006)
