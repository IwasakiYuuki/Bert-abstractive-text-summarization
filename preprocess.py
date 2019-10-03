import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlite3
import argparse
from tqdm import tqdm
import pandas.io.sql as psql
import transformer.Constants as Constants
import torch
import MeCab
from tokenizer import FullTokenizer
m = MeCab.Tagger('-Owakati')
tokenizer = FullTokenizer('checkpoint/vocab.txt')

def build_vocab(path_to_file):
    with open(path_to_file, encoding='utf-8') as f:
        vocab = f.readlines()
    token2text = {k: v.rstrip() for k, v in enumerate(vocab)}
    text2token = {v: k for k, v in token2text.items()}

    return text2token, token2text


def get_content_summary_from_df(df):
    content = df['content'].values.tolist()
    summary = df['summary'].values.tolist()

    return content, summary


def convert_text_to_token(content, summary, text2token, max_len):
    tokens_content = []
    tokens_summary = []
    for d_content, d_summary in tqdm(zip(content, summary), ascii=True, total=len(content)):
        tokens_content.append(convert_text_to_token_seq(d_content, text2token, max_len))
        tokens_summary.append(convert_text_to_token_seq(d_summary, text2token, max_len))

    return tokens_content, tokens_summary


def convert_text_to_token_seq(text, text2token, max_len):
    if len(text) > 2000:
        text = text[:2000]
    splited_text = tokenizer.tokenize(text)
#    splited_text = m.parse(text).split(' ')
#    splited_text = [s for s in splited_text if s]
    if len(splited_text) > (max_len - 2):
        splited_text = splited_text[:max_len-2]
    splited_text = [Constants.BOS] + \
                   [text2token.get(i, Constants.UNK) for i in splited_text] + \
                   [Constants.EOS]
    return splited_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vocab', required=True)
    parser.add_argument('-data', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('--max_word_seq_len',  required=True, type=int)
    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len

    connection = sqlite3.connect(opt.data)
    cursor = connection.cursor()
    df = psql.read_sql("SELECT * FROM Article;", connection)
    print('Finished reading db file.')

    text2token, token2text = build_vocab(opt.vocab)
    print('Finished building vocab.')
    content, summary = get_content_summary_from_df(df)
    content, summary = convert_text_to_token(content, summary, text2token, opt.max_word_seq_len)
    data = {
        'settings': opt,
        'dict': {
            'src': text2token,
            'tgt': text2token},
        'train': {
            'src': content[:100000],
            'tgt': summary[:100000]},
        'valid': {
            'src': content[100000:],
            'tgt': summary[100000:]}}
    torch.save(data, opt.save_data)


if __name__ == '__main__':
    main()
