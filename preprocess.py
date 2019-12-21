import io, sys
import sqlite3
import argparse
from tqdm import tqdm
import pandas.io.sql as psql
import transformer.Constants as Constants
import torch
from tokenizer import FullTokenizer
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


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


def convert_text_to_token(content, summary, text2token, max_len, tokenizer):
    tokens_content = []
    tokens_summary = []
    for d_content, d_summary in tqdm(zip(content, summary), ascii=True, total=len(content)):
        tokens_content.append(convert_text_to_token_seq(d_content, text2token, max_len, tokenizer))
        tokens_summary.append(convert_text_to_token_seq(d_summary, text2token, max_len, tokenizer))

    return tokens_content, tokens_summary


def convert_text_to_token_seq(text, text2token, max_len, tokenizer):
    if len(text) > 2000:
        text = text[:2000]
    splited_text = tokenizer.tokenize(convert_num_half_to_full(text.replace('。\n', '\n').replace('\n', '。\n')))
    if len(splited_text) > (max_len - 2):
        splited_text = splited_text[:max_len-2]
    splited_text = [Constants.BOS] + \
                   [text2token.get(i, Constants.UNK) for i in splited_text] + \
                   [Constants.EOS]
    return splited_text


def convert_num_half_to_full(text):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vocab', required=True)
    parser.add_argument('-data', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('--max_word_seq_len',  required=True, type=int)
    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len

    tokenizer = FullTokenizer(opt.vocab)
    connection = sqlite3.connect(opt.data)
    cursor = connection.cursor()
    df = psql.read_sql("SELECT * FROM Article;", connection)
    print('Finished reading db file.')
    text2token, token2text = build_vocab(opt.vocab)
    print('Finished building vocab.')
    content, summary = get_content_summary_from_df(df)
    content, summary = convert_text_to_token(content, summary, text2token, opt.max_word_seq_len, tokenizer)
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
