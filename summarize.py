import torch
import torch.utils.data
import argparse
from tqdm import tqdm

from dataloader import src_collate_fn, TextSummarizationDataset
from transformer.Translator import Summarizer


def main():

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-src', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-vocab', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Prepare DataLoader
    data = torch.load(opt.src)
    data['settings'].cuda = opt.cuda

    test_loader = torch.utils.data.DataLoader(
        TextSummarizationDataset(
            #            src_word2idx=preprocess_data['dict']['src'],
            src_word2idx=data['dict']['src'],
            #            tgt_word2idx=preprocess_data['dict']['tgt'],
            tgt_word2idx=data['dict']['tgt'],
            #            src_insts=test_src_insts),
            src_insts=data['valid']['src'][0:10]),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=src_collate_fn)
    translator = Summarizer(opt)
    with open(opt.output, 'w') as f:
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            all_hyp, all_scores = translator.translate_batch(*batch)
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    print(len(idx_seq))
                    pred_line = ''.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq])
                    f.write('[@]' + pred_line + '\n\n')
    print('[Info] Finished.')


if __name__ == "__main__":
    main()
