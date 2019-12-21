import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel, BertConfig
from transformer.Models import Decoder


class AbstractiveTextSummarizationUsingBert(nn.Module):
    def __init__(self, bert_model_path, n_tgt_vocab, len_max_seq, d_word_vec=768, d_model=768, d_inner=3072,
                 n_layers=12, n_head=12, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.encoder = BertModel.from_pretrained(bert_model_path)
        self.config = BertConfig(bert_model_path+'bert_config.json')
        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)
        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)
        self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
        self.x_logit_scale = (d_model ** -0.5)
        self.o_l = nn.Linear(d_model, 512, bias=False)
        self.h_l = nn.Linear(512, 1, bias=True)
        nn.init.xavier_normal_(self.o_l.weight)
        nn.init.xavier_normal_(self.h_l.weight)
        self.a_l_1 = nn.Linear(d_model, 512, bias=False)
        self.a_l_2 = nn.Linear(d_model, 512, bias=False)
        nn.init.xavier_normal_(self.a_l_1.weight)
        nn.init.xavier_normal_(self.a_l_2.weight)

    def forward(self, src_seq, src_sen, tgt_seq, tgt_pos):
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, _ = self.encoder(src_seq, src_sen, output_all_encoded_layers=False)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)

        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
