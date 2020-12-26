import torch.nn as nn
from transformers import BertModel
from torch.autograd import Variable
import torch
import numpy as np
class Bert_cls(nn.Module):

    def __init__(self,embedding_dim, dropout_ratio, dropout1,
                 use_cuda=False):
        super(Bert_cls, self).__init__()
        self.embedding_dim = embedding_dim
        #self.bert_sequence_ouput = BertModel.from_pretrained('/data/wwk/Bert-BiLSTM-CRF-pytorch-master/bert_weibo/')
        self.bert_sequence_ouput = BertModel.from_pretrained('/data/lmy/xbx/bert/pretrained_bert')
        self.dropout = nn.Dropout(0.25)
        self.liner = nn.Linear(embedding_dim, 2)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, input_ids,  attention_mask ):


        full = self.bert_sequence_ouput(input_ids, attention_mask=attention_mask)
        seq_output, _ = full
        #print(seq_output.size())
        s1 = self.dropout(seq_output)
        out = self.liner(s1)
        eval_l_out = self.softmax(out)
        #print(eval_l_out.size())
        #batch * token * 2
        return eval_l_out
