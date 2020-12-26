import json
import numpy as np
from tqdm import tqdm
import re, os
import torch
import torch.optim as optim
from util import seq_padding , read_data ,get_pos2idx_idx2pos,index_sequence,evaluate
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from config import Config
from model import Bert_cls
import torch.nn as nn
from transformers.optimization import AdamW
import ast,csv
using_GPU = True
verb = ['VB','VBD','VBG','VBN','VBP','VBZ']
adj = ['JJ','JJR','JJS']
nos = ['NN','NNS','NNP','NNPS']
adv = ['RB','RBR','RBS','WRB']
pos_set = set()
raw_train_vua = []
with open('../datasets/tofel_seq/tofel_formatted_train.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    for line in lines:
        pos_seq = ast.literal_eval(line[2])
        for i in range(len(pos_seq)):
            if pos_seq[i] in verb:
                pos_seq[i] = 'Verbs'
            if pos_seq[i] in adj:
                pos_seq[i] = 'Adjectives'
            if pos_seq[i] in nos:
                pos_seq[i] = 'Nouns'
            if pos_seq[i] in adv:
                pos_seq[i] = 'Adverbs'
        label_seq = ast.literal_eval(line[1])
        assert (len(pos_seq) == len(label_seq))
        assert (len(line[0].split()) == len(pos_seq))
        raw_train_vua.append([line[0], label_seq, pos_seq])
        pos_set.update(pos_seq)
raw_val_vua = []
with open('../datasets/tofel_seq/tofel_formatted_val.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    for line in lines:
        pos_seq = ast.literal_eval(line[2])
        for i in range(len(pos_seq)):
            if pos_seq[i] in verb:
                pos_seq[i] = 'Verbs'
            if pos_seq[i] in adj:
                pos_seq[i] = 'Adjectives'
            if pos_seq[i] in nos:
                pos_seq[i] = 'Nouns'
            if pos_seq[i] in adv:
                pos_seq[i] = 'Adverbs'
        label_seq = ast.literal_eval(line[1])
        assert (len(pos_seq) == len(label_seq))
        assert (len(line[0].split()) == len(pos_seq))
        raw_val_vua.append([line[0], label_seq, pos_seq])
        pos_set.update(pos_seq)
pos2idx, idx2pos = get_pos2idx_idx2pos(pos_set)


for i in range(len(raw_train_vua)):
    raw_train_vua[i][2] = index_sequence(pos2idx, raw_train_vua[i][2])
for i in range(len(raw_val_vua)):
    raw_val_vua[i][2] = index_sequence(pos2idx, raw_val_vua[i][2])
#print(raw_train_vua[50][0],'\n',raw_train_vua[50][1],'\n',raw_train_vua[50][2])
#print(type(raw_train_vua[50][0]),type(raw_train_vua[50][1]),type(raw_train_vua[50][2]))
class myDataset(Dataset):
    def __init__(self, input_ids, segment_ids, mask_ids, tokens ,TEXT, label , pos):
        self.input_ids = input_ids
        self.input_segment = segment_ids
        self.input_mask = mask_ids
        self.input_labels = label
        self.input_pos = pos
        self.len = len(self.input_ids)
        self.TEXT = TEXT
    def __getitem__(self, index):
        return self.input_ids[index], self.input_segment[index], self.input_pos[index] , self.input_mask[index], self.input_labels[index],self.TEXT[index]
    def __len__(self):

        return self.len


def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    segment_ids = [item[1] for item in batch]
    input_pos = [item[2] for item in batch]
    input_mask = [item[3] for item in batch]
    labels = [item[4] for item in batch]
    text = [item[5] for item in batch]
    input_ids = seq_padding(input_ids)
    segment_ids = seq_padding(segment_ids)
    input_pos = seq_padding(input_pos)
    input_mask = seq_padding(input_mask)
    labels = seq_padding(labels)

    return {
        'input_ids': torch.LongTensor(input_ids),
        'segment_ids': torch.LongTensor(segment_ids),
        'pos' : torch.LongTensor(input_pos),
        'input_mask': torch.LongTensor(input_mask),
        'labels': torch.LongTensor(labels),
        'text': text
    }

criterion = nn.NLLLoss()
def train(train_data,val_data):
    print('loading corpus')
    config = Config()
    # [text, label ,pos]
    input_ids, segment_ids, mask_ids, tokens ,TEXT, label , pos = read_data(train_data)
    input_ids2, segment_ids2, mask_ids2, tokens2, TEXT2, label2, pos2 = read_data(val_data)
    # input_ids, segment_ids, mask_ids,, tokens ,TEXT,label,pos
    train_dataset = myDataset(input_ids, segment_ids, mask_ids, tokens ,TEXT, label , pos)
    train_loader = DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=1,  # mini batch size
        shuffle=True,  # random shuffle for training
        num_workers=2,
        collate_fn=collate_fn,  # subprocesses for loading data
    )
    val_dataset = myDataset(input_ids2, segment_ids2, mask_ids2, tokens2, TEXT2, label2, pos2)
    val_loader = DataLoader(
        dataset=val_dataset,  # torch TensorDataset format
        batch_size=1,  # mini batch size
        shuffle=True,  # random shuffle for training
        num_workers=2,
        collate_fn=collate_fn,  # subprocesses for loading data
    )
    #input,input_segment,input_pos , input_mask,input_labels,
    model = Bert_cls(config.bert_embedding, dropout_ratio=config.dropout_ratio,
                         dropout1=config.dropout1, use_cuda=config.use_cuda)
    if using_GPU:
        model = model.cuda()
    '''
    no_decay = ["bias", "LayerNorm.weight"]
    #rnn_optimizer = optim.Adam(RNNseq_model.parameters(), lr=0.005)
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": config.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
    #optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    device = torch.device('cuda:0')
    model = model.to(device)
    #model = model.cuda()
    num_iter = 0
    '''
    loss_criterion = nn.NLLLoss()
    # Set up an optimizer for updating the parameters of the rnn_clf
    rnn_optimizer = optim.Adam(model.parameters(), lr=0.005)
    # Number of epochs (passes through the dataset) to train the model for.
    num_epochs = 15

    '''
    3. 2
    train model
    '''

    val_loss = []
    performance_matrix = None

    # A counter for the number of gradient updates
    num_iter = 0
    comparable = []
    for epoch in range(num_epochs):
        print("Starting epoch {}".format(epoch + 1))
        for batch in train_loader:
            inputs = batch['input_ids']
            masks = batch['input_mask']
            label = batch['labels']
            pos = batch['pos']
            # print(label)

            inputs, label, masks, pos = Variable(inputs), Variable(label), Variable(masks), Variable(pos)
            if config.use_cuda:
                inputs, label, masks, pos = inputs.cuda(), label.cuda(), masks.cuda(), pos.cuda()

            # predicted shape: (batch_size, seq_len, 2)
            predicted = model(inputs, masks)
            print(predicted.view(-1, 2))
            batch_loss = loss_criterion(predicted.view(-1, 2), label.view(-1))
            batch_loss = torch.sum(batch_loss.mul(masks)).float() / torch.sum(masks).float()
            rnn_optimizer.zero_grad()
            batch_loss.backward()
            rnn_optimizer.step()
            num_iter += 1
        avg_eval_loss, performance_matrix = evaluate(idx2pos, val_loader, model,
                                                     loss_criterion)
        val_loss.append(avg_eval_loss)
        print("Iteration {}. Validation Loss {}.".format(num_iter, avg_eval_loss))

'''
    for epoch in range(config.base_epoch):
        step = 0
        #loss_total = 0
        for batch in train_loader:
            step += 1
            model.train()
            #model.zero_grad()
            inputs = batch['input_ids']
            masks = batch['input_mask']
            label = batch['labels']
            pos = batch['pos']
            #print(label)

            inputs, label, masks, pos = Variable(inputs), Variable(label), Variable(masks), Variable(pos)

            if config.use_cuda:
                inputs, label, masks, pos = inputs.cuda(), label.cuda(), masks.cuda(), pos.cuda()
            #print(label.size())
            out = model(inputs, masks)
            loss = criterion(out.view(-1, 2), label.view(-1)).float()
            loss = loss.mean()
            #loss = torch.sum(loss.mul(masks)).float() / torch.sum(masks).float()
            #loss_total += loss.item()
            #loss = loss.mean().float()
            if step % 100 == 0:
                print('step: {} |  epoch: {}|  loss: {}'.format(step, epoch, loss.item()))
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            # scheduler.step()

            num_iter += 1
        avg_eval_loss, performance_matrix = evaluate(idx2pos, val_loader, model,
                                                     criterion)
        print("Iteration {}. Validation Loss {}.".format(num_iter, avg_eval_loss))
'''

train(raw_train_vua,raw_val_vua)
print("######train done##########")