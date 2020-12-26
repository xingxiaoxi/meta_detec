import json
import numpy as np
from tqdm import tqdm
from bert4keras.tokenizer import Tokenizer, load_vocab
import torch
from config import Config
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertModel
import torch.nn as nn

dict_path = 'vocab.txt'
token_dict = load_vocab(dict_path)
tokenizer = Tokenizer(token_dict)
config = Config()
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)

    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

def read_data(data):
    idxs = range(len(data))
    idxs = list(idxs)
    np.random.shuffle(idxs)
    X1, X2, Input_mask = [], [], []
    Tokens, TEXT = [], []
    label, pos = [], []
    for i in tqdm(idxs):
        text = data[i][0]
        #text,label,pos
        word2label = {}
        word2pos = {}
        w_label = []
        w_pos = []
        word = text.split()
        current_words = []
        S1, S2 = [], []
        for j in range(len(word)):
            tokens = tokenizer.tokenize(word[j])
            s1, s2 = tokenizer.encode(word[j])
            for k in range(1, len(tokens)-1):
                w_label.append(data[i][1][j])
                w_pos.append(data[i][2][j])
                S1.append(s1[k])
                S2.append(s2[k])
                current_words.append(tokens[k])
        #tokens = tokenizer.tokenize(text.strip())
        #s1, s2 = tokenizer.encode(text.strip())
        input_mask = len(S1) * [1]
        Input_mask.append(input_mask)
        TEXT.append(text)
        #for index, token in enumerate(tokens):
           # current_words.append(token)
           # S1.append(s1[index])
           # S2.append(s2[index])
        label.append(w_label)
        pos.append(w_pos)
        Tokens.append(current_words)
        X1.append(S1)
        X2.append(S2)
        #print(Tokens)
        #print(len(data[i][1]), '333333333333333333333', len(data[i][0].split()), len(S1), len(current_words))
    return [X1, X2, Input_mask, Tokens, TEXT, label ,pos]
    # input_ids, segment_ids, mask_ids,, tokens ,TEXT,label,pos
def get_pos2idx_idx2pos(vocab):

    word2idx = {}
    idx2word = {}
    for word in vocab:
        assigned_index = len(word2idx)
        word2idx[word] = assigned_index
        idx2word[assigned_index] = word
    return word2idx, idx2word


def index_sequence(item2idx, seq):

    embed = []
    for x in seq:
        embed.append(item2idx[x])
    assert (len(seq) == len(embed))
    return embed
def evaluate(idx2pos, evaluation_dataloader, model, criterion):
    """
    Evaluate the model on the given evaluation_dataloader

    :param evaluation_dataloader:
    :param model:
    :param criterion: loss criterion
    :param using_GPU: a boolean
    :return:
     average_eval_loss
     a matrix (#allpostags, 4) each row is the PRFA performance for a pos tag
    """
    # Set model to eval mode, which turns off dropout.
    model.eval()
    device = torch.device('cuda:0')
    # total_examples = total number of words
    total_eval_loss = 0
    confusion_matrix = np.zeros((len(idx2pos), 2, 2))
    for batch in evaluation_dataloader:
        inputs = batch['input_ids']
        masks = batch['input_mask']
        label = batch['labels']
        pos = batch['pos']
        text = batch['text']
        inputs, label, masks, pos = Variable(inputs), Variable(label), Variable(masks), Variable(pos)

        if config.use_cuda:
            inputs, label, masks, pos = inputs.cuda().to(device), label.cuda().to(device), masks.cuda().to(
                device), pos.cuda().to(device)

        out = model(inputs, masks)
        total_eval_loss = criterion(out.view(-1, 2), label.view(-1)).float()
        _, predicted_labels = torch.max(out.data, 2)
        #print(predicted_labels,label,text)
        confusion_matrix = update_confusion_matrix(confusion_matrix, predicted_labels, label.data, pos)

    average_eval_loss = total_eval_loss / evaluation_dataloader.__len__()

    # Set the model back to train mode, which activates dropout again.
    model.train()
    return average_eval_loss.data.item(), print_info(confusion_matrix, idx2pos)
def update_confusion_matrix(matrix, predictions, labels, pos_seqs):

    for i in range(len(pos_seqs)):
        indexed_pos_sequence = pos_seqs[i]
        prediction = predictions[i]
        label = labels[i]
        for j in range(len(indexed_pos_sequence)):
            indexed_pos = indexed_pos_sequence[j]
            p = prediction[j]
            l = label[j]
            matrix[indexed_pos][p][l] += 1
    return matrix
def print_info(matrix, idx2pos):
    """
    Prints the precision, recall, f1, and accuracy for each pos tag
    Assume that the confusion matrix is implicitly mapped with the idx2pos
    i.e. row 0 in confusion matrix is for the pos tag mapped by int 0 in idx2pos

    :param matrix: a confusion matrix of shape (#pos_tags, 2, 2)
    :param idx2pos: idx2pos: a dictionary: int --> pos tag
    :return: a matrix (#allpostags, 4) each row is the PRFA performance for a pos tag
    """
    result = {}
    for idx in range(len(idx2pos)):
        pos_tag = idx2pos[idx]
        grid = matrix[idx]
        precision = 100 * grid[1, 1] / np.sum(grid[1])
        recall = 100 * grid[1, 1] / np.sum(grid[:, 1])
        f1 = 2 * precision * recall / (precision + recall)
        precision1 = 100 * grid[0, 0] / np.sum(grid[0])
        recall1 = 100 * grid[0, 0] / np.sum(grid[:, 0])
        f2 = 2 * precision1 * recall1 / (precision1 + recall1)
        f3 = (f1 + f2) / 2
        #accuracy = 100 * (grid[1, 1] + grid[0, 0]) / np.sum(grid)
        print('F1 performance for ', pos_tag, f3)
        print(grid)
        result[pos_tag] = f3
    return result