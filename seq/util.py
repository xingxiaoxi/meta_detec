from tqdm import tqdm
import torch
import numpy as np
import mmap
import ast
import csv
import datetime
import os
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


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

def get_embedding(word2idx,  idx2word, embedding_dim ,path,normalization=False):


    glove_vectors = {}
    with open(path) as glove_file:
        for line in tqdm(glove_file, total=get_num_lines(path)):
            split_line = line.rstrip().split()
            word = split_line[0]
            if len(split_line) != (embedding_dim + 1) or word not in word2idx:
                continue
            assert (len(split_line) == embedding_dim + 1)
            vector = np.array([float(x) for x in split_line[1:]], dtype="float32")
            if normalization:
                vector = vector / np.linalg.norm(vector)
            assert len(vector) == embedding_dim
            glove_vectors[word] = vector
    print("Number of pre-trained word vectors loaded: ", len(glove_vectors))


    all_embeddings = np.array(list(glove_vectors.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_stdev = float(np.std(all_embeddings))
    print("Embeddings mean: ", embeddings_mean)
    print("Embeddings stdev: ", embeddings_stdev)


    vocab_size = len(word2idx)

    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean, embeddings_stdev)
    for i in range(2, vocab_size):
        word = idx2word[i]
        if word in glove_vectors:
            embedding_matrix[i] = torch.FloatTensor(glove_vectors[word])
    if normalization:
        for i in range(vocab_size):
            embedding_matrix[i] = embedding_matrix[i] / float(np.linalg.norm(embedding_matrix[i]))

    #embedding_matrix = torch.randn(vocab_size, embedding_dim)
    embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    embeddings.weight = nn.Parameter(embedding_matrix)
    return embeddings


def get_vocab(raw_dataset):

    vocab = []
    for example in raw_dataset:
        vocab.extend(example[0].split())
    vocab = set(vocab)
    print("vocab size: ", len(vocab))
    return vocab


def get_word2idx_idx2word(vocab):
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    for word in vocab:
        assigned_index = len(word2idx)
        word2idx[word] = assigned_index
        idx2word[assigned_index] = word
    return word2idx, idx2word


def embed_indexed_sequence(sentence, word2idx, glove_embeddings, fea_embeddings):

    words = sentence.split()

    # 1. embed the sequence by glove vector
    # Replace words with tokens, and 1 (UNK index) if words not indexed.
    indexed_sequence = [word2idx.get(x, 1) for x in words]
    # glove_part has shape: (seq_len, glove_dim)
    glove_part = glove_embeddings(Variable(torch.LongTensor(indexed_sequence)))
    fea_part = fea_embeddings(Variable(torch.LongTensor(indexed_sequence)))

    result = np.concatenate((glove_part.data, fea_part.data), axis=1)

    assert (len(words) == result.shape[0])
    return result


def evaluate(idx2pos, evaluation_dataloader, model, criterion, using_GPU):
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

    # total_examples = total number of words
    total_examples = 0
    total_eval_loss = 0
    confusion_matrix = np.zeros((len(idx2pos), 2, 2))
    for (eval_pos_seqs, eval_text, eval_lengths, eval_labels) in evaluation_dataloader:
        eval_text = Variable(eval_text, volatile=True)
        eval_lengths = Variable(eval_lengths, volatile=True)
        eval_labels = Variable(eval_labels, volatile=True)
        if using_GPU:
            eval_text = eval_text.cuda()
            eval_lengths = eval_lengths.cuda()
            eval_labels = eval_labels.cuda()

        # predicted shape: (batch_size, seq_len, 2)
        predicted = model(eval_text, eval_lengths)
        # Calculate loss for this test batch. This is averaged, so multiply
        # by the number of examples in batch to get a total.
        total_eval_loss += criterion(predicted.view(-1, 2), eval_labels.view(-1))
        # get 0 or 1 predictions
        # predicted_labels: (batch_size, seq_len)
        _, predicted_labels = torch.max(predicted.data, 2)
        total_examples += eval_lengths.size(0)
        confusion_matrix = update_confusion_matrix(confusion_matrix, predicted_labels, eval_labels.data, eval_pos_seqs)

    average_eval_loss = total_eval_loss / evaluation_dataloader.__len__()

    # Set the model back to train mode, which activates dropout again.
    model.train()
    return average_eval_loss.data.item(), print_info(confusion_matrix, idx2pos)
def test(idx2pos, evaluation_dataloader, model, criterion, using_GPU):
    model.eval()
    total_examples = 0
    total_eval_loss = 0
    confusion_matrix = np.zeros((len(idx2pos), 2, 2))
    for (eval_pos_seqs, eval_text, eval_lengths, eval_labels) in evaluation_dataloader:
        eval_text = Variable(eval_text, volatile=True)
        eval_lengths = Variable(eval_lengths, volatile=True)
        eval_labels = Variable(eval_labels, volatile=True)

        eval_text = eval_text.cuda()
        eval_lengths = eval_lengths.cuda()
        eval_labels = eval_labels.cuda()
        predicted = model(eval_text, eval_lengths)

        total_eval_loss += criterion(predicted.view(-1, 2), eval_labels.view(-1))

        _, predicted_labels = torch.max(predicted.data, 2)
        total_examples += eval_lengths.size(0)
        confusion_matrix = update_confusion_matrix(confusion_matrix, predicted_labels, eval_labels.data, eval_pos_seqs)

    average_eval_loss = total_eval_loss / evaluation_dataloader.__len__()

    # Set the model back to train mode, which activates dropout again.
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

def get_batch_predictions(predictions, pos_seqs):
    pred_lst = []
    for i in range(len(pos_seqs)):  # each example i.e. each row
        indexed_pos_sequence = pos_seqs[i]
        prediction_padded = predictions[i]
        cur_pred_lst = []
        for j in range(len(indexed_pos_sequence)):  # inside each example: up to sentence length
            cur_pred_lst.append(prediction_padded[j])
        pred_lst.append(cur_pred_lst)
    return pred_lst


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
def save_model(model, epoch, path='result', **kwargs):
    """
    默认保留所有模型
    :param model: 模型
    :param path: 保存路径
    :param loss: 校验损失
    :param last_loss: 最佳epoch损失
    :param kwargs: every_epoch or best_epoch
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    if kwargs.get('name', None) is None:
        cur_time = datetime.datetime.now().strftime('%Y-%m-%data#%H:%M:%S')
        name = cur_time + '--epoch:{}'.format(epoch)
        full_name = os.path.join(path, name)
        torch.save(model.state_dict(), full_name)
        print('Saved model at epoch {} successfully'.format(epoch))
        with open('{}/checkpoint'.format(path), 'w') as file:
            file.write(name)
            print('Write to checkpoint')

def load_model(model, path='result', **kwargs):
    if kwargs.get('name', None) is None:
        with open('{}/checkpoint'.format(path)) as file:
            content = file.read().strip()
            name = os.path.join(path, content)
    else:
        name = kwargs['name']
        name = os.path.join(path, name)
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(name, map_location=lambda storage, loc: storage).items()})
    # model.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage))
    print('load model {} successfully'.format(name))
    return model
class TextDatasetWithGloveElmoSuffix(Dataset):
    def __init__(self, embedded_text, pos_seqs, labels):

        if len(embedded_text) != len(labels):
            raise ValueError("Differing number of sentences and labels!")
        self.embedded_text = embedded_text
        self.pos_seqs = pos_seqs
        self.labels = labels

    def __getitem__(self, idx):
        example_pos_seq = self.pos_seqs[idx]
        example_text = self.embedded_text[idx]
        example_label_seq = self.labels[idx]
        example_length = example_text.shape[0]
        assert (example_length == len(example_pos_seq))
        assert (example_length == len(example_label_seq))
        return example_pos_seq, example_text, example_length, example_label_seq

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def collate_fn(batch):
        batch_padded_example_text = []
        batch_lengths = []
        batch_padded_labels = []
        batch_pos_seqs = []
        max_length = -1
        for pos, __, __, __ in batch:
            if len(pos) > max_length:
                max_length = len(pos)

        for pos, text, length, label in batch:
            batch_pos_seqs.append(pos)

            amount_to_pad = max_length - length
            pad_tensor = torch.zeros(amount_to_pad, text.shape[1])
            text = torch.Tensor(text)
            padded_example_text = torch.cat((text, pad_tensor), dim=0)
            padded_example_label = label + [0] * amount_to_pad

            batch_padded_example_text.append(padded_example_text)
            batch_lengths.append(length)
            batch_padded_labels.append(padded_example_label)
        return (batch_pos_seqs,
                torch.stack(batch_padded_example_text),
                torch.LongTensor(batch_lengths),
                torch.LongTensor(batch_padded_labels))
