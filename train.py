import numpy as np
import models.config as opt
import torch
from torch.optim import SGD, lr_scheduler, Adam
from torch.utils.data import TensorDataset, DataLoader
from models.NLIModel import GNLIModel
from torch import nn
from tqdm import tqdm


# TODO: bugs about embedding???


def create_emb_dict(emb_path):
    emb_file = open(emb_path, encoding='utf-8').read().split('\n')[:-1]
    labels, sentences, embeddings_dict, wd = [], [], {}, {}
    for i, line in enumerate(emb_file):
        values = line.split()
        wd[values[0]] = i
        vector = np.asarray(values[1:], dtype=np.float32)
        embeddings_dict[i] = torch.from_numpy(vector)
    return embeddings_dict, wd


def create_dataset(file_path, wd, max_len):
    labels, sentences = [], []
    file = open(file_path, encoding='utf-8').read().split('\n')[:-1]
    for line in file:
        ohs1 = [wd[word] for word in line.split('\t')[1].split(' ')]
        ohs2 = [wd[word] for word in line.split('\t')[2].split(' ')]
        labels.append([int(line.split('\t')[0]), len(ohs1), len(ohs1)])
        if len(ohs1) < max_len:
            ohs1 = ohs1 + [0] * (max_len - len(ohs1))
        if len(ohs2) < max_len:
            ohs2 = ohs2 + [0] * (max_len - len(ohs2))
        ohs1.extend(ohs2), sentences.append(ohs1)
    labels = torch.from_numpy(np.array(labels, dtype=np.int64))
    sentences = torch.from_numpy(np.array(sentences, dtype=np.int64))
    torch_dataset = TensorDataset(sentences, labels)
    data_loader = DataLoader(torch_dataset, 64, shuffle=False)
    return data_loader, len(torch_dataset)


def run_model(data_loader, data_len, loss_function, optim=None):
    with tqdm(total=data_len) as epoch_bar:
        epoch_bar.set_description(f'Epoch {epoch}')
        data_loss, data_acc = 0., 0.
        for idx, (bx, by) in enumerate(data_loader):
            bx = hot2emb(bx, emb_dict, 300)
            bx, by = bx.to(device), by.to(device)
            sent_pair = torch.split(bx, 82, dim=1)
            byl1l2 = torch.split(by, [1, 1, 1], dim=1)
            gt, s1l, s2l = byl1l2[0].squeeze(-1), byl1l2[1].squeeze(-1), byl1l2[2].squeeze(-1)
            out = model(sent_pair[0], sent_pair[1], s1l, s2l)
            loss = loss_function(out, gt)
            pred = torch.max(out, 1)[1]
            data_acc += (pred == gt).sum().item()
            data_loss += loss.item()
            if optim is not None:
                optim.zero_grad(), loss.backward(), optim.step()
            desc = f'Epoch {epoch} - loss {data_loss / (idx + 1):.4f}'
            epoch_bar.set_description(desc)
            epoch_bar.update(bx.shape[0])
    return data_loss, data_acc


def hot2emb(batch_tensor, embedding_dict, emb_dim):
    size_lt = batch_tensor.size()
    batch_vec = batch_tensor.numpy()
    embed_sent = np.zeros([size_lt[0], size_lt[1], emb_dim], dtype=np.float32)
    for i in range(size_lt[0]):
        for j in range(size_lt[1]):
            embed_sent[i, j, :] = embedding_dict[batch_vec[i][j]]
    return torch.from_numpy(embed_sent)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emb_dict, word_dict = create_emb_dict(opt.embedding_file)
train_loader, train_length = create_dataset('dataset/train.txt', word_dict, 82)
dev_loader, dev_length = create_dataset('dataset/dev.txt', word_dict, 82)
test_loader, test_length = create_dataset('dataset/test.txt', word_dict, 82)
model = GNLIModel(opt, 300).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.1)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, mode='max', patience=2)
for epoch in range(opt.epochs):
    train_loss, train_acc = run_model(train_loader, train_length, loss_func, optimizer)
    dev_loss, dev_acc = run_model(dev_loader, dev_length, loss_func)
    print('Train Loss: {:.3f}, Acc: {:.3f}  Dev Loss: {:.3f}, Acc: {:.3f}'.format(
        train_loss / (train_length // 64), train_acc / train_length,
        dev_loss / (dev_length // 64), dev_acc / dev_length))
    scheduler.step(train_acc)
test_loss, test_acc = run_model(test_loader, test_length, loss_func)
print('Test Loss: {:.3f}, Acc: {:.3f}'.format(
    test_loss / (train_length // 64), test_acc / train_length))
