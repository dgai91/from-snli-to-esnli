import torch
from torch import nn
import models.config as opt


class BiLSTMPoolingEncoder(nn.Module):
    def __init__(self, args, emb_dim):
        self.args = args
        super(BiLSTMPoolingEncoder, self).__init__()
        self.lstm = nn.LSTM(emb_dim, args.max_lstm_size,
                            batch_first=True, bidirectional=True)
        self.pooling = nn.AdaptiveMaxPool1d(1) if args.is_max_pooling else nn.AdaptiveAvgPool1d(1)

    def forward(self, sent, sent_len_sorted):
        _, idx_sort = torch.sort(sent_len_sorted, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        sent_repr = sent.index_select(0, idx_sort)
        sent_len_sorted = sent_len_sorted[idx_sort]
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent_repr, sent_len_sorted, True)
        sent_output = self.lstm(sent_packed)[0]
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)
        sent_output = sent_output[0].index_select(0, idx_unsort)
        sent_repr = self.pooling(sent_output.transpose(2, 1)).squeeze(-1)
        return sent_repr


class BiLSTMSelfAttEncoder(nn.Module):
    def __init__(self, args, emb_weight):
        self.args = args
        super(BiLSTMSelfAttEncoder, self).__init__()
        self.embed = nn.Embedding(emb_weight.shape[0], emb_weight.shape[1])
        self.embed.load_state_dict({'weight': emb_weight})
        self.embed.weight.requires_grad = args.is_trainable
        self.lstm = nn.LSTM(emb_weight.shape[1], args.att_lstm_size, bidirectional=True)
        self.W = nn.Linear(2 * args.att_lstm_size, 2 * args.att_lstm_size)
        self.UW = nn.Linear(2 * args.att_lstm_size, 4, bias=False)

    def forward(self, sent, sent_len_sorted):
        sent_repr = self.embed(sent)
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent_repr, sent_len_sorted, True, False)
        sent_output = self.lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)[0]
        sent_repr_ = torch.tanh(self.W(sent_output))
        multi_att = torch.softmax(self.UW(sent_repr_), dim=1)
        att_sent_repr = torch.einsum('abc,abd->abcd', [sent_repr_, multi_att])
        sent_repr = torch.sum(att_sent_repr, dim=1)
        sent_repr = sent_repr.view(-1, 4 * 2 * self.args.att_lstm_size)
        return sent_repr


class HiConvNetEncoder(nn.Module):
    def __init__(self, args, emb_weight):
        self.args = args
        super(HiConvNetEncoder, self).__init__()
        self.embed = nn.Embedding(emb_weight.shape[0], emb_weight.shape[1])
        self.embed.load_state_dict({'weight': emb_weight})
        self.embed.weight.requires_grad = args.is_trainable
        self.conv_list = nn.ModuleList()
        self.first_conv = nn.Conv1d(emb_weight.shape[1], args.conv_size,
                                    args.kernel_size, padding=args.kernel_size // 2)
        self.conv_list.append(self.first_conv)
        for layer_num in range(1, 4):
            self.conv_list.append(nn.Conv1d(args.conv_size, args.conv_size,
                                            args.kernel_size, padding=args.kernel_size // 2))
        self.pooling = nn.MaxPool1d(args.sent_length, stride=1)

    def forward(self, sent):
        sent_repr = self.embed(sent).transpose(2, 1)
        u_list = []
        for layer in self.conv_list:
            sent_repr = torch.relu(layer(sent_repr))
            u_list.append(self.pooling(sent_repr).squeeze(-1))
        sent_repr = torch.cat(u_list, dim=-1)
        return sent_repr


class GNLIModel(nn.Module):
    def __init__(self, args, emb_dim):
        self.args = args
        super(GNLIModel, self).__init__()
        self.encoder_model = globals()[args.encoder_name](args, emb_dim)
        encoder_size = args.feature_dict[args.encoder_name]
        self.hidden_layer = nn.Linear(encoder_size * 4, args.hidden_size)
        self.classifier = nn.Linear(args.hidden_size, args.num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, premise, hypothesis, p_len, h_len):
        u = self.encoder_model(premise, p_len)
        v = self.encoder_model(hypothesis, h_len)
        concat_ph = torch.cat([u, v, torch.abs(u - v), torch.mul(u, v)], dim=-1)
        hidden = self.sigmoid(self.hidden_layer(concat_ph))
        logits = self.classifier(hidden)
        return logits
