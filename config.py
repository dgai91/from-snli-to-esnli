conv_size = 256
att_lstm_size = 128
max_lstm_size = 2048
hidden_size = 512
sent_length = 82
kernel_size = 3
num_class = 3
is_trainable = False
epochs = 10
is_max_pooling = True
embedding_file = r'D:\PyCharmProject\e-snli_exp\dataset\glove.custom.300d.txt'
word_dict_path = r'D:\PyCharmProject\e-snli_exp\dataset\word_dict.txt'
encoder_name = 'BiLSTMPoolingEncoder'
feature_dict = {'HiConvNetEncoder': 4 * conv_size,
                'BiLSTMSelfAttEncoder': 4 * 2 * att_lstm_size,
                'BiLSTMPoolingEncoder': 2 * max_lstm_size}


