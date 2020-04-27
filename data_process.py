import numpy as np
import nltk


def gen_neat_file(ori_path, neat_path, max_length):
    file = open(ori_path, encoding='utf-8').read().split('\n')[1:]
    neat_file = open(neat_path, 'w', encoding='utf-8')
    label_dict = {'entailment': '0', 'neutral': '1', 'contradiction': '2'}
    for line in file[:-1]:
        line = line.split('\t')
        if line[0] != '-':
            all_words1, all_words2 = [], []
            sents1 = nltk.sent_tokenize(line[5].lower())
            for sent in sents1:
                all_words1.extend(nltk.word_tokenize(sent))
            if max_length < len(all_words1):
                max_length = len(all_words1)
            sents2 = nltk.sent_tokenize(line[6].lower())
            for sent in sents2:
                all_words2.extend(nltk.word_tokenize(sent))
            if max_length < len(all_words2):
                max_length = len(all_words2)
            neat_file.write(label_dict[line[0]] + '\t' + ' '.join(all_words1) + '\t' + ' '.join(all_words2) + '\n')
    neat_file.close()
    return max_length


def gen_word_dict(neat_path, wd):
    train_file = open(neat_path, encoding='utf-8').read().split('\n')
    for line in train_file[:-1]:
        s1_tokens = line.split('\t')[1].split(' ')
        s2_tokens = line.split('\t')[2].split(' ')
        s1_tokens.extend(s2_tokens)
        for word in s1_tokens:
            if word not in wd:
                wd[word] = len(wd) + 1
    return wd


def gen_custom_glove(glove_path, custom_path):
    emb_file = open(glove_path, encoding='utf-8').read().split('\n')
    new_emb_file = open(custom_path, 'w', encoding='utf-8')
    word_list, emb_dim = [], len(emb_file[0].split(' ')[1:])
    for line_id, line in enumerate(emb_file):
        if line.split(' ')[0] in word_dict:
            new_emb_file.write(line + '\n')
        word_list.append(line.split(' ')[0])
    for word in word_dict:
        if word not in word_list:
            new_vector = np.random.normal(loc=-0.002, scale=0.4, size=(emb_dim,))
            new_emb_file.write(word + ' ' + ' '.join([str(ele) for ele in list(new_vector)]) + '\n')


word_dict, max_len = {}, 0
max_len = gen_neat_file('snli_1.0/snli_1.0_train.txt', 'dataset/train.txt', max_len)
max_len = gen_neat_file('snli_1.0/snli_1.0_test.txt', 'dataset/test.txt', max_len)
max_len = gen_neat_file('snli_1.0/snli_1.0_dev.txt', 'dataset/dev.txt', max_len)
word_dict = gen_word_dict('dataset/train.txt', word_dict)
word_dict = gen_word_dict('dataset/test.txt', word_dict)
word_dict = gen_word_dict('dataset/dev.txt', word_dict)
print(len(word_dict), max_len)
gen_custom_glove('dataset/glove.840B.300d.txt', 'dataset/glove.custom.300d.txt')