import numpy as np
import os
import json
from nltk.tokenize import WordPunctTokenizer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type = str, default =  "../data")
parser.add_argument('--out_path', type = str, default = "prepro_data")

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
case_sensitive = False

char_limit = 16
sent_limit = 25
word_size = 100
train_distant_file_name = os.path.join(in_path, 'train_distant.json')
train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
dev_file_name = os.path.join(in_path, 'dev.json')
test_file_name = os.path.join(in_path, 'test.json')

rel2id = json.load(open(os.path.join(out_path, 'rel2id.json'), "r"))
id2rel = {v:u for u,v in rel2id.items()}
json.dump(id2rel, open(os.path.join(out_path, 'id2rel.json'), "w"))
fact_in_train = set([])
fact_in_dev_train = set([])
excess_limit = 0

def sents_2_idx(sents, word2id):
    global excess_limit
    sents_idx = np.zeros([sent_limit, word_size]) + word2id['BLANK']
    if len(sents) > sent_limit:
        excess_limit += 1
    for i, sent in enumerate(sents[:sent_limit]):
        for j, word in enumerate(sent[:word_size]):
            word = word.lower()
            if word in word2id:
                sents_idx[i][j] = word2id[word]
            else:
                sents_idx[i][j] = word2id['UNK']
    return sents_idx.tolist()

def get_corr_matrix(ins, rel2id, rel_num):
    labels = ins['labels']
    vetexSet = ins['vertexSet']
    entity_size = len(vetexSet)
    corr_matrix = np.zeros((rel_num, rel_num))
    ent_rel_table = np.zeros((entity_size, rel_num))

    for ins in labels:
       h_id = ins['h']
       t_id = ins['t']
       r_id = rel2id[ins['r']]
       ent_rel_table[h_id][r_id] += 1
       ent_rel_table[t_id][r_id] += 1
    '''
    for i in range(len(ent_rel_table)):
        for j in range(rel_num):
            for k in range(rel_num):
                corr_matrix[j][k] += min(ent_rel_table[i][j], ent_rel_table[i][k])
    '''
    table_1 = ent_rel_table.reshape(entity_size, rel_num, 1).repeat(rel_num,-1)
    table_2 = ent_rel_table.reshape(entity_size, 1, rel_num).repeat(rel_num, -2)
    corr_matrix = np.minimum(table_1, table_2).sum(0)
    return corr_matrix
def init(data_file_name, rel2id, max_length = 512, is_training = True, suffix=''):
    ori_data = json.load(open(data_file_name))

    rel_num = len(rel2id)
    rel_corr_matrix = np.identity(rel_num)
    for i in range(len(ori_data)):
        if(i%1000==0):
            print(i)
        item = ori_data[i]
        rel_corr_matrix += get_corr_matrix(item, rel2id, rel_num)
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"
    np.save(os.path.join(out_path, name_prefix + suffix + '_rel_corr.npy'), rel_corr_matrix)


    '''

    Ma = 0
    Ma_e = 0
    data = []
    intrain = notintrain = notindevtrain = indevtrain = 0
    word2id = json.load(open(os.path.join(out_path, "word2id.json")))
    for i in range(len(ori_data)):
        Ls = [0]
        L = 0
        for x in ori_data[i]['sents']:
            L += len(x)
            Ls.append(L)

        vertexSet =  ori_data[i]['vertexSet']
        # point position added with sent start position
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet[j])):
                vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

                sent_id = vertexSet[j][k]['sent_id']
                dl = Ls[sent_id]
                pos1 = vertexSet[j][k]['pos'][0]
                pos2 = vertexSet[j][k]['pos'][1]
                vertexSet[j][k]['pos'] = (pos1+dl, pos2+dl)

        ori_data[i]['vertexSet'] = vertexSet

        item = {}
        item['vertexSet'] = vertexSet
        labels = ori_data[i].get('labels', [])

        train_triple = set([])
        new_labels = []
        for label in labels:
            rel = label['r']
            assert(rel in rel2id)
            label['r'] = rel2id[label['r']]

            train_triple.add((label['h'], label['t']))


            if suffix=='_train':
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_dev_train.add((n1['name'], n2['name'], rel))


            if is_training:
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_train.add((n1['name'], n2['name'], rel))

            else:
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        if (n1['name'], n2['name'], rel) in fact_in_train:
                            intrain += 1
                            label['intrain'] = True
                        else:
                            notintrain += 1
                            label['intrain'] = False

                        if suffix == '_dev' or suffix == '_test':
                            if (n1['name'], n2['name'], rel) in fact_in_dev_train:
                                indevtrain += 1
                                label['indev_train'] = True
                            else:
                                notindevtrain += 1
                                label['indev_train'] = False

            new_labels.append(label)

        item['labels'] = new_labels
        item['title'] = ori_data[i]['title']

        na_triple = []
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet)):
                if (j != k):
                    if (j, k) not in train_triple:
                        na_triple.append((j, k))

        item['na_triple'] = na_triple
        item['Ls'] = Ls
        item['sents'] = ori_data[i]['sents']
        item['sents_idx'] = sents_2_idx(ori_data[i]['sents'], word2id)
        if i%1000==0:
            print(i)
        data.append(item)

        Ma = max(Ma, len(vertexSet))
        Ma_e = max(Ma_e, len(item['labels']))


    print ('data_len:', len(ori_data))
    # print ('Ma_V', Ma)
    # print ('Ma_e', Ma_e)
    # print (suffix)
    print ('fact_in_train', len(fact_in_train))
    print (intrain, notintrain)
    print ('fact_in_devtrain', len(fact_in_dev_train))
    print (indevtrain, notindevtrain)
    print('excess sent limit', excess_limit)


    # saving
    print("Saving files")
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    json.dump(data , open(os.path.join(out_path, name_prefix + suffix + '.json'), "w"))

    char2id = json.load(open(os.path.join(out_path, "char2id.json")))
    # id2char= {v:k for k,v in char2id.items()}
    # json.dump(id2char, open("data/id2char.json", "w"))

    word2id = json.load(open(os.path.join(out_path, "word2id.json")))
    ner2id = json.load(open(os.path.join(out_path, "ner2id.json")))

    sen_tot = len(ori_data)
    sen_word = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_pos = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_ner = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_char = np.zeros((sen_tot, max_length, char_limit), dtype = np.int64)

    for i in range(len(ori_data)):
        item = ori_data[i]

        #rel_corr_matrix += get_corr_matrix(item, rel2id, rel_num)

        words = []
        for sent in item['sents']:
            words += sent

        for j, word in enumerate(words):
            word = word.lower()

            if j < max_length:
                if word in word2id:
                    sen_word[i][j] = word2id[word]
                else:
                    sen_word[i][j] = word2id['UNK']

            for c_idx, k in enumerate(list(word)):
                if c_idx>=char_limit:
                    break
                sen_char[i,j,c_idx] = char2id.get(k, char2id['UNK'])

        for j in range(j + 1, max_length):
            sen_word[i][j] = word2id['BLANK']

        vertexSet = item['vertexSet']

        for idx, vertex in enumerate(vertexSet, 1):
            for v in vertex:
                sen_pos[i][v['pos'][0]:v['pos'][1]] = idx
                sen_ner[i][v['pos'][0]:v['pos'][1]] = ner2id[v['type']]

    print("Finishing processing")
    np.save(os.path.join(out_path, name_prefix + suffix + '_word.npy'), sen_word)
    np.save(os.path.join(out_path, name_prefix + suffix + '_pos.npy'), sen_pos)
    np.save(os.path.join(out_path, name_prefix + suffix + '_ner.npy'), sen_ner)
    np.save(os.path.join(out_path, name_prefix + suffix + '_char.npy'), sen_char)
    #np.save(os.path.join(out_path, name_prefix + suffix + '_rel_corr.npy'), rel_corr_matrix)
    print("Finish saving")

    '''



init(train_distant_file_name, rel2id, max_length = 512, is_training = True, suffix='')
init(train_annotated_file_name, rel2id, max_length = 512, is_training = False, suffix='_train')
init(dev_file_name, rel2id, max_length = 512, is_training = False, suffix='_dev')
init(test_file_name, rel2id, max_length = 512, is_training = False, suffix='_test')


