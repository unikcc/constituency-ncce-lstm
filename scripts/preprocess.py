#!/usr/bin/env python

import OpenHowNet
import os
import pickle as pkl
import json
from collections import Counter
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from scipy import sparse
import yaml

from stanfordcorenlp import StanfordCoreNLP
import logging
import random
import requests as rq

class Preprocessor:

    def __init__(self, args, config_name=None):
        self.args = args

        basename = os.path.basename(os.getcwd())
        #config_name = basename if config_name == None else config_name
        #config = json.load(open('config.json', 'r', encoding='utf-8'))
        #for cfg in config:
            #self.__setattr__(cfg, config[cfg])
        #self.max_length = 512
        with open('config.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for w, z in config.items():
            self.__setattr__(w, z)
            #setattr(args, w, z)

        self.data_dir = self.data_dir.format(basename)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        #self.target_dir = self.target_dir.format(basename)


    def omit_char(self, content):
        """
        [
        {
        'text': '\ufeff2019年西安数字经济产业博览会盛大开幕',
        'event_chain': [
        {
        'event': [{'entity: {'start': [0], 'end':[16], 'text': '\ufeff2019年西安数字经济产业博览会盛大开幕'},
                 ''trigger':{'start': [0], 'end':[16], 'text': '\ufeff2019年西安数字经济产业博览会盛大开幕'}}]
        'chain_index
        }
        ]
        }]
        :param content:
        :return:
        """
        omits = ['\ufeff', ' ', '\u3000', '\xa0', '\u200b', '\u200c']
        res = []
        for sentence_index, sentence in enumerate(content):
            text = sentence['text']
            event_chain = sentence['event_chain']
            copy_chain = deepcopy(event_chain)

            for i in range(len(copy_chain)):
                for j in range(len(copy_chain[i]['event'])):
                    for k in range(len(copy_chain[i]['event'][j]['trigger']['start'])):
                        copy_chain[i]['event'][j]['trigger']['start'][k] = 0
                        copy_chain[i]['event'][j]['trigger']['end'][k] = 0
                    for k in range(len(copy_chain[i]['event'][j]['entity']['start'])):
                        copy_chain[i]['event'][j]['entity']['start'][k] = 0
                        copy_chain[i]['event'][j]['entity']['end'][k] = 0

            new_event_chain = []
            for index, word in enumerate(text):
                if word in omits:
                    for i in range(len(event_chain)):
                        for j in range(len(event_chain[i]['event'])):
                            for k in range(len(event_chain[i]['event'][j]['trigger']['start'])):
                                tmp = event_chain[i]['event'][j]['trigger']['start'][k]
                                if tmp > index:
                                    copy_chain[i]['event'][j]['trigger']['start'][k] += 1
                                tmp = event_chain[i]['event'][j]['trigger']['end'][k]
                                if tmp >= index:
                                    copy_chain[i]['event'][j]['trigger']['end'][k]  += 1

                            for k in range(len(event_chain[i]['event'][j]['entity']['start'])):
                                tmp = event_chain[i]['event'][j]['entity']['start'][k]
                                if tmp > index:
                                    copy_chain[i]['event'][j]['entity']['start'][k] += 1
                                tmp = event_chain[i]['event'][j]['entity']['end'][k]
                                if tmp >= index:
                                    copy_chain[i]['event'][j]['entity']['end'][k]  += 1

            for i in range(len(copy_chain)):
                for j in range(len(copy_chain[i]['event'])):
                    for k in range(len(copy_chain[i]['event'][j]['trigger']['start'])):
                        event_chain[i]['event'][j]['trigger']['start'][k] -= copy_chain[i]['event'][j]['trigger']['start'][k]
                        event_chain[i]['event'][j]['trigger']['end'][k] -= copy_chain[i]['event'][j]['trigger']['end'][k]
                    for k in range(len(copy_chain[i]['event'][j]['entity']['start'])):
                        event_chain[i]['event'][j]['entity']['start'][k] -= copy_chain[i]['event'][j]['entity']['start'][k]
                        event_chain[i]['event'][j]['entity']['end'][k] -= copy_chain[i]['event'][j]['entity']['end'][k]
            text = ''.join([w for w in text if not w in omits])
            res.append({'text': text, 'event_chain':event_chain})

        temp_res = json.dumps(res, ensure_ascii=False)
        for w in omits:
            temp_res = temp_res.replace(w, '')
        temp_res = json.loads(temp_res)

        return temp_res

    def have_labeled(self, sentence, start, end):
        '''
        if sentence[start] == 'B' and all([sentence[i] == 'I' for i in range(start + 1, end + 1)]):
            return False
        '''
        if all([sentence[i] == 'O' for i in range(start, end + 1)] ):
            return False
        return True

    def mytokenize(self, line):
        line = line.replace('\t', '。')
        location_dict = {}
        tokens = self.tokenizer.tokenize(line)
        i, j = 0, 0
        while i < len(tokens) or j < len(line):
            if tokens[i] == line[j].lower() or tokens[i] == self.UNK:
                location_dict[j] = i
                i += 1
                j += 1
            else:
                tmp_length = len(tokens[i].replace('#', ''))
                #print(line[j:j+tmp_length], tokens[i])
                #print("line", ord(line[j]))
                assert line[j:j+tmp_length].lower() == tokens[i].replace('#', '')
                for k in range(tmp_length):
                    location_dict[j+k] = i
                j += tmp_length
                i += 1
        return tokens, location_dict

    def execute(self, mode):
        path = os.path.join(self.source_dir, '{}.json'.format(mode))
        a = open(path, 'r', encoding='utf-8').read().splitlines()
        a = [json.loads(w) for w in a]
        if self.args.test_mode == 'test':
            #a = a[:10]
            pass
        #a = a[:20]

        documents, document_chars, mention_set, labels, part_of_speeches = [], [], [], [], []
        for document_index, sentences in tqdm(enumerate(a), total=len(a)):
            #if document_index < 1700:
                #continue
            document, document_char, label, part_of_speech = [], [], [], []
            mention_dict = {}
            sentences = self.omit_char(sentences)
            # 遍历每一句话
            document_length = 0
            for sentence in sentences:
                text = sentence['text'][:self.max_length - 2]
                if len(text) < 3:
                    continue

                tokenized_text = list(text)
                #tokenized_text, location_dict = self.mytokenize(text)
                if len(tokenized_text) == 0 or (len(tokenized_text) == 1 and tokenized_text[0] == '。'):
                    continue
                #tokenized_text, chars, sentence_pos = self.get_pos(text, document_word_length,document_char_length)
                tokenized_text, chars, sentence_pos, index2dict = self.get_pos(text)
                event_chain = sentence['event_chain']
                chain_index = [w['chain_index'] for w in event_chain]
                event_chain = [w['event'] for w in event_chain]
                bios = ['O' for w in tokenized_text]
                # 遍历一句话中的每一个事件链
                for sub_chain_index, sub_event in zip(chain_index, event_chain):
                    # 遍历一个链中的每一个事件
                    if sub_chain_index not in mention_dict:
                        mention_dict[sub_chain_index] = []
                    for event in sub_event:
                        entity = event['entity']
                        if len(entity['start']) == 0:
                            continue
                        start = entity['start'][-1]
                        end = entity['end'][-1]
                        if start > self.max_length - 3:
                            continue
                        if end > self.max_length - 3:
                            end = self.max_length - 3

                        #start = location_dict[start]
                        #end = location_dict[end]
                        start = index2dict[start]
                        end = index2dict[end]

                        if self.have_labeled(bios, start, end):
                            continue
                        bios[start] = 'B'
                        for i in range(start + 1, end+1):
                            bios[i] = 'I'

                        start = start + document_length
                        end = end + document_length
                        if (start, end) in mention_dict[sub_chain_index]:
                            continue
                        mention_dict[sub_chain_index].append((start, end))
                document_length += len(tokenized_text)
                label.append(bios)
                document.append(tokenized_text)
                document_char.append(chars)
                part_of_speech.append(sentence_pos)
            documents.append(document)
            document_chars.append(document_char)
            labels.append(label)
            mention_set.append([v for k, v in mention_dict.items()])
            part_of_speeches.append(part_of_speech)

        '''
        count = -1
        doc = documents[count]
        mention = mention_set[count]
        doc = [w for line in doc for w in line]
        #doc = ''.join(doc)
        for ment in mention:
            for mt in ment:
                print(doc[mt[0]:mt[1]+1])
                print(mt[0], mt[1])
            print('==')
        print('----')
        '''
        max_num = 0
        for mention in mention_set:
            for s in mention:
                max_num = max(max_num, len(s))

        return documents, document_chars, labels, mention_set, part_of_speeches
    
    def build_dict(self, train_data, test_data, valid_data):
        
        train_document = train_data[0] + valid_data[0] + train_data[1] + valid_data[1]
        test_document = test_data[0]
        #train_document_char = train_data[1] + valid_data[1]

        word_count = {self.PAD : 1e10, self.UNK: 1e10-1}
        for sentences in train_document:
            for sentence in sentences:
                for word in sentence:
                    if isinstance(word, str):
                        word_count[word] = word_count[word] + 1 if word in word_count else 1
                    else:
                        for char in word:
                            word_count[char] = word_count[char] + 1 if char in word_count else 1

        for sentences in test_document:
            for sentence in sentences:
                for word in sentence:
                    if isinstance(word, str):
                        if word not in word_count:
                            word_count[word] = 0
                    else:
                        for char in word:
                            if char not in word_count:
                                word_count[char] = 0
                        
        word_count = Counter(word_count).most_common()
        word_dict = {w:i for i, (w, z) in enumerate(word_count)}
        path = os.path.join(self.data_dir, 'word_dict.pkl')
        pkl.dump(word_dict, open(path, 'wb'))
        self.word_dict = word_dict
    
    def build_embedding(self, word_dict, emb_file):
        embedding = {}
        doc = open(emb_file, 'r')
        for line in tqdm(doc, total=8824331):
            line = line.split()
            word = line[0]
            if word in word_dict:
                embedding[word] = line[1:]
        vocab_size = len(word_dict)
        embedding_dim = len(list(embedding.values())[0])
        embedding_matrix = np.zeros([vocab_size, embedding_dim], dtype=np.float)
        for i, word in enumerate(word_dict):
            try:
                embedding_matrix[i] = embedding[word]
            except:
                embedding_matrix[i] = np.random.normal(embedding_dim)

        emb_path = os.path.join(self.data_dir, 'emb.pkl')
        pkl.dump(embedding_matrix, open(emb_path, 'wb'))

    def get_pos(self, sentence):
        #return ['O' for w in line]
        #print([(i,w) for i, w in enumerate(sentence)])
        sentences = ''.join(sentence)
        url = 'http://10.201.147.173:8088/?properties={"annotators":"tokenize, ssplit,  pos, lemma, ner, depparse","outputFormat":"json", "pipelineLanguage":"zh"}'
        res = rq.post(url, data=sentence.encode('utf-8')).text
        res = json.loads(res)
        tokens = [w for line in res['sentences'] for w in line['tokens']]
        poses = [w['pos'] for w in tokens]
        segs = [w['originalText'] for w in tokens]

        sentences_poses = []
        tokenized_sentence = [w for line in segs for w in line]
        # [AB A ACB]
        chars = [list(w) for w in tokenized_sentence]
        index_2_dict = {}
        #print("After", tokenized_sentence)
        #print([(i,w) for i, w in enumerate(''.join(tokenized_sentence))])
        for i, w in enumerate(tokenized_sentence):
            for j in range(len(w)):
                index_2_dict[len(index_2_dict)] = i
        assert len(sentences) == len(''.join(tokenized_sentence))
            
        return tokenized_sentence, chars, poses, index_2_dict
        
    def build_label_dict(self, content):
        res = []
        res = [w for document in tqdm(content) for sentence in document for w in sentence if w != '[UNK]']
        res = Counter(res).most_common()[:10]
        res = {word:i + 2 for i, (word, freq) in enumerate(res)}
        #res['[CLS]'] = 3
        #res['[SEP]'] = 2
        res['[UNK]'] = 1
        res['[PAD]'] = 0
        path = os.path.join(self.data_dir, 'pos_dict.pkl')
        pkl.dump(res, open(path, 'wb'))
        self.pos_dict = res

    def get_sentence_index(self, sentences, max_length):
        indexs = []
        tmp_index = []
        tmp_count = 0
        for i, w in enumerate(sentences):
            if len(w) > max_length - 2:
                w = w[:max_length - 2]
            if tmp_count + len(w) <= max_length - 2:
                tmp_index.append(i)
                tmp_count += len(w)
            else:
                indexs.append(tmp_index)
                tmp_count = len(w)
                tmp_index = [i]
        if len(tmp_index) > 0:
            indexs.append(tmp_index)
        return indexs

    def transform2indices(self, data, mode='train'):
        documents, document_chars, source_labels, mention_set, part_of_speeches = data
        input_ids, input_id_chars, input_masks, input_segments, input_labels, input_poses = [], [], [], [], [], []
        input_lengths = []
        label_dict = {'B':0, 'I':1, 'O':2}
        print("start... {}".format(mode))
        for document, document_char, source_label, source_pos in tqdm(zip(documents, document_chars, source_labels, part_of_speeches), total=len(documents)):
            sentence_ids, sentence_id_chars, sentence_masks, sentence_segments, sentence_labels, sentence_poses = [], [], [], [], [], []
            # 句子数，单词数 ,义原数
            global_max = 10
            tmp = [[w for line in document for w in line][:self.max_length - 2]]
            sentence_lengths = []

            for sentence, sentence_char, s_label, s_pos in zip(document, document_char, source_label, source_pos):
                input_pos = [w for w in s_pos]
                input_pos = [self.pos_dict[w if w in self.pos_dict else '[UNK]'] for w in input_pos]

                tokens = sentence
                sentence_lengths.append(len(tokens))
                input_id = [self.word_dict[w if w in self.word_dict else '[UNK]'] for w in tokens]
                input_id_char = [[self.word_dict[w if w in self.word_dict else '[UNK]']
                                  for w in line] for line in sentence_char]
                input_label = [label_dict[w] for w in s_label]

                padding = [0] * (self.max_length - len(input_id))

                input_id += padding
                input_label += padding
                input_pos += padding
                assert len(input_id) == self.max_length
                assert len(input_label) == self.max_length
                assert len(input_pos) == self.max_length

                sentence_ids.append(input_id)
                sentence_id_chars.append(input_id_char)
                sentence_labels.append(input_label)
                sentence_poses.append(input_pos)

            input_ids.append(sentence_ids)
            input_id_chars.append(sentence_id_chars)
            input_lengths.append(sentence_lengths)
            input_labels.append(sentence_labels)
            input_poses.append(sentence_poses)
        path = open(os.path.join(self.data_dir, '{}.pkl'.format(mode)), 'wb')
        pkl.dump((input_ids, input_lengths, input_labels, mention_set, input_poses), path)
        # pkl.dump((input_ids, input_id_chars, input_lengths, input_labels, mention_set, input_poses, input_sememes, input_sememes_nums), path)

    def manage(self):
        modes = ['train', 'test', 'valid']
        print("start building source file")
        train_data = self.execute('train')
        valid_data = self.execute('valid')
        test_data = self.execute('test')
        print("start building pos dict")
        self.build_label_dict(train_data[-1] + valid_data[-1] + test_data[-1])
        print("start building dict")

        self.build_dict(train_data, test_data, valid_data)
        print("start building embedding")
        self.build_embedding(self.word_dict,self.embedding_path)

        print("Start transforme to indices")

        self.transform2indices(train_data)
        self.transform2indices(test_data, 'test')
        self.transform2indices(valid_data, 'valid')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', default='test')
    parser.add_argument('--doc', default='doc')
    args = parser.parse_args()
    preprocessor = Preprocessor(args)
    preprocessor.manage()
