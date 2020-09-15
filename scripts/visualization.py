#!/usr/bin/env python  
#_*_ coding:utf-8 _*_  

""" 
@author: libobo
@file: visualization.py 
@time: 20/9/14 21:29
"""


def refind_gold(self, input_ids, gold_indices, input_masks, mention_label):
    print("gold", mention_label)
    print("length", len(gold_indices))
    print("len mention", len(mention_label))
    if len(mention_label) > 13:
        return
    nomask = []
    sms = 0
    for i in range(len(input_masks)):
        sm = sum(input_masks[i])
        nomask += [j + sms for j in range(sm)][1:-1]
        sms += sm

    input_ids = input_ids.reshape(-1)
    input_ids = self.get_mention_nomask(input_ids, input_masks)
    input_ids = input_ids[nomask]

    result_html = pkl.load(open('result_html/res.pkl', 'rb'))
    # word_dict = pkl.load(open('../data/preprocessed/scripts/word_dict.pkl', 'rb'))
    res = []
    res = input_ids.tolist()
    res = self.tokenizer.convert_ids_to_tokens(res)
    # clusters = mention_label
    clusters = [gold_indices]

    i, j = 0, 0
    result_text = '<p>'
    colors = ['red', 'green', 'blue', 'yellow']
    colors = ['red'] * 4

    result_text = '<p>'
    for index, cluster in enumerate(clusters):
        i, j = 0, 0
        predict_indices = cluster
        while i < len(res) and j < len(predict_indices):
            if i < predict_indices[j][0]:
                result_text += res[i]
            elif i == predict_indices[j][0]:
                result_text += '<font color="{}">'.format(colors[index % 4]) + res[i]
                if predict_indices[j][1] == i:
                    j += 1
                    result_text += '</font>'
            elif i > predict_indices[j][0] and i < predict_indices[j][1]:
                result_text += res[i]
            elif i == predict_indices[j][1]:
                result_text += res[i] + '</font>'
                j += 1
            i += 1
        while i < len(res):
            result_text += res[i]
            i += 1
        result_text += '</p>'
        result_text += '<p></p><p>'
    result_text += '</p>'
    result_html = result_html.format(result_text)
    open('result_html/res_gold.html', 'w').write(result_html)
    print("input...")
    a = input()


def refind(self, input_ids, predict_indices, input_masks, gold='no'):
    nomask = []
    sms = 0
    for i in range(len(input_masks)):
        sm = sum(input_masks[i])
        nomask += [j + sms for j in range(sm)]
        sms += sm

    input_ids = input_ids.reshape(-1)
    input_ids = self.get_mention_nomask(input_ids, input_masks)
    input_ids = input_ids[nomask]

    result_html = pkl.load(open('res.pkl', 'rb'))
    # word_dict = pkl.load(open('../data/preprocessed/scripts/word_dict.pkl', 'rb'))
    res = []
    res = input_ids.tolist()

    d = {w: i for i, w in self.dict.items()}
    res = [d[i] for i in res]
    # res = self.tokenizer.convert_ids_to_tokens(res)

    # clusters = get_cluster(predict_indices, mention_label)
    clusters = predict_indices

    i, j = 0, 0
    result_text = '<p>'
    colors = ['fuchsia', 'yellow', 'aqua', 'lime']

    result_text = '<p>'
    for index, cluster in enumerate(clusters):
        i, j = 0, 0
        predict_indices = cluster
        while i < len(res) and j < len(predict_indices):
            if i < predict_indices[j][0]:
                result_text += res[i]
            elif i == predict_indices[j][0]:
                result_text += '<font style="background-color:{}">'.format(colors[index % 4]) + res[i]
                if predict_indices[j][1] == i:
                    j += 1
                    result_text += '</font>'
            elif i > predict_indices[j][0] and i < predict_indices[j][1]:
                result_text += res[i]
            elif i == predict_indices[j][1]:
                result_text += res[i] + '</font>'
                j += 1
            i += 1
        while i < len(res):
            result_text += res[i]
            i += 1
        result_text += '</p>'
        result_text += '<p></p><p>'
    result_text += '</p>'
    # result_text = result_text.replace('p>', 'h4>')
    if gold == 'g':
        a = str(open('res/{}.html'.format(self.valid_index), 'r').read())
        a = a.replace('ggg', '{}').format(result_text)
        open('res/{}.html'.format(self.valid_index), 'w').write(a)
    else:
        result_html = result_html.format(result_text, 'ggg')
        f = open('res/{}.html'.format(self.valid_index), 'w')
        f.write(result_html)
        f.flush()
        f.close()
