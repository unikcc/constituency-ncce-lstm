import torch
import torch.nn as nn
import pickle as pkl
from utils import get_f1_by_bio
from utils import get_f1_by_bio_nomask
from utils import get_mention_f1
from getCluster import get_cluster
from utils import evaluate_coref
from metrics_back import CorefEvaluator



#from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from pytorch_transformers.modeling_bert import BertModel, BertPreTrainedModel, BertConfig
from pytorch_transformers import BertTokenizer

class myLSTM(BertPreTrainedModel):
    """
    My lstm for text classification
    """
    def __init__(self, config, device):
        self.valid_index =  0
        # super(myClassification, self).__init__(config)
        super(myLSTM, self).__init__(config)
        # self.bert = BertModel(config, output_attentions=False, keep_multihead_output=False)
        self.bert = BertModel(config)
        #self.tokenizer = BertTokenizer.from_pretrained('/usr/local/bert/bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        self.device = device
        self.hidden_size = config.hidden_size
        self.output_size = config.num_labels
        self.dropout = nn.Dropout(0.2)
        #self.pos_lstm = nn.LSTM()

        hidden_size = self.hidden_size
        output_size = self.output_size
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.lin = nn.Linear(hidden_size * 2, hidden_size)

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.atten_linear = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Linear(hidden_size, 1),
        )
        hidden_size_low = 100

        #self.mention_linear = nn.Linear(hidden_size * 2, output_size + 1)
        self.mention_linear = nn.Sequential(
            nn.Linear( hidden_size_low+ config.pos_emb_size * 2 + config.sememe_emb_size * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
        self.mention_linear_no_pos = nn.Sequential(
            #nn.Linear( hidden_size_low + config.sememe_emb_size * 2, hidden_size),
            nn.Linear( hidden_size_low , hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )


        #self.linear = nn.Linear(hidden_size_low * 4 + config.sememe_emb_size * 8, output_size - 1)
        self.linear = nn.Linear(hidden_size_low * 4 , output_size - 1)
        self.linear_1 = nn.Linear(hidden_size, hidden_size_low)
        self.part_of_speech_embedding = nn.Embedding(config.num_pos, config.pos_emb_size)
        self.pos_lstm = nn.LSTM(config.pos_emb_size, config.pos_emb_size, bidirectional=True, batch_first=True)

        self.sememe_embedding = nn.Embedding(config.num_sememe, config.sememe_emb_size)
        self.sememe_lstm = nn.LSTM(config.sememe_emb_size, config.sememe_emb_size, bidirectional=True, batch_first=True)

    def get_label_f1(self, predict_indices, gold_indices):

        tp, fp, tn = 0, 0, 0

        predict_indices = set(predict_indices)
        gold_indices = set(gold_indices)

        for w in predict_indices:
            if w in gold_indices:
                tp += 1
        p = tp / len(predict_indices)
        r = tp / len(gold_indices)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        return f1

    def get_mention_indices(self, outputs):
        #lstm_out = self.mention_linear(lstm_out)
        if len(outputs.shape) > 1:
            outputs = [w.argmax(-1) for w in outputs ]
        bio_dict = {'B': 0, 'I': 1, 'O':2}
        bio_dict = {v:k for k, v in bio_dict.items()}
        outputs = [outputs]
        outputs = [[bio_dict[w.item()] for w in out] for out in outputs]

        indices = []
        length = 0
        for output in outputs:
            start, end = -1, -1
            for i, out in enumerate(output):
                index = i + length
                if out == 'B':
                    if start != -1:
                        indices.append((start, end))
                    start, end = index, index
                elif out == 'O':
                    if start != -1:
                        indices.append((start, end))
                    start, end = -1, -1
                else:
                    end = index
            if start != -1:
                indices.append((start, end))
            length += len(output)

        return indices

    def get_mention_emb(self, lstm_out, mention_index):
        #lstm_out = torch.cat(lstm_out, 0)
        mention_emb_list = []
        mention_start, mention_end = zip(*mention_index)
        mention_start = torch.tensor(mention_start).to(self.device)
        mention_end = torch.tensor(mention_end).to(self.device)
        mention_emb_list.append(lstm_out.index_select(0, mention_start))
        mention_emb_list.append(lstm_out.index_select(0, mention_end))

        mention_emb = torch.cat(mention_emb_list, 1)
        return mention_emb

    def get_mention_labels(self, predict_indices, gold_sets):

        mention_matrix = torch.zeros(len(predict_indices), len(predict_indices)).long().to(self.device)
        indices_dict = {w : i for i, w in enumerate(predict_indices)}
        for i in range(len(predict_indices)):
            mention_matrix[i, i] = 1
            pass
        for gold_set in gold_sets:
            for mention_0 in gold_set:
                if mention_0 not in indices_dict:
                    continue
                for mention_1 in gold_set:
                    if mention_1 not in indices_dict:
                        continue
                    s1, s2 = indices_dict[mention_0], indices_dict[mention_1]
                    mention_matrix[s1, s2] = 1
                    mention_matrix[s2, s1] = 1
        return mention_matrix

    def get_mask(self, lengths):
        tmp = lengths.cpu()
        return torch.arange(max(tmp))[None, :] < tmp[:, None]

    def get_mention_nomask(self, lstm_out, masks):
        if len(masks.shape) > 1:
            masks = masks.reshape([-1, ])
        masks = masks.nonzero().reshape(-1)

        return lstm_out[masks]

    def get_cluster(self, predict_indices, mention_label):
        if len(mention_label.shape) == 3:
            predict_indices = [tuple(w) for w in predict_indices]


        cluster = dict()

        for i in range(len(mention_label)):
            for j in range(i):
                if mention_label[i, j] == 1:
                    if predict_indices[j] not in cluster:
                        cluster[predict_indices[j]] = set()
                    if predict_indices[i] in cluster[predict_indices[j]]:
                        continue
                    cluster[predict_indices[j]].add(predict_indices[i])

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

        result_html = pkl.load(open('res.pkl', 'rb'))
        #word_dict = pkl.load(open('../data/preprocessed/scripts/word_dict.pkl', 'rb'))
        res = []
        res = input_ids.tolist()
        res = self.tokenizer.convert_ids_to_tokens(res)
        #clusters = mention_label
        clusters = [gold_indices]

        i, j = 0, 0
        result_text = '<p>'
        colors = ['red', 'green', 'blue', 'yellow']
        colors = ['red'] * 4

        result_text = '<p>'
        for index, cluster in enumerate(clusters):
            i,j = 0, 0
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
        open('res_gold.html', 'w').write(result_html)
        print("input...")
        a = input()
    def refind(self, input_ids, predict_indices, input_masks, gold='no'):
        print("length", len(predict_indices))
        nomask = []
        sms = 0
        for i in range(len(input_masks)):
            sm = sum(input_masks[i])
            nomask += [j + sms for j in range(sm)][1:-1]
            sms += sm

        input_ids = input_ids.reshape(-1)
        input_ids = self.get_mention_nomask(input_ids, input_masks)
        input_ids = input_ids[nomask]


        result_html = pkl.load(open('res.pkl', 'rb'))
        #word_dict = pkl.load(open('../data/preprocessed/scripts/word_dict.pkl', 'rb'))
        res = []
        res = input_ids.tolist()
        res = self.tokenizer.convert_ids_to_tokens(res)

        #clusters = get_cluster(predict_indices, mention_label)
        clusters = [predict_indices]
        print(clusters)

        i, j = 0, 0
        result_text = '<p>'
        colors = ['red', 'green', 'blue', 'yellow']
        colors = ['red'] * 4

        result_text = '<h4>'
        for index, cluster in enumerate(clusters):
            i,j = 0, 0
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
            result_text += '</h4>'
            result_text += '<p></p><p>'
        result_text += '</p>'
        #result_html = result_html.format(result_text)
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

    def check(self, input_ids, lengths, gold_mention):
        word_dict = pkl.load(open('../data/preprocessed/scripts/word_dict.pkl', 'rb'))
        word_dict = {v:w for w, v in word_dict.items()}
        res = []
        for input_id, length in zip(input_ids, lengths):
            tmp = input_id[:length].cpu().tolist()
            print(tmp)
            tmp = self.tokenizer.convert_ids_to_tokens(tmp)
            res+= tmp
        print(res)
        gold_mention = [w for line in gold_mention for w in line]
        for ment in gold_mention:
            print(''.join(res[ment[0]:ment[1]+1]))
            print('input..')

    def get_f1_by_bio(self, predict_bio, gold_bio):
        pass

    def delete_head_tail(self, outputs, masks):
        lengths = [w.sum().item() for w in masks]
        out = []
        start = 0
        i = 0
        while start < outputs.shape[0]:
            if lengths[i] > 2:
                out.append(outputs[start+1:start + lengths[i]-1])
            start += lengths[i]
            i += 1
        return out

    
    def get_hownet_mask(self, lengths):
        # 20, 512, 30
        max_num = max([max(w) for w in lengths.cpu().tolist()])
        res = []
        for length in lengths:
            tmp = length.cpu()
            tmp = torch.arange(max_num)[None, :] < tmp[:, None]
            res.append(tmp.float())
            #res.append(tmp)
            #res.append(torch.stack(res0, 0))
        res = torch.stack(res, 0).to(self.device)
        return res

    def forward(self, input_ids, input_masks, input_segments, input_labels, mention_sets, input_poses, input_sememes, input_sememes_nums, use_pos='pos', show_res=False, coref_evaluator=None) :
        ls = [w.sum().item() for w in input_masks]
        title_l = input_masks[0].sum() - 2
        mention_sets = [[(p[0][0].item(), p[1][0].item()) for p in k] for k in mention_sets]

        output = self.bert(input_ids, token_type_ids=input_segments, attention_mask=input_masks)[0]
        output = self.dropout(output)

        criterion = nn.CrossEntropyLoss()
        output = self.linear_1(output)
        import copy
        masks = copy.deepcopy(input_masks)
        for i in range(len(masks)):
            masks[i, masks[i].sum()-1] = 0
            masks[i, 0] = 0
        flatten_output = output.reshape([-1, output.shape[-1]])
        flatten_output_nomask = self.get_mention_nomask(flatten_output, masks)
        labels = input_labels.reshape(-1)
        labels = self.get_mention_nomask(labels, masks)
        predict_output = self.mention_linear_no_pos(flatten_output_nomask)

        loss1 = criterion(predict_output, labels)

        predict_indices = self.get_mention_indices(predict_output)
        gold_indices = self.get_mention_indices(labels)
        if len(predict_indices) == 0:
            predict_indices = gold_indices[:1]

        self.best_epoch = 10
        if self.status == 'pred':
            if self.mode == 'train':
                predict_indices = gold_indices
            else:
                a = pkl.load(open('pipe/{}_{}.pkl'.format(self.mode, self.best_epoch), 'rb'))
                predict_indices = a[self.num_index]
        flatten_output_nomask = self.dropout(flatten_output_nomask)

        mention_emb = self.get_mention_emb(flatten_output_nomask, predict_indices)
        mention_label = self.get_mention_labels(predict_indices, mention_sets)

        # mention_interaction = torch.matmul(mention_interaction, mention_emb.transpose(1, 0))
        mention_emb_r = mention_emb.unsqueeze(1)
        mention_emb_c = mention_emb.unsqueeze(0)
        mention_emb = mention_emb_c * mention_emb_r

        mention_emb = torch.cat((mention_emb, mention_emb_r + mention_emb_c), -1)

        mention_interaction = self.linear(mention_emb)

        import random

        correct_count, count = 0, 0
        new_mention_interaction = []
        new_mention_label = []
        for i in range(mention_interaction.shape[0]):
            new_mention_interaction.append(mention_interaction[i, :i + 1])
            new_mention_label.append(mention_label[i, :i + 1])
        new_mention_label = torch.cat(new_mention_label)
        new_mention_interaction = torch.cat(new_mention_interaction, 0)
        criterion = nn.CrossEntropyLoss()
        loss2 = criterion(new_mention_interaction, new_mention_label)

        if self.status[:4] == 'pipe':
            losses = loss1
        else:
            losses = loss2
        # losses = losses + tmp_loss
        filename = self.status[:5]
        if self.status[:4] == 'pipe' and self.mode != 'train':
            num_index = self.num_index
            if num_index == 0:
                pkl.dump([predict_indices], open('pipe/{}_{}.pkl'.format(self.mode, self.epoch_index), 'wb'))
            else:
                a = pkl.load(open('pipe/{}_{}.pkl'.format(self.mode, self.epoch_index), 'rb'))
                a.append(predict_indices)
                pkl.dump(a, open('pipe/{}_{}.pkl'.format(self.mode, self.epoch_index), 'wb'))

        import numpy as np
        # mention_p, mention_r, mention_f1 = get_mention_f1(np.array(new_mention_interaction.cpu().tolist()), np.array(new_mention_label.cpu().tolist()))
        #losses = losses + tmp_loss


        self.predict_mention = new_mention_interaction.cpu().tolist()
        self.gold_mention = new_mention_label.cpu().tolist()

        #clusters, mention_to_predict = getPredictCluster(predict_indices, mention_interaction)
        # evaluate_coref(predict_indices, mention_interaction, gold_mention_set, evaluator)
        pred_cluster, gold_cluster = evaluate_coref(predict_indices, mention_interaction, mention_sets, coref_evaluator)
        new_x = CorefEvaluator()
        evaluate_coref(predict_indices, mention_interaction, mention_sets, new_x)
        tmp_prf = new_x.get_prf()
        self.single_prf = (gold_cluster, tmp_prf)


        if show_res:
            pass

            i = copy.deepcopy(input_ids)
            #self.refind(input_ids, predict_indices, input_masks)
            #self.refind(i, gold_indices, input_masks, 'g')
            #self.refind_gold(input_ids, gold_indices, input_masks, mention_sets)

        #return losses, predict_output, labels
        return losses, correct_count, count, predict_output,labels, new_mention_interaction.cpu().tolist(), new_mention_label.cpu().tolist()


'''
def get_hownet_mask(lengths):
    res = []
    for length in lengths:
        tmp = length.cpu()
        tmp = torch.arange(max(tmp))[None, :] < tmp[:, None]
        res.append(tmp.int())
    res = torch.stack(res, 0)
    return res

device = torch.device('cuda:1')

if __name__ == '__main__':
    length = torch.tensor([[3,2,1,0], [2,1,3,0]]).to(device)
    net = get_hownet_mask(length)
    print(net)
'''
