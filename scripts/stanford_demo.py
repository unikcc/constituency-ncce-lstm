import re

#from stanfordcorenlp import StanfordCoreNLP
#nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05/', lang='zh')
##nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05/')
#sentence = '我爱北京天安门！hello world'
##sentence = 'Hello World!'
#print(nlp.word_tokenize(sentence))
#print (nlp.pos_tag(sentence))
#print (nlp.ner(sentence))
#print (nlp.parse(sentence))
#print (nlp.dependency_parse(sentence))
#nlp.close()

'''
A sample code usage of the python package stanfordcorenlp to access a Stanford CoreNLP server.
Written as part of the blog post: https://www.khalidalnajjar.com/how-to-setup-and-use-stanford-corenlp-server-with-python/ 
'''

from stanfordcorenlp import StanfordCoreNLP
import logging
import json

class StanfordNLP:
    def __init__(self, host='http://172.16.133.173', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port, timeout=15000, quiet=False, logging_level=logging.DEBUG, lang='zh')   # , lang='zh' , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens


if __name__ == '__main__':
    sNLP = StanfordNLP()
    sens = "上述贷款中4000多万元被颜某用于偿还相关银行的贷款,3000多万元被用于偿还其个人高利贷及个人挥霍,给银行造成巨额损失。针对涉案的银行贷款,在王某案发后涉案银行迅速展开相关工作,确认有还款隐患的约1000多万元,其中进入清收环节的只有几百万元,但经过近2年的清收工作,目前上述涉案贷款清收工作已基本结束,没有对该行资产造成损失。"
    print("初始：", sens)
#    sentence = ' '.join(sens)
    sentence = sens
    word_lst = sNLP.word_tokenize(sentence)
    pos_lst = sNLP.pos(sentence)
    dependency_parse_lst = sNLP.dependency_parse(sentence)

    print(word_lst)
    print(pos_lst)
    print(dependency_parse_lst)
    print("---")
            
