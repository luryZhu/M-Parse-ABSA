from tqdm import *
from nltk.tokenize import TreebankWordTokenizer
from allennlp.predictors.predictor import Predictor
from stanfordcorenlp import StanfordCoreNLP
from lxml import etree
import json
import os
import stanza


# PATHS
MODELS_DIR = './models'
biaffine_path = os.path.join(
    MODELS_DIR, "biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
corenlp_path =r'D:\nlp\stanford-corenlp-4.5.1'


class Parser:

    def parse_raw(self, file_path):
        with open(file_path, 'r') as f:
            sentences = f.readlines()
        docs = []
        print('Predicting dependency information...')
        for i in tqdm(range(len(sentences))):
            docs.append(self.predict(sentence=sentences[i]))

        return docs

    def parse(self, file_path):
        docs = self.parse_raw(file_path)
        sentences = [self.format(doc) for doc in docs]
        return sentences

    def write(self, sentences, origin_file):
        json_data = []
        tk = TreebankWordTokenizer()
        mismatch_counter = 0
        idx = 0
        with open(origin_file, 'rb') as fopen:
            raw = fopen.read()
            root = etree.fromstring(raw)
            for sentence in root:
                example = dict()
                example["sentence"] = sentence.find('text').text  # 从raw文件取出原始句子

                # for RAN
                terms = sentence.find('aspectTerms')
                if terms is None:
                    continue

                example['tokens'] = sentences[idx]['tokens']  # 取出解析后的 tokens 数组
                example['tags'] = sentences[idx]['tags']  # 取出解析后的 tokens 对应的词性标签
                example['predicted_dependencies'] = sentences[idx]['predicted_dependencies']  # 取出解析后的以 tokens 为终点的边的依赖关系
                example['predicted_heads'] = sentences[idx]['predicted_heads']  # 上述边的 起始节点对应的 token 编号，从1开始编号
                example['dependencies'] = sentences[idx]['dependencies']  # 以边依赖关系，起始点，终点编号（注意从1开始计数）为一组

                example["aspect_sentiment"] = []  # 存储方面词，对应情感极性对
                example['from_to'] = []  # # 方面词起始和结束位置偏移值，以char为单位

                for c in terms:
                    if c.attrib['polarity'] == 'conflict':  # 去除 confilct，只考虑 positive negative neural
                        continue
                    target = c.attrib['term']
                    example["aspect_sentiment"].append((target, c.attrib['polarity']))  # 将方面词和情感极性成对加入

                    # index in strings, we want index in tokens
                    left_index = int(c.attrib['from'])  # 方面词起始位置偏移值，按字母
                    right_index = int(c.attrib['to'])  # 方面词结束位置偏移值，按字母

                    left_word_offset = len(tk.tokenize(example['sentence'][:left_index]))  # 方面词起始位置偏移值，按单词
                    to_word_offset = len(tk.tokenize(example['sentence'][:right_index]))  # 方面词结束位置偏移值，按单词

                    example['from_to'].append((left_word_offset, to_word_offset))
                if len(example['aspect_sentiment']) == 0:
                    idx += 1
                    continue
                json_data.append(example)
                idx += 1
        extended_filename = origin_file.replace('.xml', '_'+self.name+'_depparsed.json')
        with open(extended_filename, 'w') as f:
            json.dump(json_data, f)
        print('done', len(json_data))
        print(idx)


'''
    Biaffine Parser 使用 AllenNLP
'''

class Biaffine_Parser(Parser):
    def __init__(self):
        self.predictor = Predictor.from_path(biaffine_path)
        self.name = 'Biaffine'

    # 解析句法依存树
    def predict(self, sentence):
        return self.predictor.predict(sentence)

    # 将AllenNLP产生的解析树转换结构
    def format(self, doc):
        '''
            Format annotation: sentence of keys
                                        - tokens
                                        - tags
                                        - predicted_dependencies
                                        - predicted_heads
                                        - dependencies
            '''
        sentence = {}
        sentence['tokens'] = doc['words']
        sentence['tags'] = doc['pos']
        # sentence['energy'] = doc['energy']
        predicted_dependencies = doc['predicted_dependencies']
        predicted_heads = doc['predicted_heads']
        sentence['predicted_dependencies'] = doc['predicted_dependencies']
        sentence['predicted_heads'] = doc['predicted_heads']
        sentence['dependencies'] = []
        for idx, item in enumerate(predicted_dependencies):
            dep_tag = item
            frm = predicted_heads[idx]
            to = idx + 1
            sentence['dependencies'].append([dep_tag, frm, to])

        return sentence



'''
    CoreNLP
'''

class CoreNLP_Parser(Parser):
    def __init__(self):
        self.nlp = StanfordCoreNLP(corenlp_path)
        self.name = 'CoreNLP'

    def predict(self, sentence):
        nlp = self.nlp
        doc = {}
        doc['tokens'] = nlp.word_tokenize(sentence)
        doc['tags'] = [x[1] for x in nlp.pos_tag(sentence)]
        doc['dependencies'] = list(nlp.dependency_parse(sentence))
        return doc

    def format(self, doc):
        '''
        Format annotation: sentence of keys
                                    - tokens
                                    - tags
                                    - predicted_dependencies
                                    - predicted_heads
                                    - dependencies
        '''
        sentence = doc
        sentence['dependencies'].sort(key=lambda x: x[2])
        sentence['predicted_dependencies'] = [x[0] for x in sentence['dependencies']]
        sentence['predicted_heads'] = [x[1] for x in sentence['dependencies']]
        return sentence


class Stanza_Parser(Parser):
    def __init__(self):
        stanza.download('en')
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
        self.name = 'Stanza'

    def predict(self, sentence):
        nlp = self.nlp
        doc = {}
        doc['tokens'] = nlp.word_tokenize(sentence)
        doc['tags'] = [x[1] for x in nlp.pos_tag(sentence)]
        doc['dependencies'] = list(nlp.dependency_parse(sentence))
        return doc

    def format(self, doc):
        '''
        Format annotation: sentence of keys
                                    - tokens
                                    - tags
                                    - predicted_dependencies
                                    - predicted_heads
                                    - dependencies
        '''
        sentence = doc
        sentence['dependencies'].sort(key=lambda x: x[2])
        sentence['predicted_dependencies'] = [x[0] for x in sentence['dependencies']]
        sentence['predicted_heads'] = [x[1] for x in sentence['dependencies']]
        return sentence




def get_parsers(args):
    parsers = [
        Biaffine_Parser(),
        CoreNLP_Parser()
    ]
    return parsers

