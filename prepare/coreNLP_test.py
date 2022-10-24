from stanfordcorenlp import StanfordCoreNLP
import json
import os
import argparse


# PATHS
DATA_DIR = './data'
data_path = os.path.join(
    DATA_DIR, "semeval14")

# ARGUMENTS
def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--data_path', type=str, default=data_path,
                        help='Directory of where semeval14 or twiiter data held.')
    return parser.parse_args()

def dependencies2format(doc):  # doc.sentences[i]
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


# 将解析后的信息存入json
def syntaxInfo2json(sentences, origin_file):
    json_data = []
    tk = TreebankWordTokenizer()
    mismatch_counter = 0
    idx = 0
    with open(origin_file, 'rb') as fopen:
        raw = fopen.read()
        root = etree.fromstring(raw)
        for sentence in root:
            example = dict()
            example["sentence"] = sentence.find('text').text    # 从raw文件去除原始句子

            # for RAN
            terms = sentence.find('aspectTerms')
            if terms is None:
                continue

            example['tokens'] = sentences[idx]['tokens']    # 取出解析后的 tokens 数组
            example['tags'] = sentences[idx]['tags']    # 取出解析后的 tokens 对应的词性标签
            example['predicted_dependencies'] = sentences[idx]['predicted_dependencies']    # 取出解析后的以 tokens 为终点的边的依赖关系
            example['predicted_heads'] = sentences[idx]['predicted_heads']  # 上述边的 起始节点对应的 token 编号，从1开始编号
            example['dependencies'] = sentences[idx]['dependencies'] # 以边依赖关系，起始点，终点编号（注意从1开始计数）为一组

            example["aspect_sentiment"] = []    # 存储方面词，对应情感极性对
            example['from_to'] = []  # # 方面词起始和结束位置偏移值，以char为单位

            for c in terms:
                if c.attrib['polarity'] == 'conflict':  # 去除 confilct，只考虑 positive negative neural
                    continue
                target = c.attrib['term']
                example["aspect_sentiment"].append((target, c.attrib['polarity']))  # 将方面词和情感极性成对加入

                # index in strings, we want index in tokens
                left_index = int(c.attrib['from'])  # 方面词起始位置偏移值，按字母
                right_index = int(c.attrib['to'])   # 方面词结束位置偏移值，按字母

                left_word_offset = len(tk.tokenize(example['sentence'][:left_index]))   # 方面词起始位置偏移值，按单词
                to_word_offset = len(tk.tokenize(example['sentence'][:right_index]))   # 方面词结束位置偏移值，按单词

                example['from_to'].append((left_word_offset, to_word_offset))
            if len(example['aspect_sentiment']) == 0:
                idx += 1
                continue
            json_data.append(example)
            idx += 1
    extended_filename = origin_file.replace('.xml', '_biaffine_depparsed.json')
    with open(extended_filename, 'w') as f:
        json.dump(json_data, f)
    print('done', len(json_data))
    print(idx)


def get_dependencies(file_path, predictor):
    docs = text2docs(file_path, predictor)
    sentences = [dependencies2format(doc) for doc in docs]
    return sentences


def main():
    args = parse_args()
    predictor = StanfordCoreNLP(r'D:\nlp\stanford-corenlp-4.5.1')

    # 读入 txt
    data = [('Restaurants_Train_v2.xml', 'Restaurants_Test_Gold.xml'),
            ('Laptop_Train_v2.xml', 'Laptops_Test_Gold.xml')]
    for train_file, test_file in data:
        # 获取解析树并存储

        # txt -> json
        train_sentences = get_dependencies(
            os.path.join(args.data_path, train_file.replace('.xml', '_text.txt')), predictor)
        test_sentences = get_dependencies(os.path.join(
            args.data_path, test_file.replace('.xml', '_text.txt')), predictor)

        print(len(train_sentences), len(test_sentences))

        syntaxInfo2json(train_sentences, os.path.join(args.data_path, train_file))
        syntaxInfo2json(test_sentences, os.path.join(args.data_path, test_file))


if __name__ == "__main__":
    main()



