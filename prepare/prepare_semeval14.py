import os
import argparse
from lxml import etree
from parsers import get_parsers


# PATHS
MODELS_DIR = './models'
model_path = os.path.join(
    MODELS_DIR, "biaffine-dependency-parser-ptb-2020.04.06.tar.gz")

DATA_DIR = './data'
data_path = os.path.join(
    DATA_DIR, "semeval14")


# ARGUMENTS
def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--model_path', type=str, default=model_path,
                        help='Path to biaffine dependency parser.')
    parser.add_argument('--data_path', type=str, default=data_path,
                        help='Directory of where semeval14 or twiiter data held.')
    return parser.parse_args()


# 将原始的XML数据转换为txt格式
def xml2txt(file_path):
    output = file_path.replace('.xml', '_text.txt')
    sent_list = []
    with open(file_path, 'rb') as f:
        raw = f.read()
        root = etree.fromstring(raw)
        for sentence in root:
            sent = sentence.find('text').text
            terms = sentence.find('aspectTerms')
            if terms is None:
                continue
            if terms:
                sent_list.append(sent)
    with open(output, 'w', encoding='utf-8') as f:
        for s in sent_list:
            f.write(s+'\n')
    print('processed', len(sent_list), 'of', file_path)


def main():
    args = parse_args()
    parsers = get_parsers(args)
    # predictor = Predictor.from_path(args.model_path)

    # 读入数据并提取句子为txt
    # data = [('Restaurants_Train_v2.xml', 'Restaurants_Test_Gold.xml'),
    #         ('Laptop_Train_v2.xml', 'Laptops_Test_Gold.xml')
    data = [('train.xml', 'test.xml')]
    for train_file, test_file in data:
        # xml -> txt
        xml2txt(os.path.join(args.data_path, train_file))
        xml2txt(os.path.join(args.data_path, test_file))

        # 获取解析树并存储
        # txt -> json
        for parser in parsers:
            train_sentences = parser.parse(
                os.path.join(args.data_path, train_file.replace('.xml', '_text.txt')))
            test_sentences = parser.parse(os.path.join(
                args.data_path, test_file.replace('.xml', '_text.txt')))

            print(len(train_sentences), len(test_sentences))

            parser.write(train_sentences, os.path.join(args.data_path, train_file))
            parser.write(test_sentences, os.path.join(args.data_path, test_file))


if __name__ == "__main__":
    main()
