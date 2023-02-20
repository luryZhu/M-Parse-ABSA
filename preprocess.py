import os
import argparse
import logging
import json
from copy import deepcopy
from nltk import word_tokenize
from collections import Counter, defaultdict
import pickle
from dataset import ASBA_Depparsed_Dataset
import linecache
import numpy as np
import torch


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--dataset_name', type=str, default='rest',
                        choices=['rest', 'laptop', 'twitter'],
                        help='Choose absa dataset.')
    parser.add_argument('--parser_name', type=str, default='CoreNLP',
                        choices=['CoreNLP', 'Biaffine', 'Stanza'],
                        help='Choose dependency parser.')
    parser.add_argument('--dataset_dir', type=str, default='data/depparsed',
                        choices=['rest', 'laptop', 'twitter'],
                        help='Directory to load depparsed data.')
    parser.add_argument('--output_dir', type=str, default='data/output',
                        help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')
    return parser.parse_args()

'''
    load
'''
def load_depparsed(file_path):
    # with open(file_path, 'r',encoding='UTF-8') as f:
    with open(file_path, 'r',encoding='UTF-8') as f:
        data = json.load(f)
        return data


def get_depparsed_file(args):
    dataset_map = {
        'train': {
            'rest': 'Restaurants_Train_v2',
            'laptop': 'Laptops_Train_v2'
        },
        'test': {
            'rest': 'Restaurants_Test_Gold',
            'laptop': 'Laptops_Test_Gold'
        }
    }
    train_file = os.path.join(args.dataset_dir,
                              args.dataset_name,
                              dataset_map['train'][args.dataset_name]+'_'+args.parser_name+'_depparsed.json')
    test_file = os.path.join(args.dataset_dir,
                             args.dataset_name,
                             dataset_map['test'][args.dataset_name]+'_'+args.parser_name+'_depparsed.json')

    return train_file, test_file


def load_dataset(args):
    train_file, test_file=get_depparsed_file(args)

    train = list(load_depparsed(train_file))
    logger.info('# Read %s Train set: %d', args.dataset_name, len(train))

    test = list(load_depparsed(test_file))
    logger.info("# Read %s Test set: %d", args.dataset_name, len(test))
    return train, test


'''
    reshape
'''

def reshape(as_start, as_end, dependencies, multi_hop=True, add_non_connect=True, tokens=None, max_hop = 5):
    '''
    重构依赖树，以方面词为根节点
        as_start        方面词起始坐标
        as_end          方面词结束坐标
        dependencies    依赖树的所有边
        multi_hop       是否添加 n_con
        add_non_connect 是否添加 non_connect
        tokens          分类
        max_hop         n_con 最大跳数，大于的剪枝
    '''
    dep_tag = []    # 边的标注
    dep_idx = []    # 终点坐标（从0开始）
    dep_dir = []    # 边的方向（1:方面词Root->idx, 2:idx->Root, 0:空）

    # 与方面词直连的节点
    for i in range(as_start, as_end):
        # 遍历依赖树种所有边
        for dep in dependencies:
            # 如果边的起点是方面词
            if i == dep[1] - 1:
                # 如果边的终点不是方面词，并且不是根节点，并且终点未被收录
                if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                        # 标注不为punct，直接插入，否则将punct关系替换为<pad>
                        dep_tag.append(dep[0])
                        dep_dir.append(1)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[2] - 1)
            # 如果边的终点是方面词
            elif i == dep[2] - 1:
                # 如果边的起点不是方面词，并且不是根节点，并且终点未被收录
                if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(2)
                    else: # punct 替换为<pad>
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[1] - 1)

    if multi_hop: # 广度优先搜索
        current_hop = 2
        added = True
        while current_hop <= max_hop and len(dep_idx) < len(tokens) and added:
            added = False
            dep_idx_temp = deepcopy(dep_idx)    # 暂存已经处理过的树节点
            for i in dep_idx_temp: # 遍历所有已经处理过的边的终点
                for dep in dependencies: # 遍历所有边
                    if i == dep[1] - 1: # 如果点是边的起点
                        # not root, not aspect 逻辑同上，边的终点不是方面词，不是根，且未被处理过才能加入解集
                        if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop)) # 根与当前节点的跳数就是current_hop
                                dep_dir.append(1)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                            dep_idx.append(dep[2] - 1)
                            added = True
                    elif i == dep[2] - 1:
                        # not root, not aspect
                        if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop))
                                dep_dir.append(2)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                            dep_idx.append(dep[1] - 1)
                            added = True
            current_hop += 1

    if add_non_connect:  # 没有被上面代码统计到的边全部剪枝，标注未non-connect（方面词内部的边和距离超过max_hop的边）
        for idx, token in enumerate(tokens):
            if idx not in dep_idx and (idx < as_start or idx >= as_end):
                dep_tag.append('non-connect')
                dep_dir.append(0)
                dep_idx.append(idx)

    # 将 方面词内部的边置为<pad>
    for idx, token in enumerate(tokens):
        if idx not in dep_idx:
            dep_tag.append('<pad>')
            dep_dir.append(0)
            dep_idx.append(idx)

    index = [i[0] for i in sorted(enumerate(dep_idx), key=lambda x:x[1])]
    dep_tag = [dep_tag[i] for i in index]
    dep_idx = [dep_idx[i] for i in index]
    dep_dir = [dep_dir[i] for i in index]

    assert len(tokens) == len(dep_idx), 'length wrong'
    return dep_tag, dep_idx, dep_dir


def reshape_and_merge(as_start, as_end, trees, multi_hop=True, add_non_connect=True, tokens=None, max_hop = 5):
    '''
    重构依赖树，以方面词为根节点
        as_start        方面词起始坐标
        as_end          方面词结束坐标
        dependencies    依赖树的所有边
        multi_hop       是否添加 n_con
        add_non_connect 是否添加 non_connect
        tokens          分类
        max_hop         n_con 最大跳数，大于的剪枝
    '''
    dep_tag = []    # 边的标注
    dep_idx = []    # 终点坐标（从0开始）
    dep_dir = []    # 边的方向（1:方面词Root->idx, 2:idx->Root, 0:空）
    dep_hop = []    # 边的距离


    # 与方面词直连的节点
    for i in range(as_start, as_end):
        for dependencies in trees:
            # 遍历依赖树种所有边
            for dep in dependencies:
                # 如果边的起点是方面词
                if i == dep[1] - 1:
                    # 如果边的终点不是方面词，并且不是根节点
                    if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0:
                        # 如果终点未被收录
                        if dep[2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                                # 标注不为punct，直接插入，否则将punct关系替换为<pad>
                                dep_tag.append(dep[0])
                                dep_dir.append(1)
                                dep_hop.append(1)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                                dep_hop.append(0)
                            dep_idx.append(dep[2] - 1)
                        # 如果终点已被收录
                        else:
                            list.index(dep[2] - 1)
                # 如果边的终点是方面词
                elif i == dep[2] - 1:
                    # 如果边的起点不是方面词，并且不是根节点，并且起点未被收录
                    if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                        if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                            dep_tag.append(dep[0])
                            dep_dir.append(2)
                            dep_hop.append(1)
                        else: # punct 替换为<pad>
                            dep_tag.append('<pad>')
                            dep_dir.append(0)
                            dep_hop.append(0)
                        dep_idx.append(dep[1] - 1)

    if multi_hop: # 广度优先搜索
        current_hop = 2
        added = True
        while current_hop <= max_hop and len(dep_idx) < len(tokens) and added:
            added = False
            dep_idx_temp = deepcopy(dep_idx)    # 暂存已经处理过的树节点
            for i in dep_idx_temp: # 遍历所有已经处理过的边的终点
                for dep in dependencies: # 遍历所有边
                    if i == dep[1] - 1: # 如果点是边的起点
                        # not root, not aspect 逻辑同上，边的终点不是方面词，不是根，且未被处理过才能加入解集
                        if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop)) # 根与当前节点的跳数就是current_hop
                                dep_dir.append(1)
                                dep_hop.append(current_hop)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                                dep_hop.append(0)
                            dep_idx.append(dep[2] - 1)
                            added = True
                    elif i == dep[2] - 1:
                        # not root, not aspect
                        if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop))
                                dep_dir.append(2)
                                dep_hop.append(current_hop)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                                dep_hop.append(0)
                            dep_idx.append(dep[1] - 1)
                            added = True
            current_hop += 1

    if add_non_connect:  # 没有被上面代码统计到的边全部剪枝，标注未non-connect（方面词内部的边和距离超过max_hop的边）
        for idx, token in enumerate(tokens):
            if idx not in dep_idx and (idx < as_start or idx >= as_end):
                dep_tag.append('non-connect')
                dep_dir.append(0)
                dep_idx.append(idx)

    # 将 方面词内部的边置为<pad>
    for idx, token in enumerate(tokens):
        if idx not in dep_idx:
            dep_tag.append('<pad>')
            dep_dir.append(0)
            dep_idx.append(idx)

    index = [i[0] for i in sorted(enumerate(dep_idx), key=lambda x:x[1])]
    dep_tag = [dep_tag[i] for i in index]
    dep_idx = [dep_idx[i] for i in index]
    dep_dir = [dep_dir[i] for i in index]

    assert len(tokens) == len(dep_idx), 'length wrong'
    return dep_tag, dep_idx, dep_dir

def get_reshaped_data(input_data, args):
    logger.info('*** Start processing data(unrolling and reshaping) ***')
    sentiments_map = {'negative': 0, 'positive': 1, 'neutral': 2}
    reshaped_data = []
    for e in input_data:
        e['tokens'] = [x.lower() for x in e['tokens']] # token 转为小写
        pos_class = e['tags']  # 词性标注
        aspects = []        # 方面词
        froms = []          # 方面词 起点
        tos = []            # 方面词 终点
        sentiments = []     # 情感极性
        # 重构后的依赖树，以方面词为root
        dep_tags = []       # 依存关系 标注，新增<pad>
        dep_index = []      # 依存关系 终点坐标（从0开始）
        dep_dirs = []       # 依存关系 方向 0:<pad> 1:root->idx 2:idx->root

        for i in range(len(e['aspect_sentiment'])):
            aspect = e['aspect_sentiment'][i][0].lower()
            # 对方面词做tokenization
            aspect = word_tokenize(aspect)
            sentiment = sentiments_map[e['aspect_sentiment'][i][1]]
            frm = e['from_to'][i][0]
            to = e['from_to'][i][1]

            aspects.append(aspect)
            sentiments.append(sentiment)
            froms.append(frm)
            tos.append(to)

            dep_tag, dep_idx, dep_dir = reshape(frm, to, e['dependencies'],
                                                multi_hop=args.multi_hop,
                                                add_non_connect=args.add_non_connect,
                                                tokens=e['tokens'], max_hop=args.max_hop)

            dep_tags.append(dep_tag)
            dep_index.append(dep_idx)
            dep_dirs.append(dep_dir)

            reshaped_data.append(
                {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspect': aspect,
                 'sentiment': sentiment,
                 'predicted_dependencies': e['predicted_dependencies'], 'predicted_heads': e['predicted_heads'],
                 'from': frm, 'to': to, 'dep_tag': dep_tag, 'dep_idx': dep_idx, 'dep_dir': dep_dir,
                 'dependencies': e['dependencies']})

    logger.info('Total sentiment counter: %s', len(reshaped_data))
    return reshaped_data


def build_pos_tag_vocab(data, vocab_size=1000, min_freq=1):
    """
    Part of speech tags vocab.
    """
    counter = Counter()
    for d in data:
        tags = d['tags']
        counter.update(tags)

    itos = ['<pad>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


def _default_unk_index():
    return 1


def build_dep_tag_vocab(data, vocab_size=1000, min_freq=0):
    counter = Counter()
    for d in data:
        tags = d['dep_tag']
        counter.update(tags)

    itos = ['<pad>', '<unk>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        if word == '<pad>':
            continue
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


def build_text_vocab(data, vocab_size=100000, min_freq=2):
    counter = Counter()
    for d in data:
        s = d['sentence']
        counter.update(s)

    itos = ['[PAD]', '[UNK]']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


def load_glove_embedding(word_list, glove_dir, uniform_scale, dimension_size):
    glove_words = []
    with open(os.path.join(glove_dir, 'glove.840B.300d.txt'), 'r',encoding="utf-8") as fopen:
        for line in fopen:
            glove_words.append(line.strip().split(' ')[0])
    word2offset = {w: i for i, w in enumerate(glove_words)}
    word_vectors = []
    for word in word_list:
        if word in word2offset:
            line = linecache.getline(os.path.join(
                glove_dir, 'glove.840B.300d.txt'), word2offset[word]+1)
            assert(word == line[:line.find(' ')].strip())
            word_vectors.append(np.fromstring(
                line[line.find(' '):].strip(), sep=' ', dtype=np.float32))
        elif word == '<pad>':
            word_vectors.append(np.zeros(dimension_size, dtype=np.float32))
        else:
            word_vectors.append(
                np.random.uniform(-uniform_scale, uniform_scale, dimension_size))
    return word_vectors



def get_vocabs(data, args):
    '''
    创建token、postag、deprel的词表，并持久化缓存
    '''
    pkls_path = os.path.join(args.output_dir, 'pkls')
    if not os.path.exists(pkls_path):
        os.makedirs(pkls_path)

    # 创建/加载 word vocab 和 glove embeddings.
    # Elmo and bert 有自己的vocab和embedding，可以不考虑
    if args.embedding_type == 'glove':
        cached_word_vocab_file = os.path.join(
            pkls_path, 'cached_{}_{}_word_vocab.pkl'.format(args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vocab_file):
            logger.info('Loading word vocab from %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'rb') as f:
                word_vocab = pickle.load(f)
        else:
            logger.info('Creating word vocab from dataset %s',
                        args.dataset_name)
            word_vocab = build_text_vocab(data)
            logger.info('Word vocab size: %s', word_vocab['len'])
            logging.info('Saving word vocab to %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'wb') as f:
                pickle.dump(word_vocab, f, -1)

        cached_word_vecs_file = os.path.join(pkls_path, 'cached_{}_{}_word_vecs.pkl'.format(
            args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vecs_file):
            logger.info('Loading word vecs from %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'rb') as f:
                word_vecs = pickle.load(f)
        else:
            logger.info('Creating word vecs from %s', args.glove_dir)
            word_vecs = load_glove_embedding(
                word_vocab['itos'], args.glove_dir, 0.25, args.embedding_dim)
            logger.info('Saving word vecs to %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'wb') as f:
                pickle.dump(word_vecs, f, -1)
    else:
        word_vocab = None
        word_vecs = None
    # word_vocab = None
    # word_vecs = None

    # Build vocab of dependency tags
    cached_dep_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_{}_dep_tag_vocab.pkl'.format(args.dataset_name, args.parser_name))
    if os.path.exists(cached_dep_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'rb') as f:
            dep_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        dep_tag_vocab = build_dep_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',
                    dep_tag_vocab['len'], cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'wb') as f:
            pickle.dump(dep_tag_vocab, f, -1)

    # Build vocab of part of speech tags.
    cached_pos_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_{}_pos_tag_vocab.pkl'.format(args.dataset_name, args.parser_name))
    if os.path.exists(cached_pos_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'rb') as f:
            pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        pos_tag_vocab = build_pos_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',
                    pos_tag_vocab['len'], cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(pos_tag_vocab, f, -1)

    return word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab


# def get_merged_data(reshaped_data):

def my_load_dataset(args):
    train_file = os.path.join(args.dataset_dir, args.dataset_name, 'py_train',
                              'py_' + args.parser_name + '.json')
    test_file = os.path.join(args.dataset_dir, args.dataset_name, 'py_test',
                              'py_' + args.parser_name + '.json')

    train = list(load_depparsed(train_file))
    logger.info('# Read %s Train set: %d', args.dataset_name, len(train))

    test = list(load_depparsed(test_file))
    logger.info("# Read %s Test set: %d", args.dataset_name, len(test))
    return train, test



def preprocess(args):
    # train, test = load_dataset(args)
    # train_reshaped = get_reshaped_data(train, args)
    # test_reshaped = get_reshaped_data(test, args)

    train_reshaped, test_reshaped = my_load_dataset(args)

    logger.info('****** After reshaping ******')
    logger.info('Train set size: %s', len(train_reshaped))
    logger.info('Test set size: %s,', len(test_reshaped))

    # 创建词表(part of speech, dep_tag) 并持久化存储 pickles.
    word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab = get_vocabs(
        train_reshaped + test_reshaped, args)

    logger.info('****** After getting vocab ******')

    if args.embedding_type == 'glove':
        embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32))
        args.glove_embedding = embedding

    # 创建 Dataset
    train_dataset = ASBA_Depparsed_Dataset(
        train_reshaped, args, word_vocab, dep_tag_vocab, pos_tag_vocab)
    test_dataset = ASBA_Depparsed_Dataset(
        test_reshaped, args, word_vocab, dep_tag_vocab, pos_tag_vocab)
    logger.info('****** After building dataset ******')

    return train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab

def load_dataset_infer(args):
    train_file, test_file=get_depparsed_file(args)

    train = list(load_depparsed(train_file))
    logger.info('# Read %s Infer set: %d', args.dataset_name, len(train))

    test = list(load_depparsed(test_file))
    logger.info("# Read %s Test set: %d", args.dataset_name, len(test))
    return train, test

def preprocess_infer(args):
    train, test = load_dataset(args)
    train_reshaped = get_reshaped_data(train, args)
    test_reshaped = get_reshaped_data(test, args)

    logger.info('****** After reshaping ******')
    logger.info('Train set size: %s', len(train_reshaped))
    logger.info('Test set size: %s,', len(test_reshaped))

    # 创建词表(part of speech, dep_tag) 并持久化存储 pickles.
    word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab = get_vocabs(
        train_reshaped + test_reshaped, args)

    logger.info('****** After getting vocab ******')

    # 创建 Dataset
    train_dataset = ASBA_Depparsed_Dataset(
        train_reshaped, args, word_vocab, dep_tag_vocab, pos_tag_vocab)
    test_dataset = ASBA_Depparsed_Dataset(
        test_reshaped, args, word_vocab, dep_tag_vocab, pos_tag_vocab)
    logger.info('****** After building dataset ******')

    return train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab


def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    args=parse_args()
    preprocess(args)


if __name__ == '__main__':
    main()