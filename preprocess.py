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
    with open(file_path, 'r') as f:
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
    ??????????????????????????????????????????
        as_start        ?????????????????????
        as_end          ?????????????????????
        dependencies    ?????????????????????
        multi_hop       ???????????? n_con
        add_non_connect ???????????? non_connect
        tokens          ??????
        max_hop         n_con ??????????????????????????????
    '''
    dep_tag = []    # ????????????
    dep_idx = []    # ??????????????????0?????????
    dep_dir = []    # ???????????????1:?????????Root->idx, 2:idx->Root, 0:??????

    # ???????????????????????????
    for i in range(as_start, as_end):
        # ???????????????????????????
        for dep in dependencies:
            # ??????????????????????????????
            if i == dep[1] - 1:
                # ????????????????????????????????????????????????????????????????????????????????????
                if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                        # ????????????punct???????????????????????????punct???????????????<pad>
                        dep_tag.append(dep[0])
                        dep_dir.append(1)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[2] - 1)
            # ??????????????????????????????
            elif i == dep[2] - 1:
                # ????????????????????????????????????????????????????????????????????????????????????
                if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(2)
                    else: # punct ?????????<pad>
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[1] - 1)

    if multi_hop: # ??????????????????
        current_hop = 2
        added = True
        while current_hop <= max_hop and len(dep_idx) < len(tokens) and added:
            added = False
            dep_idx_temp = deepcopy(dep_idx)    # ?????????????????????????????????
            for i in dep_idx_temp: # ??????????????????????????????????????????
                for dep in dependencies: # ???????????????
                    if i == dep[1] - 1: # ????????????????????????
                        # not root, not aspect ?????????????????????????????????????????????????????????????????????????????????????????????
                        if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop)) # ?????????????????????????????????current_hop
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

    if add_non_connect:  # ????????????????????????????????????????????????????????????non-connect???????????????????????????????????????max_hop?????????
        for idx, token in enumerate(tokens):
            if idx not in dep_idx and (idx < as_start or idx >= as_end):
                dep_tag.append('non-connect')
                dep_dir.append(0)
                dep_idx.append(idx)

    # ??? ???????????????????????????<pad>
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
    ??????????????????????????????????????????
        as_start        ?????????????????????
        as_end          ?????????????????????
        dependencies    ?????????????????????
        multi_hop       ???????????? n_con
        add_non_connect ???????????? non_connect
        tokens          ??????
        max_hop         n_con ??????????????????????????????
    '''
    dep_tag = []    # ????????????
    dep_idx = []    # ??????????????????0?????????
    dep_dir = []    # ???????????????1:?????????Root->idx, 2:idx->Root, 0:??????
    dep_hop = []    # ????????????


    # ???????????????????????????
    for i in range(as_start, as_end):
        for dependencies in trees:
            # ???????????????????????????
            for dep in dependencies:
                # ??????????????????????????????
                if i == dep[1] - 1:
                    # ?????????????????????????????????????????????????????????
                    if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0:
                        # ????????????????????????
                        if dep[2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                                # ????????????punct???????????????????????????punct???????????????<pad>
                                dep_tag.append(dep[0])
                                dep_dir.append(1)
                                dep_hop.append(1)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                                dep_hop.append(0)
                            dep_idx.append(dep[2] - 1)
                        # ????????????????????????
                        else:
                            list.index(dep[2] - 1)
                # ??????????????????????????????
                elif i == dep[2] - 1:
                    # ????????????????????????????????????????????????????????????????????????????????????
                    if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                        if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                            dep_tag.append(dep[0])
                            dep_dir.append(2)
                            dep_hop.append(1)
                        else: # punct ?????????<pad>
                            dep_tag.append('<pad>')
                            dep_dir.append(0)
                            dep_hop.append(0)
                        dep_idx.append(dep[1] - 1)

    if multi_hop: # ??????????????????
        current_hop = 2
        added = True
        while current_hop <= max_hop and len(dep_idx) < len(tokens) and added:
            added = False
            dep_idx_temp = deepcopy(dep_idx)    # ?????????????????????????????????
            for i in dep_idx_temp: # ??????????????????????????????????????????
                for dep in dependencies: # ???????????????
                    if i == dep[1] - 1: # ????????????????????????
                        # not root, not aspect ?????????????????????????????????????????????????????????????????????????????????????????????
                        if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop)) # ?????????????????????????????????current_hop
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

    if add_non_connect:  # ????????????????????????????????????????????????????????????non-connect???????????????????????????????????????max_hop?????????
        for idx, token in enumerate(tokens):
            if idx not in dep_idx and (idx < as_start or idx >= as_end):
                dep_tag.append('non-connect')
                dep_dir.append(0)
                dep_idx.append(idx)

    # ??? ???????????????????????????<pad>
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
        e['tokens'] = [x.lower() for x in e['tokens']] # token ????????????
        pos_class = e['tags']  # ????????????
        aspects = []        # ?????????
        froms = []          # ????????? ??????
        tos = []            # ????????? ??????
        sentiments = []     # ????????????
        # ???????????????????????????????????????root
        dep_tags = []       # ???????????? ???????????????<pad>
        dep_index = []      # ???????????? ??????????????????0?????????
        dep_dirs = []       # ???????????? ?????? 0:<pad> 1:root->idx 2:idx->root

        for i in range(len(e['aspect_sentiment'])):
            aspect = e['aspect_sentiment'][i][0].lower()
            # ???????????????tokenization
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
    ??????token???postag???deprel??????????????????????????????
    '''
    pkls_path = os.path.join(args.output_dir, 'pkls')
    if not os.path.exists(pkls_path):
        os.makedirs(pkls_path)

    # ??????/?????? word vocab ??? glove embeddings.
    # Elmo and bert ????????????vocab???embedding??????????????????
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


def preprocess(args):
    train, test = load_dataset(args)
    train_reshaped = get_reshaped_data(train, args)
    test_reshaped = get_reshaped_data(test, args)

    logger.info('****** After reshaping ******')
    logger.info('Train set size: %s', len(train_reshaped))
    logger.info('Test set size: %s,', len(test_reshaped))

    # ????????????(part of speech, dep_tag) ?????????????????? pickles.
    word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab = get_vocabs(
        train_reshaped + test_reshaped, args)

    logger.info('****** After getting vocab ******')

    if args.embedding_type == 'glove':
        embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32))
        args.glove_embedding = embedding

    # ?????? Dataset
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

    # ????????????(part of speech, dep_tag) ?????????????????? pickles.
    word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab = get_vocabs(
        train_reshaped + test_reshaped, args)

    logger.info('****** After getting vocab ******')

    # ?????? Dataset
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