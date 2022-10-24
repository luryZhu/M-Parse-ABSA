import torch
from torch.utils.data import Dataset


class ASBA_Depparsed_Dataset(Dataset):
    '''
    Convert examples to features, numericalize text to ids.
    data:
        -list of dict:
            keys: sentence, tags, pos_class, aspect, sentiment,
                predicted_dependencies, predicted_heads,
                from, to, dep_tag, dep_idx, dependencies, dep_dir

    After processing,
    data:
        sentence
        tags
        pos_class
        aspect
        sentiment
        from
        to
        dep_tag
        dep_idx
        dep_dir
        predicted_dependencies_ids
        predicted_heads
        dependencies
        sentence_ids
        aspect_ids
        tag_ids
        dep_tag_ids
        text_len
        aspect_len
        if bert:
            input_ids
            word_indexer

    Return from getitem:
        sentence_ids
        aspect_ids
        dep_tag_ids
        dep_dir_ids
        pos_class
        text_len
        aspect_len
        sentiment
        deprel
        dephead
        aspect_position
        if bert:
            input_ids
            word_indexer
            input_aspect_ids
            aspect_indexer
        or:
            input_cat_ids
            segment_ids
    '''

    def __init__(self, data, args, word_vocab, dep_tag_vocab, pos_tag_vocab):
        self.data = data
        self.args = args
        self.word_vocab = word_vocab
        self.dep_tag_vocab = dep_tag_vocab
        self.pos_tag_vocab = pos_tag_vocab

        self.convert_features()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        items = e['dep_tag_ids'], \
            e['pos_class'], e['text_len'], e['aspect_len'], e['sentiment'],\
            e['dep_rel_ids'], e['predicted_heads'], e['aspect_position'], e['dep_dir_ids']
        if self.args.embedding_type == 'glove':
            non_bert_items = e['sentence_ids'], e['aspect_ids']
            items_tensor = non_bert_items + items
            items_tensor = tuple(torch.tensor(t) for t in items_tensor)
        elif self.args.embedding_type == 'elmo':
            items_tensor = e['sentence_ids'], e['aspect_ids']
            items_tensor += tuple(torch.tensor(t) for t in items)
        else:  # bert
            if self.args.pure_bert:
                bert_items = e['input_cat_ids'], e['segment_ids']
                items_tensor = tuple(torch.tensor(t) for t in bert_items)
                items_tensor += tuple(torch.tensor(t) for t in items)
            else:
                bert_items = e['input_ids'], e['word_indexer'], e['input_aspect_ids'], e['aspect_indexer'], e['input_cat_ids'], e['segment_ids']
                # segment_id
                items_tensor = tuple(torch.tensor(t) for t in bert_items)
                items_tensor += tuple(torch.tensor(t) for t in items)
        return items_tensor

    def convert_features_bert(self, i):
        """
        BERT features.
        convert sentence to feature.
        """
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = 0
        # tokenizer = self.args.tokenizer

        tokens = []
        word_indexer = []
        aspect_tokens = []
        aspect_indexer = []

        for word in self.data[i]['sentence']:
            word_tokens = self.args.tokenizer.tokenize(word)
            token_idx = len(tokens)
            tokens.extend(word_tokens)
            # word_indexer is for indexing after bert, feature back to the length of original length.
            word_indexer.append(token_idx)

        # aspect
        for word in self.data[i]['aspect']:
            word_aspect_tokens = self.args.tokenizer.tokenize(word)
            token_idx = len(aspect_tokens)
            aspect_tokens.extend(word_aspect_tokens)
            aspect_indexer.append(token_idx)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0

        tokens = [cls_token] + tokens + [sep_token]
        aspect_tokens = [cls_token] + aspect_tokens + [sep_token]
        word_indexer = [i+1 for i in word_indexer]
        aspect_indexer = [i+1 for i in aspect_indexer]

        input_ids = self.args.tokenizer.convert_tokens_to_ids(tokens)
        input_aspect_ids = self.args.tokenizer.convert_tokens_to_ids(
            aspect_tokens)

        # check len of word_indexer equals to len of sentence.
        assert len(word_indexer) == len(self.data[i]['sentence'])
        assert len(aspect_indexer) == len(self.data[i]['aspect'])

        # THE STEP:Zero-pad up to the sequence length, save to collate_fn.

        if self.args.pure_bert:
            input_cat_ids = input_ids + input_aspect_ids[1:]
            segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])

            self.data[i]['input_cat_ids'] = input_cat_ids
            self.data[i]['segment_ids'] = segment_ids
        else:
            input_cat_ids = input_ids + input_aspect_ids[1:]
            segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])

            self.data[i]['input_cat_ids'] = input_cat_ids
            self.data[i]['segment_ids'] = segment_ids
            self.data[i]['input_ids'] = input_ids
            self.data[i]['word_indexer'] = word_indexer
            self.data[i]['input_aspect_ids'] = input_aspect_ids
            self.data[i]['aspect_indexer'] = aspect_indexer

    def convert_features(self):
        '''
        Convert sentence, aspects, pos_tags, dependency_tags to ids.
        '''
        for i in range(len(self.data)):
            if self.args.embedding_type == 'glove':
                self.data[i]['sentence_ids'] = [self.word_vocab['stoi'][w]
                                                for w in self.data[i]['sentence']]
                self.data[i]['aspect_ids'] = [self.word_vocab['stoi'][w]
                                              for w in self.data[i]['aspect']]
            elif self.args.embedding_type == 'elmo':
                self.data[i]['sentence_ids'] = self.data[i]['sentence']
                self.data[i]['aspect_ids'] = self.data[i]['aspect']
            else:  # self.args.embedding_type == 'bert'
                self.convert_features_bert(i)

            self.data[i]['text_len'] = len(self.data[i]['sentence'])
            self.data[i]['aspect_position'] = [0] * self.data[i]['text_len']
            try:  # find the index of aspect in sentence
                for j in range(self.data[i]['from'], self.data[i]['to']):
                    self.data[i]['aspect_position'][j] = 1
            except:
                for term in self.data[i]['aspect']:
                    self.data[i]['aspect_position'][self.data[i]
                                                    ['sentence'].index(term)] = 1

            self.data[i]['dep_tag_ids'] = [self.dep_tag_vocab['stoi'][w]
                                           for w in self.data[i]['dep_tag']]
            self.data[i]['dep_dir_ids'] = [idx
                                           for idx in self.data[i]['dep_dir']]
            self.data[i]['pos_class'] = [self.pos_tag_vocab['stoi'][w]
                                             for w in self.data[i]['tags']]
            self.data[i]['aspect_len'] = len(self.data[i]['aspect'])

            self.data[i]['dep_rel_ids'] = [self.dep_tag_vocab['stoi'][r]
                                           for r in self.data[i]['predicted_dependencies']]