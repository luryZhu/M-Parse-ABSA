import argparse
import logging
from preprocess import preprocess
from model import Aspect_Bert_GAT
from trainer import get_collate_fn
import os
import torch
import random
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--cuda_id', type=str, default='0',
                        help='Choose which GPUs to run')
    parser.add_argument('--seed', type=int, default=2019,
                        help='random seed for initialization')

    parser.add_argument('--dataset_name', type=str, default='rest',
                        choices=['rest', 'laptop', 'twitter'],
                        help='Choose absa dataset.')
    parser.add_argument('--parser_name', type=str, default='CoreNLP',
                        choices=['CoreNLP', 'Biaffine', 'Stanza'],
                        help='Choose dependency parser.')
    parser.add_argument('--dataset_dir', type=str, default='data/depparsed',
                        choices=['rest', 'laptop', 'twitter'],
                        help='Directory to load depparsed data.')
    parser.add_argument('--output_dir', type=str, default='../output',
                        help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')
    parser.add_argument('--bert_model_dir', type=str, default='../bert',
                        help='Path to pre-trained Bert model.')

    parser.add_argument('--multi_hop', type=bool, default=True,
                        help='open multi_hop')
    parser.add_argument('--add_non_connect', type=bool, default=True,
                        help='open add_non_connect')
    parser.add_argument('--max_hop', type=int, default=4,
                        help='set max_hop')

    # model
    parser.add_argument('--pure_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')
    parser.add_argument('--gat_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')
    parser.add_argument('--highway', action='store_true',
                        help='Use highway embed.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers of bilstm or highway or elmo.')
    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of heads for gat.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate for embedding.')

    # GAT
    parser.add_argument('--gat', action='store_true',
                        help='GAT')
    parser.add_argument('--gat_our', action='store_true',
                        help='GAT_our')
    parser.add_argument('--gat_attention_type', type=str, choices=['linear', 'dotprod', 'gcn'], default='dotprod',
                        help='The attention used for gat')

    parser.add_argument('--embedding_type', type=str, default='bert', choices=['glove', 'bert'])
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Dimension of glove embeddings')
    parser.add_argument('--dep_relation_embed_dim', type=int, default=300,
                        help='Dimension for dependency relation embeddings.')

    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--final_hidden_size', type=int, default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--num_mlps', type=int, default=2,
                        help='Number of mlps in the last of model.')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes of ABSA.')

    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")

    return parser.parse_args()


def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    args = parse_args()
    args.device = 'cpu'
    args.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
    train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab = preprocess(args)

    # 获取 模型
    model = Aspect_Bert_GAT(args, dep_tag_vocab['len'], pos_tag_vocab['len'])  # R-GAT + Bert

    m_state_dict = torch.load(os.path.join(args.output_dir, 'md.pt'))
    best_model = model.load_state_dict(m_state_dict)

    collate_fn=get_collate_fn(args)
    eval_dataset=test_dataset[2]
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    predict = best_model(test_dataset.data[0])
    print(predict)







if __name__ == '__main__':
    main()