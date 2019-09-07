"""
data_json has
0. refs:       [{ref_id, ann_id, box, image_id, split, category_id, sent_ids, att_wds}]
1. images:     [{image_id, ref_ids, file_name, width, height, h5_id}]
2. anns:       [{ann_id, category_id, image_id, box, h5_id}]
3. sentences:  [{sent_id, tokens, h5_id}]
4. word_to_ix: {word: ix}
5. att_to_ix : {att_wd: ix}
6. att_to_cnt: {att_wd: cnt}
7. label_length: L

Note, box in [xywh] format
label_h5 has
/labels is (M, max_length) uint32 array of encoded labels, zeros padded
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import io
import json
import _init_paths
from loaders.loader import Loader
import os
import argparse
import os.path as osp
import numpy as np


import numpy as np
import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class ComSim(nn.Module):
    def __init__(self, embedding_mat):
        super(ComSim, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_mat))
        self.cossim = nn.CosineSimilarity()

    def cal_sim(self, ann_cats, sen_cats, none_idx):
        ann_cat_embs = self.get_cat_emb(ann_cats)
        sen_cat_embs = self.get_cat_emb(sen_cats)
        cos_sims = []
        sen_wd_emb = []
        for sen_cat_emb in sen_cat_embs:
            sen_cos_sim = []
            sen_wd_emb.append(torch.stack(sen_cat_emb).sum(0).squeeze().numpy().tolist())
            for ann_cat_emb in ann_cat_embs:
                sen_cos_sim += [self.com_sim(sen_cat_emb, ann_cat_emb, none_idx)]
            cos_sims += [sen_cos_sim]
        return cos_sims, sen_wd_emb

    def com_sim(self, sen_cat_emb, ann_cat_emb, none_idx):
        max_sim = 0
        none_emb = self.embedding(torch.LongTensor(none_idx))
        for sen_wd  in sen_cat_emb:
            for ann_wd  in ann_cat_emb:
                if torch.equal(sen_wd, none_emb):
                    sim = torch.zeros(1)
                else:
                    sim = self.cossim(sen_wd, ann_wd)
                if sim >= max_sim or sim < 0:
                    max_sim = sim
        return max_sim.numpy().tolist()

    def get_cat_emb(self, cats):
        cat_embs = []
        for cat in cats:
            cat_emb = []
            #  注意wds是否都是list
            for wds in cat:
                wd_embedding = self.embedding(torch.LongTensor([wds]))
                cat_emb += [wd_embedding]
            cat_embs += [cat_emb]
        return cat_embs

# load vocabulary file
def load_vocab_dict_from_file(dict_file):
    if (sys.version_info > (3, 0)):
        with open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    else:
        with io.open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    vocab_dict = {words[n]: n for n in range(len(words))}
    return vocab_dict

# get word location in vocabulary
def words2vocab_indices(words, vocab_dict, UNK_IDENTIFIER):
    if isinstance(words, str):
        vocab_indices = [vocab_dict[words] if words in vocab_dict else vocab_dict[UNK_IDENTIFIER]]
    else:
        vocab_indices = [(vocab_dict[w] if w in vocab_dict else vocab_dict[UNK_IDENTIFIER])
                         for w in words]
    return vocab_indices

# get ann category
def get_ann_category(image_id, Images, vocab_file, Anns, ix_to_cat, UNK_IDENTIFIER):
    image = Images[image_id]
    ann_ids = image['ann_ids']
    vocab_dict = load_vocab_dict_from_file(vocab_file)
    ann_cats = []

    for ann_id in ann_ids:
        ann = Anns[ann_id]
        cat_id = ann['category_id']
        cat = ix_to_cat[cat_id]
        cat = cat.split(' ')
        cat = words2vocab_indices(cat, vocab_dict, UNK_IDENTIFIER)
        ann_cats += [cat]
    return ann_cats

# get category for each sentence
def get_sent_category(image_id, Images, vocab_file, Refs, sub_obj_wds, UNK_IDENTIFIER):
    image = Images[image_id]
    ref_ids = image['ref_ids']
    vocab_dict = load_vocab_dict_from_file(vocab_file)
    sub_wds = []
    obj_wds = []
    for ref_id in ref_ids:
        ref = Refs[ref_id]
        sent_ids = ref['sent_ids']
        for sent_id in sent_ids:
            att_wds = sub_obj_wds[str(sent_id)]
            sub_wd = att_wds['r1']
            sub_wd = words2vocab_indices(sub_wd, vocab_dict, UNK_IDENTIFIER)
            obj_wd = att_wds['r6']
            obj_wd = words2vocab_indices(obj_wd, vocab_dict, UNK_IDENTIFIER)
            sub_wds += [sub_wd]
            obj_wds += [obj_wd]
    none_idx = words2vocab_indices('none', vocab_dict, UNK_IDENTIFIER)


    sent_att = {}
    sent_att['sub_wds'] = sub_wds
    sent_att['obj_wds'] = obj_wds
    return sent_att, none_idx

def main(params):
    # dataset_splitBy
    data_root, dataset, splitBy = params['data_root'], params['dataset'], params['splitBy']
    dataset_splitBy = dataset + '_' + splitBy
    # max_length
    if params['max_length'] == None:
        if params['dataset'] in ['refcoco', 'refclef', 'refcoco+']:
            params['max_length'] = 10
            params['topK'] = 50
        elif params['dataset'] in ['refcocog']:
            params['max_length'] = 20
            params['topK'] = 50
        else:
            raise NotImplementedError

    # mkdir and write json file
    if not osp.isdir(osp.join('cache/similarity', dataset + '_' + splitBy)):
        os.makedirs(osp.join('cache/similarity', dataset + '_' + splitBy))

    # load data
    data_json = osp.join('cache/prepro', dataset_splitBy, 'data.json')
    sub_obj_wds = osp.join('cache/sub_obj_wds', dataset_splitBy, 'sub_obj.json')

    vocab_file = 'cache/word_embedding/vocabulary_72700.txt'
    UNK_IDENTIFIER = '<unk>'  # <unk> is the word used to identify unknown words
    num_vocab = 72704
    embed_dim = 300
    embedmat_path = 'cache/word_embedding/embed_matrix.npy'
    embedding_mat = np.load(embedmat_path)

    # load the json file which contains info about the dataset
    print('Loader loading subject and object words:', sub_obj_wds)
    sub_obj_wds_info = json.load(open(sub_obj_wds))
    sub_obj_wds = sub_obj_wds_info['sub_obj_wds']
    print('Loader loading data.json: ', data_json)
    info = json.load(open(data_json))
    cat_to_ix = info['cat_to_ix']
    ix_to_cat = {ix: cat for cat, ix in cat_to_ix.items()}
    print('object cateogry size is ', len(ix_to_cat))
    images = info['images']
    anns = info['anns']
    refs = info['refs']
    sentences = info['sentences']
    print('we have %s images.' % len(images))
    print('we have %s anns.' % len(anns))
    print('we have %s refs.' % len(refs))
    print('we have %s sentences.' % len(sentences))

    # construct mapping
    Refs = {ref['ref_id']: ref for ref in refs}
    Images = {image['image_id']: image for image in images}
    Anns = {ann['ann_id']: ann for ann in anns}
    sim = {}
    for image in images:
        image_id = image['image_id']

        # get ann category
        ann_cats = get_ann_category(image_id, Images, vocab_file, Anns, ix_to_cat, UNK_IDENTIFIER)

        # get subject, object, location words for each sentence
        sent_att, none_idx = get_sent_category(image_id, Images, vocab_file, Refs, sub_obj_wds, UNK_IDENTIFIER)

        # compute similarity
        com_sim = ComSim(embedding_mat)

        sub_sim, sub_emb = com_sim.cal_sim(ann_cats, sent_att['sub_wds'], none_idx)  # (sent_num, ann_num)
        obj_sim, obj_emb = com_sim.cal_sim(ann_cats, sent_att['obj_wds'], none_idx)  # (sent_num, ann_num)

        sim[image_id] = {'sub_sim': sub_sim, 'obj_sim': obj_sim, 'sub_emb': sub_emb, 'obj_emb': obj_emb}
        print('image %s is written! %d/%d ' %(image_id, len(sim), len(images)))

    json.dump({'sim': sim},
              open(osp.join('cache/similarity', dataset + '_' + splitBy, params['output_json']), 'w'))
    print('similarity written!')






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_json', default='similarity.json', help='output json file')
    parser.add_argument('--data_root', default='data', type=str,
                        help='data folder containing images and four datasets.')
    parser.add_argument('--dataset', default='refcoco', type=str, help='refcoco/refcoco+/refcocog')
    parser.add_argument('--splitBy', default='unc', type=str, help='unc/google')
    parser.add_argument('--max_length', type=int, help='max length of a caption')  # refcoco 10, refclef 10, refcocog 20
    parser.add_argument('--images_root', default='', help='root location in which images are stored')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--topK', default=50, type=int, help='top K attribute words')

    # argparse
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))

    # call main
    main(params)