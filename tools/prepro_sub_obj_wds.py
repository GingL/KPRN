"""
Preprocess a raw json dataset into hdf5 and json files for use in lib/loaders

Input: refer loader
Output: json file has
- refs:       [{ref_id, ann_id, box, image_id, split, category_id, sent_ids, att_wds}]
- images:     [{image_id, ref_ids, file_name, width, height, h5_id}]
- anns:       [{ann_id, category_id, image_id, box, h5_id}]
- sentences:  [{sent_id, tokens, h5_id}]
- word_to_ix: {word: ix}
- att_to_ix : {att_wd: ix}
- att_to_cnt: {att_wd: cnt}
- label_length: L

Output: hdf5 file has
/labels is (M, seq_length) int32 array of encoded labels, zeros padded in the end
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import argparse
import os.path as osp


forbidden_att = ['none', 'other', 'sorry', 'pic', 'extreme', 'rightest', 'tie', 'leftest', 'hard', 'only',
                 'darkest', 'foremost', 'topmost', 'leftish', 'utmost', 'lemon', 'good', 'hot', 'more', 'least', 'less',
                 'cant', 'only', 'opposite', 'upright', 'lightest', 'single', 'touching', 'bad', 'main', 'remote',
                 '3pm',
                 'same', 'bottom', 'middle']
forbidden_verb = ['none', 'look', 'be', 'see', 'have', 'head', 'show', 'strip', 'get', 'turn', 'wear',
                  'reach', 'get', 'cross', 'turn', 'point', 'take', 'color', 'handle', 'cover', 'blur', 'close', 'say',
                  'go',
                  'dude', 'do', 'let', 'think', 'top', 'head', 'take', 'that', 'say', 'carry', 'man', 'come', 'check',
                  'stuff',
                  'pattern', 'use', 'light', 'follow', 'rest', 'watch', 'make', 'stop', 'arm', 'try', 'want', 'count',
                  'lead',
                  'know', 'mean', 'lap', 'moniter', 'dot', 'set', 'cant', 'serve', 'surround', 'isnt', 'give', 'click']
forbidden_noun = ['none', 'picture', 'pic', 'screen', 'background', 'camera', 'edge', 'standing', 'thing',
                  'holding', 'end', 'view', 'bottom', 'center', 'row', 'piece']


def get_sub_obj(refer, params, att_types=['r1', 'r4', 'r5','r6']):
    """
  Load sents = [{tokens, atts, sent_id, parse, raw, sent left}]
  from pyutils/refer-parser2/cache/parsed_atts/dataset_splitBy/sents.json
  """
    sents = json.load(open(osp.join('pyutils/refer-parser2/cache/parsed_atts',
                                    params['dataset'] + '_' + params['splitBy'], 'sents.json')))
    sentToRef = refer.sentToRef

    forbidden = forbidden_noun + forbidden_att + forbidden_verb

    sub_obj_wds = {}
    for sent in sents:
        sent_id = sent['sent_id']
        atts = sent['atts']
        ref_id = sentToRef[sent_id]['ref_id']
        sent_sub_obj_wds = {}
        for att_type in att_types:
            # att_wds = [wd for wd in atts[att_type] if wd not in forbidden]
            att_wds = [wd for wd in atts[att_type] if wd not in forbidden]
            if len(att_wds) > 0:
                sent_sub_obj_wds[att_type] = att_wds
            else:
                sent_sub_obj_wds[att_type] = 'none'
        sub_obj_wds[sent_id] = sent_sub_obj_wds
    return sub_obj_wds


def main(params):
    # dataset_splitBy
    data_root, dataset, splitBy = params['data_root'], params['dataset'], params['splitBy']

    # mkdir and write json file
    if not osp.isdir(osp.join('cache/sub_obj_wds', dataset + '_' + splitBy)):
        os.makedirs(osp.join('cache/sub_obj_wds', dataset + '_' + splitBy))

    # load refer
    sys.path.insert(0, 'pyutils/refer')
    from refer import REFER
    refer = REFER(data_root, dataset, splitBy)

    # create subject and object
    sub_obj_wds = get_sub_obj(refer, params, ['r1', 'r4', 'r5', 'r6'])
    json.dump({'sub_obj_wds': sub_obj_wds},
              open(osp.join('cache/sub_obj_wds', dataset + '_' + splitBy, params['output_json']), 'w'))
    print('sub_obj_wds written!')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_json', default='sub_obj_wds.json', help='output json file')
    parser.add_argument('--data_root', default='data', type=str,
                        help='data folder containing images and four datasets.')
    parser.add_argument('--dataset', default='refcoco', type=str, help='refcoco/refcoco+/refcocog')
    parser.add_argument('--splitBy', default='unc', type=str, help='unc/google')
    parser.add_argument('--images_root', default='', help='root location in which images are stored')


    # argparse
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))

    # call main
    main(params)
