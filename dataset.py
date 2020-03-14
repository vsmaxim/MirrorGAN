from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from config import cfg
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import json
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
import pickle


def get_imgs(img_path, imsize, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []

    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Resize(imsize[i])(img)
            else:
                re_img = img
            re_img = normalize(re_img)
            ##Move channel last
            ##convert to numpy
            re_img = torch.Tensor.numpy(re_img).transpose(1, 2, 0)
            ret.append(re_img)

    return ret


class TextDataset(data.Dataset):
    def __init__(self,
                 data_dir,
                 split='train',
                 base_size=64,
                 transform=None,
                 target_transform=None,
                 fraction=1):
        self.transform = transform

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []

        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir

        self.filenames_df, \
        self.annotations_df, \
        self.instances_df = self.load_filenames_and_annotations(fraction)
        
        self.train_df, self.test_df = self.load_test_train_splitted_data()

        split_dir = os.path.join(data_dir, split)

        self.captions, self.ixtoword, \
        self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.number_example = len(self.filenames)

    def load_filenames_and_annotations(self, fraction):
        filepath = os.path.join(self.data_dir, 'annotations/captions_train2014.json')

        with open(filepath) as f:
          json_dataset = json.load(f)
        
        df_filenames = pd.DataFrame(json_dataset['images'])\
          .sample(frac=fraction)\
          .reset_index(drop=True)

        sampled_ids = df_filenames['id']
        
        df_annotations = pd.DataFrame(json_dataset['annotations'])
        df_annotations = df_annotations[df_annotations['image_id'].isin(sampled_ids)]
        df_annotations.set_index('image_id')

        instances_filepath = os.path.join(self.data_dir, 'annotations/instances_train2014.json')

        with open(filepath) as f:
          instances_json = json.load(f)

        df_instances = pd.DataFrame(json_dataset['instances'])
        df_instances = df_instances[df_instances['image_id'].isin(sampled_ids)]
        df_instances.set_index('image_id')

        return df_filenames, df_annotations, df_instances

    def load_captions(self, filenames_df):
        all_captions = []

        for file_id in filenames_df:
            rows = self.annotation_df[self.annotation_df['id'] == file_id]

            for cnt, caption in enumerate(rows['caption']):
                # Replace all unnecessary 
                caption = caption.replace('\ufffd\ufffd', ' ')
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(caption.lower())

                all_captions.append([
                  token.encode('ascii', 'ignore').decode('ascii')
                  for token in tokens
                ])

                if cnt == self.embeddings_num:
                    break

        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions

        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = word_count.keys()

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0

        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [
            train_captions_new, test_captions_new, ixtoword, wordtoix,
            len(ixtoword)
        ]
    
    def load_test_train_splitted_data(self):
        train = self.filenames_df.sample(frac=0.8, random_state=40)
        test = self.filenames_df.drop(train)
        return train, test

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')

        if not os.path.isfile(filepath):
            train_captions = self.load_captions(self.train_df)
            test_captions = self.load_captions(self.test_df)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump(
                    [train_captions, test_captions, ixtoword, wordtoix],
                    f,
                    protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
        else:  # split=='test'
            captions = test_captions
        return captions, ixtoword, wordtoix, n_words

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        #
        file_row = self.filenames_df.iloc[index]
        
        key = file_row['id']
        filename = file_row['file_name']
        bbox = self.instances_df[key]['bbox']
        img_name = os.path.join(self.data_dir, 'images/', filename)
        
        imgs = get_imgs(
            img_name,
            self.imsize,
            bbox,
            self.transform,
            normalize=self.norm,
        )

        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)
