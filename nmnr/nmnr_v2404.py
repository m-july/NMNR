#!/usr/bin/env python
# coding: utf-8

# # [A] 参数和常量

# In[1]:


import sys
import random
import datetime
import numbers

import copy
#import regex as re

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pickle
import json
import zhconv

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns

import pandas as pd
pd.set_option('display.max_rows', 40)
pd.set_option('display.min_rows', 40)
pd.set_option('display.max_columns', None)
import openpyxl

import sklearn
import scipy

import skimage
import PIL
import wand
from wand.drawing import Drawing
from wand.image import Image

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# ## [A1] 随机参数工具

# In[2]:


class My_Parameters:
    """初始化随机参数工具"""
    
    def __init__(self, params_path, font_info_path):
        """初始化"""
        
        # read param
        excel_sheets = pd.read_excel(params_path, sheet_name=None)
        self.params = excel_sheets['Main']
        self.params['volume_noise'] = 0
        self.params['song_noise'] = 0
        self.params['item_noise'] = 0
        self.params.set_index('param_name', inplace=True)
        self.item_update_dict = {}
        for index in self.params.index:
            if self.params.loc[index, 'make_noise']:
                item_update_type = self.params.loc[index, 'item_update_type']
                if item_update_type in self.item_update_dict.keys():
                    self.item_update_dict[item_update_type].append(index)
                else:
                    self.item_update_dict[item_update_type] = [index,]
                    
        # read font_info
        excel_sheets_font = pd.read_excel(font_info_path, sheet_name=None)
        self.font_info = excel_sheets_font['Main']
        self.font_categories = self.font_info['usage'].unique()
        self.font_indexer = {cate: [] for cate in self.font_categories}
        for index in self.font_info.index:
            self.font_indexer[self.font_info.at[index, 'usage']].append(index)

        # add path
        self.font_info['font_file_path'] = ''
        for index in self.font_info.index:
            self.font_info.loc[index, 'font_file_path'] = os.path.join('..\\wand_assets', self.font_info.loc[index, 'font_file'])
        
        self.active_font_mapping = {
            'number': 0,
            'hanzi': 1,
            'timesig_num': 2,
        }
        
    def update_random_font(self):
        self.active_font_mapping = {cate: random.choice(self.font_indexer[cate]) for cate in self.font_categories}
        
    def update_volume(self):
        """重新生成每卷层次的随机值"""
        for index in self.params.index:
            if self.params.loc[index, 'make_noise']:
                self.params['volume_noise'] = np.random.normal(scale=self.params['volume_sigma'])

    def update_song(self):
        """重新生成每曲层次的随机值"""
        for index in self.params.index:
            if self.params.loc[index, 'make_noise']:
                self.params['song_noise'] = np.random.normal(scale=self.params['song_sigma'])

    def update_item(self, item_update_type):
        """重新生成项目层次的随机值，需要指出更新类型"""
        for index in self.item_update_dict[item_update_type]:
            self.params['item_noise'] = np.random.normal(scale=self.params['item_sigma'])
    
    def get(self, index):
        """获取参数值"""
        if self.params.loc[index, 'make_noise']:
            v = self.params.loc[index, 'base_value'] + self.params.loc[index, 'volume_noise'] + self.params.loc[index, 'song_noise'] + self.params.loc[index, 'item_noise']
        else:
            v = self.params.loc[index, 'base_value']
        minval = self.params.loc[index, 'min_val']
        maxval = self.params.loc[index, 'max_val']
        if minval != np.nan and v < minval:
            v = minval
        if maxval != np.nan and v > maxval:
            v = maxval
        return int(round(v)) if self.params.loc[index, 'is_int'] else v

    def get_a(self, index, is_appog):
        """获取参数值（判断是否为倚音版）"""
        if not is_appog:
            return self.get(index)
        else:
            if self.params.loc[index, 'appog_resizable']:
                if self.params.loc[index, 'make_noise']:
                    v = self.params.loc[index, 'base_value'] + self.params.loc[index, 'volume_noise'] + self.params.loc[index, 'song_noise'] + self.params.loc[index, 'item_noise']
                else:
                    v = self.params.loc[index, 'base_value']
                v *= self.get('APPOG_SIZE_RATIO')

            elif self.params.loc[index, 'appog_rename']:
                v = self.get(index + '_APPOG')

            else:
                assert False

        minval = self.params.loc[index, 'min_val']
        maxval = self.params.loc[index, 'max_val']
        if minval != np.nan and v < minval:
            v = minval
        if maxval != np.nan and v > maxval:
            v = maxval
        return int(round(v)) if self.params.loc[index, 'is_int'] else v
                
    def get_dict(self, indices, is_appog=False):
        """获取一系列参数值，返回对应的词典"""
        result = {}
        for index in indices:
            if is_appog:
                result[index] = self.get_a(index, True)
            else:
                result[index] = self.get(index)
        return result
                
    def get_active_font(self, s):
        """获取活动字体信息，是一个参数表"""
        assert s in self.active_font_mapping.keys()
        return self.font_info.iloc[self.active_font_mapping[s]]
        
    def set_active_font(self, s, index):
        """设置活动字体，需要传入下标"""
        self.active_font_mapping[s] = index


# In[3]:


PARAMS_ = My_Parameters('..\\parameters.xlsx', '..\\wand_assets\\wand_font_info.xlsx')


# ## [A3] 谱例

# In[4]:


#V06.S0382

example_monophonic_seq = {
    'notes': [
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 1,
        },
        {
            'note_type': 'bar',
        },
        {
            'note_type': 'bar',
        },
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 1,
        },
        {
            'note_type': 'bar',
        },
        {
            'note_type': 'number',
            'number': 5,
            'dot_top': 1,
        },
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 1,
        },
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 1,
        },
        {
            'note_type': 'number',
            'number': 2,
            'dot_top': 1,
        },
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 1,
        },
        {
            'note_type': 'number',
            'number': 2,
            'dot_top': 1,
            'dot_right': 1,
        },
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 1,
        },
        {
            'note_type': 'number',
            'number': 1,
            'dot_top': 1,
        },
        {
            'note_type': 'number',
            'number': 1,
            'dot_top': 1,
        },
        {
            'note_type': 'number',
            'number': 6,
        },
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 1,
            'dot_right': 1,
        },
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 1,
        },
        {
            'note_type': 'number',
            'number': 2,
            'dot_top': 1,
        },
        {
            'note_type': 'number',
            'number': 1,
            'dot_top': 1,
        },
    ],
    'underlines': [
        {
            'begin_id': 8,
            'end_id': 9,
        },
        {
            'begin_id': 10,
            'end_id': 11,
        },
        {
            'begin_id': 11,
            'end_id': 11,
        },
        {
            'begin_id': 14,
            'end_id': 15,
        },
        {
            'begin_id': 14,
            'end_id': 14,
        },
        {
            'begin_id': 17,
            'end_id': 18,
        },
    ],
    'curves': [
        {
            'curve_type': 'simple',
            'begin_id': 0,
            'end_id': 3,
        },
        {
            'curve_type': 'simple',
            'begin_id': 3,
            'end_id': 5,
        },
        {
            'curve_type': 'simple',
            'begin_id': 10,
            'end_id': 11,
        },
        {
            'curve_type': 'arrowed',
            'begin_id': 14,
            'end_id': 15,
        },
    ],
    'time_signatures': [
        {
            'ts_type': "fraction",
            'pos_id': 0,
            'ts_top': '3',
            'ts_bottom': '4',
        },
    ],
    'appogs': [
        {
            'align_id': 0,
            'orientation': 'before',
            'notes': [
                {
                    'note_type': 'number',
                    'number': 5,
                },
            ],
        },
        {
            'align_id': 16,
            'orientation': 'before',
            'notes': [
                {
                    'note_type': 'number',
                    'number': 2,
                },
            ],
        },
    ],
    'barlines': [
        {
            'barline_type': 'Simple',
            'pos_id': 3,
        },
        {
            'barline_type': 'Simple',
            'pos_id': 6,
        },
        {
            'barline_type': 'Simple',
            'pos_id': 10,
        },
        {
            'barline_type': 'Simple',
            'pos_id': 14,
        },
        {
            'barline_type': 'Simple',
            'pos_id': 19,
        },
    ],
    'hint': [
    ],
    'annos': [
        {
            'align_id': 1,
            'position': 'top',
            'content': '♩=85',
        },
    ],
    'lyrics': [
        [
            {
                'lyric_type': 'Bracket',
                'orientation': 'forward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '哎',
                'align_at': 1,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'backward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '党',
                'align_at': 9,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '的',
                'align_at': 10,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'forward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '那',
                'align_at': 11,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '个',
                'align_at': 12,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'backward',
            },
        ],
    ],
}


# In[5]:


#V06.S0382

example_monophonic_seq_0 = {
    'notes': [
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 2,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 3,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'bar',
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 2,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 3,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 2,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'number',
            'number': 5,
            'dot_bottom': 1,
        },
        {
            'note_type': 'bar',
        },
    ],
    'underlines': [
        {
            'begin_id': 0,
            'end_id': 1,
        },
        {
            'begin_id': 3,
            'end_id': 5,
        },
        {
            'begin_id': 3,
            'end_id': 4,
        },
        {
            'begin_id': 6,
            'end_id': 7,
        },
        {
            'begin_id': 10,
            'end_id': 12,
        },
        {
            'begin_id': 11,
            'end_id': 12,
        },
        {
            'begin_id': 13,
            'end_id': 14,
        },
        {
            'begin_id': 19,
            'end_id': 22,
        },
        {
            'begin_id': 19,
            'end_id': 22,
        },
    ],
    'curves': [
        {
            'curve_type': 'simple',
            'begin_id': 3,
            'end_id': 4,
        },
        {
            'curve_type': 'simple',
            'begin_id': 7,
            'end_id': 8,
        },
        {
            'curve_type': 'simple',
            'begin_id': 11,
            'end_id': 12,
        },
        {
            'curve_type': 'simple',
            'begin_id': 13,
            'end_id': 14,
        },
        {
            'curve_type': 'simple',
            'begin_id': 19,
            'end_id': 22,
        },
    ],
    'time_signatures': [
        {
            'ts_type': "fraction",
            'pos_id': 0,
            'ts_top': '2',
            'ts_bottom': '4',
        },
    ],
    'appogs': [
        {
            'align_id': 0,
            'orientation': 'before',
            'notes': [
                {
                    'note_type': 'number',
                    'number': 1,
                },
                {
                    'note_type': 'number',
                    'number': 2,
                },
            ],
        },
        {
            'align_id': 8,
            'orientation': 'before',
            'notes': [
                {
                    'note_type': 'number',
                    'number': 1,
                },
                {
                    'note_type': 'number',
                    'number': 2,
                },
                {
                    'note_type': 'number',
                    'number': 1,
                },
            ],
        },
        {
            'align_id': 23,
            'orientation': 'before',
            'notes': [
                {
                    'note_type': 'number',
                    'number': 1,
                },
            ],
        },
        {
            'align_id': 24,
            'orientation': 'before',
            'notes': [
                {
                    'note_type': 'number',
                    'number': 6,
                    'dot_bottom': 1,
                },
                {
                    'note_type': 'number',
                    'number': 7,
                    'dot_bottom': 1,
                },
                {
                    'note_type': 'number',
                    'number': 6,
                    'dot_bottom': 1,
                },
            ],
        },
    ],
    'barlines': [
        {
            'barline_type': 'Simple',
            'pos_id': 3,
        },
        {
            'barline_type': 'Simple',
            'pos_id': 8,
        },
        {
            'barline_type': 'Simple',
            'pos_id': 10,
        },
        {
            'barline_type': 'Simple',
            'pos_id': 15,
        },
        {
            'barline_type': 'Simple',
            'pos_id': 17,
        },
        {
            'barline_type': 'Simple',
            'pos_id': 19,
        },
        {
            'barline_type': 'Simple',
            'pos_id': 24,
        },
        {
            'barline_type': 'Fin',
            'pos_id': 26,
        },
    ],
    'hint': [
        {
            'hint_type': 'Fermata',
            'align_id': 7,
        },
        {
            'hint_type': 'Fermata',
            'align_id': 18,
        },
    ],
    'annos': [
        {
            'align_id': 0,
            'position': 'top',
            'content': '稍慢',
        },
    ],
    'lyrics': [
        [
            {
                'lyric_type': 'Lyric',
                'lyric': '太',
                'align_at': 0,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '阳',
                'align_at': 1,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'forward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '呃',
                'align_at': 2,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'backward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '一',
                'align_at': 3,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '出',
                'align_at': 5,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'forward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '呃',
                'align_at': 6,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '呃',
                'align_at': 7,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'backward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '往',
                'align_at': 10,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '上',
                'align_at': 11,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '升，',
                'align_at': 13,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '听',
                'align_at': 15,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '我',
                'align_at': 16,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '唱',
                'align_at': 17,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '个',
                'align_at': 18,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '车',
                'align_at': 19,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '水',
                'align_at': 23,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '歌。',
                'align_at': 24,
            },
        ],
        [
            {
                'lyric_type': 'Lyric',
                'lyric': '混',
                'align_at': 0,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '沌',
                'align_at': 1,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'forward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '呃',
                'align_at': 2,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'backward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '初',
                'align_at': 3,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '开',
                'align_at': 5,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'forward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '呃',
                'align_at': 6,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '呃',
                'align_at': 7,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'backward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '盘',
                'align_at': 10,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '古',
                'align_at': 11,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '皇，',
                'align_at': 13,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '三',
                'align_at': 15,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '皇',
                'align_at': 16,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '五',
                'align_at': 17,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '帝',
                'align_at': 18,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '治',
                'align_at': 19,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '纲',
                'align_at': 23,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '常。',
                'align_at': 24,
            },
        ],
    ],
}


# In[6]:


#V06.S0382

example_monophonic_seq_0_haha = {
    'notes': [
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 2,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 3,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'bar',
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 2,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 3,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 2,
        },
        {
            'note_type': 'number',
            'number': 1,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'number',
            'number': 6,
            'dot_bottom': 1,
        },
        {
            'note_type': 'number',
            'number': 5,
            'dot_bottom': 1,
        },
        {
            'note_type': 'bar',
        },
    ],
    'underlines': [
        {
            'begin_id': 0,
            'end_id': 1,
        },
        {
            'begin_id': 3,
            'end_id': 5,
        },
        {
            'begin_id': 3,
            'end_id': 4,
        },
        {
            'begin_id': 6,
            'end_id': 7,
        },
        {
            'begin_id': 10,
            'end_id': 12,
        },
        {
            'begin_id': 11,
            'end_id': 12,
        },
        {
            'begin_id': 13,
            'end_id': 14,
        },
        {
            'begin_id': 19,
            'end_id': 22,
        },
        {
            'begin_id': 19,
            'end_id': 22,
        },
    ],
    'curves': [
        {
            'curve_type': 'simple',
            'begin_id': 3,
            'end_id': 4,
        },
        {
            'curve_type': 'simple',
            'begin_id': 7,
            'end_id': 8,
        },
        {
            'curve_type': 'simple',
            'begin_id': 11,
            'end_id': 12,
        },
        {
            'curve_type': 'simple',
            'begin_id': 13,
            'end_id': 14,
        },
        {
            'curve_type': 'simple',
            'begin_id': 19,
            'end_id': 22,
        },
    ],
    'time_signatures': [
        {
            'ts_type': "fraction",
            'pos_id': 0,
            'ts_top': '2',
            'ts_bottom': '4',
        },
        {
            'ts_type': "rubato",
            'pos_id': 3,
        },
    ],
    'appogs': [
        {
            'align_id': 0,
            'orientation': 'before',
            'notes': [
                {
                    'note_type': 'number',
                    'number': 1,
                },
                {
                    'note_type': 'number',
                    'number': 2,
                },
            ],
        },
        {
            'align_id': 8,
            'orientation': 'before',
            'notes': [
                {
                    'note_type': 'number',
                    'number': 1,
                },
                {
                    'note_type': 'number',
                    'number': 2,
                },
                {
                    'note_type': 'number',
                    'number': 1,
                },
            ],
        },
        {
            'align_id': 23,
            'orientation': 'before',
            'notes': [
                {
                    'note_type': 'number',
                    'number': 1,
                },
            ],
        },
        {
            'align_id': 24,
            'orientation': 'before',
            'notes': [
                {
                    'note_type': 'number',
                    'number': 6,
                    'dot_bottom': 1,
                },
                {
                    'note_type': 'number',
                    'number': 7,
                    'dot_bottom': 1,
                },
                {
                    'note_type': 'number',
                    'number': 6,
                    'dot_bottom': 1,
                },
            ],
        },
    ],
    'barlines': [
        {
            'barline_type': 'Simple',
            'pos_id': 3,
        },
    ],
    'hints': [
        {
            'hint_type': 'Fermata',
            'align_id': 7,
        },
        {
            'hint_type': 'Fermata',
            'align_id': 18,
        },
        {
            'hint_type': 'Tremor',
            'align_id': 14,
        },
        {
            'hint_type': 'DownTriangle',
            'align_id': 2,
        },
        {
            'hint_type': 'Circle',
            'align_id': 3,
        },
        {
            'hint_type': 'Sharp',
            'align_id': 14,
        },
        {
            'hint_type': 'Flat',
            'align_id': 13,
        },
        {
            'hint_type': 'Natural',
            'align_id': 20,
        },
    ],
    'annos': [
        {
            'align_id': 0,
            'position': 'top',
            'content': '稍慢',
        },
    ],
    'lyrics': [
        [
            {
                'lyric_type': 'Lyric',
                'lyric': '太',
                'align_at': 0,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '阳',
                'align_at': 1,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'forward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '呃',
                'align_at': 2,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'backward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '一',
                'align_at': 3,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '出',
                'align_at': 5,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'forward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '呃',
                'align_at': 6,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '呃',
                'align_at': 7,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'backward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '往',
                'align_at': 10,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '上',
                'align_at': 11,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '升，',
                'align_at': 13,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '听',
                'align_at': 15,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '我',
                'align_at': 16,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '唱',
                'align_at': 17,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '个',
                'align_at': 18,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '车',
                'align_at': 19,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '水',
                'align_at': 23,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '歌。',
                'align_at': 24,
            },
        ],
        [
            {
                'lyric_type': 'Lyric',
                'lyric': '混',
                'align_at': 0,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '沌',
                'align_at': 1,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'forward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '呃',
                'align_at': 2,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'backward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '初',
                'align_at': 3,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '开',
                'align_at': 5,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'forward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '呃',
                'align_at': 6,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '呃',
                'align_at': 7,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'backward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '盘',
                'align_at': 10,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '古',
                'align_at': 11,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '皇，',
                'align_at': 13,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '三',
                'align_at': 15,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '皇',
                'align_at': 16,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '五',
                'align_at': 17,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '帝',
                'align_at': 18,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '治',
                'align_at': 19,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '纲',
                'align_at': 23,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '常。',
                'align_at': 24,
            },
        ],
    ],
}


# In[7]:


#V06.S0382

example_monophonic_seq_2 = {
    'notes': [
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 2,
        },
        {
            'note_type': 'bar',
        },
        {
            'note_type': 'bar',
        },
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 1,
        },
        {
            'note_type': 'bar',
        },
        {
            'note_type': 'number',
            'number': 5,
            'dot_top': 2,
        },
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 3,
        },
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 2,
        },
        {
            'note_type': 'number',
            'number': 2,
            'dot_top': 1,
        },
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 1,
        },
        {
            'note_type': 'number',
            'number': 2,
            'dot_top': 1,
            'dot_right': 1,
        },
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 1,
        },
        {
            'note_type': 'number',
            'number': 1,
            'dot_bottom': 1,
        },
        {
            'note_type': 'number',
            'number': 1,
            'dot_top': 1,
        },
        {
            'note_type': 'number',
            'number': 6,
        },
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 1,
            'dot_right': 1,
        },
        {
            'note_type': 'number',
            'number': 3,
            'dot_top': 1,
        },
        {
            'note_type': 'number',
            'number': 2,
            'dot_bottom': 2,
        },
        {
            'note_type': 'number',
            'number': 1,
            'dot_top': 1,
        },
    ],
    'underlines': [
        {
            'begin_id': 8,
            'end_id': 9,
        },
        {
            'begin_id': 10,
            'end_id': 11,
        },
        {
            'begin_id': 11,
            'end_id': 11,
        },
        {
            'begin_id': 14,
            'end_id': 15,
        },
        {
            'begin_id': 14,
            'end_id': 14,
        },
        {
            'begin_id': 17,
            'end_id': 18,
        },
    ],
    'curves': [
        {
            'curve_type': 'simple',
            'begin_id': 0,
            'end_id': 3,
        },
        {
            'curve_type': 'simple',
            'begin_id': 3,
            'end_id': 5,
        },
        {
            'curve_type': 'simple',
            'begin_id': 10,
            'end_id': 11,
        },
        {
            'curve_type': 'arrowed',
            'begin_id': 14,
            'end_id': 15,
        },
    ],
    'time_signatures': [
        {
            'ts_type': "fraction",
            'pos_id': 0,
            'ts_top': '3',
            'ts_bottom': '4',
        },
    ],
    'appogs': [
        {
            'align_id': 0,
            'orientation': 'before',
            'notes': [
                {
                    'note_type': 'number',
                    'number': 5,
                },
                {
                    'note_type': 'number',
                    'number': 1,
                    'dot_top': 1,
                },
                {
                    'note_type': 'number',
                    'number': 7,
                    'dot_bottom': 1,
                },
            ],
        },
        {
            'align_id': 16,
            'orientation': 'before',
            'notes': [
                {
                    'note_type': 'number',
                    'number': 2,
                },
            ],
        },
    ],
    'barlines': [
        {
            'barline_type': 'Simple',
            'pos_id': 3,
        },
        {
            'barline_type': 'Simple',
            'pos_id': 6,
        },
        {
            'barline_type': 'Simple',
            'pos_id': 10,
        },
        {
            'barline_type': 'Simple',
            'pos_id': 14,
        },
        {
            'barline_type': 'Simple',
            'pos_id': 19,
        },
    ],
    'hints': [
    ],
    'annos': [
        {
            'align_id': 1,
            'position': 'top',
            'content': '♩=85',
        },
    ],
    'lyrics': [
        [
            {
                'lyric_type': 'Bracket',
                'orientation': 'forward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '哎',
                'align_at': 0,
            },
            {
                'lyric_type': 'Anno',
                'anno': '①',
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'backward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '党',
                'align_at': 6,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '的',
                'align_at': 7,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'forward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '那',
                'align_at': 8,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '个',
                'align_at': 9,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'backward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '恩',
                'align_at': 10,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '情',
                'align_at': 12,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'forward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '咧',
                'align_at': 13,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'backward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '重',
                'align_at': 14,
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '如',
                'align_at': 16,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'forward',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '那',
                'align_at': 17,
            },
            {
                'lyric_type': 'Anno',
                'anno': '②③④',
            },
            {
                'lyric_type': 'Lyric',
                'lyric': '个',
                'align_at': 18,
            },
            {
                'lyric_type': 'Bracket',
                'orientation': 'backward',
            },
        ],
    ],
}


# # [B] 基础渲染器

# In[8]:


# # 'none' == "rgba(0, 0, 0, 0.0)"
# with Drawing() as draw:
#     with Image(width=200, height=200, background=wand.color.Color('none')) as img:  
#         draw.font = '..\wand_assets\impact.ttf'
#         draw.font_size = 20
#         draw.text(10, 80, 'Hello, Wand library!')
#         draw(img)
#         img.save(filename='test.png')


# ## [B1] 点类 `Position`

# In[9]:


class Position(object):
    """
    表示 2D 空间中的位置（或者向量）
    """

    def __init__(self, x, y):
        """初始化位置"""
        self.x = x
        self.y = y

    @staticmethod
    def zero():
        """零点"""
        return Position(0, 0)
        
    def __add__(self, other):
        """向量加法"""
        if isinstance(other, Position):
            return Position(self.x + other.x, self.y + other.y)
        else:
            raise TypeError("Unsupported operand type for +: 'Position' and " + type(other).__name__)
    def __sub__(self, other):
        """向量减法"""
        if isinstance(other, Position):
            return Position(self.x - other.x, self.y - other.y)
        else:
            raise TypeError("The sub operation is not defined for Position and " + type(other).__name__)

    def __mul__(self, other):
        """向量数乘"""
        return Position(self.x * other, self.y * other)
    def __rmul__(self, other):
        """向量数乘（数在右侧）"""
        return self.__mul__(other)
    def __neg__(self):
        """向量求负"""
        return Position(-self.x, -self.y)
    
    def totuple(self):
        """转成元组(x, y)"""
        return (self.x, self.y)
    def tolist(self):
        """转成列表[x, y]"""
        return [self.x, self.y]
    
    def round(self):
        """取整"""
        return Position(int(round(self.x)), int(round(self.y)))
    
    def __str__(self):
        """字符串化"""
        if isinstance(self.x, numbers.Integral) and isinstance(self.y, numbers.Integral):
            return "Position(%d, %d)" % (self.x, self.y)
        else:
            return "Position(%.4f, %.4f)" % (self.x, self.y)
    def __repr__(self):
        return self.__str__()
    
    def draw(self, draw, type='x', size=5.0, width=1.0, fill='none', color='r'):
        """绘制，type 为 'x', 'o', 'box' 之一"""
        draw.stroke_color = wand.color.Color(color)
        draw.stroke_width = width
        print("draw point at %s" % str(self))
        hs = size / 2
        if type == 'x':
            draw.line((self.x - hs, self.y - hs), (self.x + hs, self.y + hs))
            draw.line((self.x + hs, self.y - hs), (self.x - hs, self.y + hs))
        elif type == 'o':
            draw.fill_color = wand.color.Color(fill)
            draw.circle((self.x, self.y), # Center point
                        (self.x + hs, self.y)) # Perimeter point
        elif type == 'box':
            draw.fill_color = wand.color.Color(fill)
            draw.rectangle(left = self.x - hs,
                           top = self.y - hs,
                           right = self.x + hs,
                           bottom = self.y + hs)
        else:
            raise ValueError("Invalid param 'type' provided: {}".format(type))


# ## [B2] 盒子类 `Box`

# In[10]:


class Box(object):
    """
    表示以原点为中心的盒子。
    """
    
    # left = xmin, right = xmax, up = ymin, down = ymax
    def __init__(self, left, right, up, down):
        """初始化盒子"""
        self.left = left
        self.right = right
        self.up = up
        self.down = down

    @staticmethod
    def zero():
        """零点盒子"""
        return Box(0, 0, 0, 0)
    @staticmethod
    def pos_inf():
        """正无穷盒子"""
        inf = float('inf')
        return Box(-inf, inf, -inf, inf)
    @staticmethod
    def neg_inf():
        """负无穷盒子"""
        inf = float('inf')
        return Box(inf, -inf, inf, -inf)
        
    def expand_x(self, exp_x):
        """左右扩展"""
        return Box(self.left - exp_x, self.right + exp_x, self.up, self.down)
    def expand_y(self, exp_y):
        """上下扩展"""
        return Box(self.left, self.right, self.up - exp_y, self.down + exp_y)
    def expand(self, value):
        """四向扩展"""
        return Box(self.left - value, self.right + value, self.up - value, self.down + value)

    def offset_x(self, off_x):
        """水平位移"""
        return Box(self.left + off_x, self.right + off_x, self.up, self.down)
    def offset_y(self, off_y):
        """竖直位移"""
        return Box(self.left, self.right, self.up + off_y, self.down + off_y)
    def offset(self, pos):
        """向量位移"""
        if isinstance(pos, Position):
            return Box(self.left + pos.x, self.right + pos.x, self.up + pos.y, self.down + pos.y)
        else:
            raise TypeError("Expecting 'pos' an instance of 'Position', but " + type(pos).__name__ + " encountered.")
    def __add__(self, other):
        """向量位移"""
        if isinstance(other, Position):
            return self.offset(other)
        else:
            raise TypeError("Unsupported operand type for +: 'Position' and " + type(other).__name__)
    def __sub__(self, other):
        """向量反向位移"""
        if isinstance(other, Position):
            return self.offset(-other)
        else:
            raise TypeError("Unsupported operand type for -: 'Position' and " + type(other).__name__)
        
    def get_upleft(self):
        """左上角点"""
        return Position(self.left, self.up)
    def get_upright(self):
        """右上角点"""
        return Position(self.right, self.up)
    def get_downleft(self):
        """左下角点"""
        return Position(self.left, self.down)
    def get_downright(self):
        """右下角点"""
        return Position(self.right, self.down)
    
    def size(self):
        """获取大小"""
        return Position(self.right - self.left, self.down - self.up)
    
    def is_valid(self):
        """是否合法，即没有左右或上下反过来"""
        return self.right >= self.left and self.down >= self.up
    def __bool__(self):
        """是否合法，即没有左右或上下反过来"""
        result = self.is_valid()
        if isinstance(result, np.bool_):
            return bool(self.is_valid())
        else:
            return result
    
    def is_point(self):
        """是否缩为一点"""
        return self.left == self.right and self.up == self.down
    def is_zero_included(self):
        """原点是否在包围盒内部（含边界）"""
        return self.left <= 0 and self.up <= 0 and self.right >= 0 and self.down >= 0
    def is_zero_point(self):
        """是否缩为原点"""
        return self.is_point() and self.left == 0 and self.up == 0
    
    def contains(self, other):
        """是否包含（边界重叠也算包含）"""
        if isinstance(other, Box):
            return self.left <= other.left and self.up <= other.up and self.right >= other.right and self.down >= other.down
        else:
            raise TypeError("Expecting 'other' an instance of 'Box', but " + type(other).__name__ + " encountered.")
    def __ge__(self, other):
        """是否包含（边界重叠也算包含）"""
        if isinstance(other, Box):
            return self.contains(other)
        else:
            raise TypeError("Unsupported operand type for >=: 'Position' and " + type(other).__name__)
    def __le__(self, other):
        """是否包含（边界重叠也算包含）"""
        if isinstance(other, Box):
            return other.contains(self)
        else:
            raise TypeError("Unsupported operand type for <=: 'Position' and " + type(other).__name__)
    def contains_strictly(self, other):
        """是否包含（但边界重叠不算包含）"""
        if isinstance(other, Box):
            return self.left < other.left and self.up < other.up and self.right > other.right and self.down > other.down
        else:
            raise TypeError("Expecting 'other' an instance of 'Box', but " + type(other).__name__ + " encountered.")
    def __gt__(self, other):
        """是否包含（但边界重叠不算包含）"""
        if isinstance(other, Box):
            return self.contains_strictly(other)
        else:
            raise TypeError("Unsupported operand type for >: 'Position' and " + type(other).__name__)
    def __lt__(self, other):
        """是否包含（但边界重叠不算包含）"""
        if isinstance(other, Box):
            return other.contains_strictly(self)
        else:
            raise TypeError("Unsupported operand type for <: 'Position' and " + type(other).__name__)
        
    @staticmethod
    def union_boxes(boxes):
        """求多个盒子的并盒子"""
        if len(boxes) == 0:
            return Box.neg_inf() # 并是从负无穷盒子开始扩张
        lefts = [box.left for box in boxes]
        rights = [box.right for box in boxes]
        ups = [box.up for box in boxes]
        downs = [box.down for box in boxes]
        return Box(
            min(lefts),
            max(rights),
            min(ups),
            max(downs),
        )
    def union(self, other):
        """并操作"""
        return Box(
            min(self.left, other.left),
            max(self.right, other.right),
            min(self.up, other.up),
            max(self.down, other.down),
        )
    def __or__(self, other):
        """并操作"""
        return self.union(other)
    
    @staticmethod
    def intersection_boxes(boxes):
        """求多个盒子的交盒子"""
        if len(boxes) == 0:
            return Box.pos_inf() # 交是从正无穷盒子开始收束
        lefts = [box.left for box in boxes]
        rights = [box.right for box in boxes]
        ups = [box.up for box in boxes]
        downs = [box.down for box in boxes]
        return Box(
            max(lefts),
            min(rights),
            max(ups),
            min(downs),
        )
    def intersection(self, other):
        """交操作"""
        return Box(
            max(self.left, other.left),
            min(self.right, other.right),
            max(self.up, other.up),
            min(self.down, other.down),
        )
    def __and__(self, other):
        """交操作"""
        return self.intersection(other)
    
    def round(self):
        """取整"""
        return Box(left=int(round(self.left)), right=int(round(self.right)), up=int(round(self.up)), down=int(round(self.down)))
    
    def __str__(self):
        """字符串化"""
        if isinstance(self.left, numbers.Integral) and isinstance(self.right, numbers.Integral) and isinstance(self.up, numbers.Integral) and isinstance(self.down, numbers.Integral):
            return "Box(l=%d, r=%d, u=%d, d=%d)" % (self.left, self.right, self.up, self.down)
        else:
            return "Box(l=%.4f, r=%.4f, u=%.4f, d=%.4f)" % (self.left, self.right, self.up, self.down)
    def __repr__(self):
        return self.__str__()
    
    def draw(self, draw, type='\\', width=1.0, fill='none', color='red'):
        """绘制，type 为 '\\', 'x', 'box' 之一"""
        draw.stroke_color = wand.color.Color(color)
        draw.stroke_width = width
        print("draw box at %s" % str(self))
        
        if type == '\\':
            draw.line((self.left, self.up), (self.right, self.down))
        elif type == 'x':
            draw.line((self.left, self.up), (self.right, self.down))
            draw.line((self.right, self.up), (self.left, self.down))
        elif type == 'box':
            draw.fill_color = wand.color.Color(fill)
            draw.rectangle(left = self.left,
                           top = self.up,
                           right = self.right,
                           bottom = self.down)
        else:
            raise ValueError("Invalid param 'type' provided: {}".format(type))


# In[11]:


# all([]) == True


# In[12]:


# any([]) == False


# ## [B3] 渲染元素类 `Element`

# In[13]:


class Element(object):
    """
    一个渲染元素
    """
    
    def __init__(self, position=None):
        """初始化渲染元素，此方法需要重载"""
        self.physical_box = None
        self.render_box = None
        self.position = Position(0, 0) if position is None else position
    
    def check_valid(self):
        """用于检查初始化的结果是否合法"""
        return self.render_box and self.physical_box and self.render_box >= self.physical_box
    
    def draw_debug(self, draw, offset=None, render=False, physical=True, type='box', render_color='yellow', physical_color='blue', width=1.0, use_position=True):
        """在指定位置绘制调试盒子，type 参数可以为 'x' 或者 'box'"""
        if offset is None or isinstance(offset, Position):
            offset = Position(0, 0) if offset is None else offset
            if use_position:
                offset += self.position
            if render:
                (self.render_box + offset).draw(draw, type=type, width=width, color=render_color)
            if physical:
                (self.physical_box + offset).draw(draw, type=type, width=width, color=render_color)
        else:
            raise TypeError("Expecting 'offset' None or an instance of 'Position', but " + type(offset).__name__ + " encountered.")
            
    def draw(self, draw, offset=None, debug=False, render=False, physical=True, type='box', render_color='yellow', physical_color='blue', width=1.0, use_position=True):
        """在指定位置绘制"""
        if offset is None or isinstance(offset, Position):
            offset = Position(0, 0) if offset is None else offset
            if use_position:
                offset += self.position
            self.draw_inner(draw, offset, debug=debug)
            if debug:
                self.draw_debug(draw, offset=offset, render=render, physical=physical, type=type, render_color=render_color, physical_color=physical_color, width=width, use_position=False)
        else:
            raise TypeError("Expecting 'offset' None or an instance of 'Position', but " + type(offset).__name__ + " encountered.")
            
    def draw_inner(self, draw, offset, debug=False):
        """在指定位置绘制的实现。此方法需要重载"""
        print('Warning: Element.draw() is an abstract method, please override it.')
        
    def __str__(self):
        """字符串化"""
        return "Element<%s>{phy=%s, ren=%s, pos=%s}" % (type(self).__name__, str(self.physical_box), str(self.render_box), str(self.position))
    def __repr__(self):
        return self.__str__()
        
    def render_to_path(self, path, debug=False, padding=1.0):
        """单独渲染出结果，并输出到 png"""
        with Drawing() as draw:
            draw_position = -self.render_box.get_upleft() + Position(padding, padding)
            self.draw(draw, draw_position, debug=debug, use_position=False)
            size = self.render_box.expand(padding).size().round()
            with Image(width = size.x, height = size.y, background = wand.color.Color('none')) as img:
                draw(img)
                img.format = 'png'
                img.save(filename=path)
                return draw_position, size


# ## [B4] 渲染元素容器 `ElementContainer`

# In[14]:


class ElementContainer(Element):
    """
    可容纳和绘制多个渲染元素的容器
    """
    
    def __init__(self, from_array=None, position=None):
        """初始化"""
        super().__init__(position=position)
        if from_array is None:
            self.elements = []
            self.physical_box = Box.neg_inf()
            self.render_box = Box.neg_inf()
        else:
            self.elements = from_array
            self.physical_box = Box.neg_inf()
            self.render_box = Box.neg_inf()
            self._recalculate_boxes()
        
    def length(self):
        """获取存储的元素数目"""
        return len(self.elements)
    def __len__(self):
        return self.length()
    def is_empty(self):
        """是否为空"""
        return self.length() == 0
    
    def add(self, ele):
        """尾增元素"""
        is_empty = self.is_empty()
        self.elements.append(ele)
        self.physical_box = (ele.physical_box + ele.position) if is_empty else (self.physical_box | (ele.physical_box + ele.position))
        self.render_box = (ele.render_box + ele.position) if is_empty else (self.render_box | (ele.render_box + ele.position))
    def insert(self, index, ele):
        """添加元素"""
        is_empty = self.is_empty()
        self.elements.insert(index, ele)
        self.physical_box = (ele.physical_box + ele.position) if is_empty else (self.physical_box | (ele.physical_box + ele.position))
        self.render_box = (ele.render_box + ele.position) if is_empty else (self.render_box | (ele.render_box + ele.position))
        
    def get(self, index):
        """获取元素"""
        return self.elements[index]
    
    def _recalculate_boxes(self):
        if self.is_empty():
            self.physical_box = Box.neg_inf()
            self.render_box = Box.neg_inf()
        else:
            self.physical_box = Box.union_boxes([ele.physical_box + ele.position for ele in self.elements])
            self.render_box = Box.union_boxes([ele.render_box + ele.position for ele in self.elements])
    def remove(self, index):
        """删除元素"""
        self.elements.pop(index)
        self._recalculate_boxes()
        
    def draw(self, draw, offset=None, debug=False, render=False, physical=True, type='box', render_color='yellow', physical_color='blue', width=1.0, use_position=True):
        """在指定位置绘制"""
        # 相较于 Element.draw() 多了一个 is_empty 的判断，如果为空就不画盒子。
        if offset is None or isinstance(offset, Position):
            offset = Position(0, 0) if offset is None else offset
            if use_position:
                offset += self.position
            self.draw_inner(draw, offset, debug=debug)
            if self.physical_box.is_valid() and debug:
                self.draw_debug(draw, offset=offset, render=render, physical=physical, type=type, render_color=render_color, physical_color=physical_color, width=width, use_position=False)
        else:
            raise TypeError("Expecting 'offset' None or an instance of 'Position', but " + type(offset).__name__ + " encountered.")
        
    def __str__(self):
        """字符串化"""
        return "%s:" % (super().__str__()) + "".join([("\n----%02d#:%s" % (i, str(it))) for i, it in enumerate(self.elements)])
    def __repr__(self):
        return self.__str__()
        
    def draw_inner(self, draw, offset, debug=False):
        for ele in self.elements:
            if ele.physical_box.is_valid():
                ele.draw(draw, offset=offset, debug=debug)


# ## [B5] 特殊符号绘制器 `ESymbol`

# In[15]:


class SymbolManager:
    """初始化随机参数工具"""
    
    def __init__(self, symbol_info_path):
        """初始化"""
        
        # read symbol_info
        excel_sheets = pd.read_excel(symbol_info_path, sheet_name=None)
        self.info = excel_sheets['Main']
        self.info = self.info[self.info['enabled']].copy()
        self.grouped = self.info.groupby('symbol_name')

        # add path
        self.info['font_file_path'] = ''
        for index in self.info.index:
            self.info.loc[index, 'font_file_path'] = os.path.join('..\\wand_assets', self.info.loc[index, 'font_file'])
    
    def get_element(self, PARAMS, name, appog=False, position=None):
        group = self.grouped.get_group(name)
        row = group.sample(n=1, weights='p_weight').iloc[0]
        return ESymbol(PARAMS, appog, row, position)
    
    def debug_draw_element(self, PARAMS, name, output_png='tmp.png', appog=False, position=None):
        group = self.grouped.get_group(name)
        row = group.sample(n=1, weights='p_weight').iloc[0]
        print(row)
        temp = ESymbol(PARAMS, appog, row, position)
        temp.render_to_path(output_png, debug=True)


# In[16]:


class ESymbol(Element):
    def __init__(self, PARAMS, appog, info, position=None):
        super().__init__(position=position)
        PARAMS.update_item('symbol')
        self.info = info
        
        # get varsize
        self.varsize = 1.0
        if info['varsize_type'] == 'note_deco':
            self.varsize = PARAMS.get('NOTE_DECO_VARSIZE')
        self.font_size = info['basic_font_size'] * self.varsize
        self.width = self.font_size * info['width(px_per_pt)']
        
        self.physical_box = Box(
            left = -self.width / 2,
            right = self.width / 2,
            up = -info['height(px_per_pt)'] * self.font_size,
            down = 0,
        )
        self.render_box = self.physical_box.expand(20)
        assert self.check_valid()
        
    def draw_inner(self, draw, offset, debug=False):
        draw.font = self.info['font_file_path']
        draw.font_size = self.font_size
        draw.text_alignment = 'center'
        draw.stroke_color = wand.color.Color('none')
        draw.fill_color = wand.color.Color('black')
        x = self.info['x_offset(px_per_pt)'] * self.font_size + offset.x
        y = self.info['y_offset(px_per_pt)'] * self.font_size + offset.y
        print("draw Symbol '%s', x=%.2f, y=%.2f, font_name='%s', font_size=%.2f" % (self.info['symbol_name'], x, y, self.info['font_file'], self.font_size)) if debug else None
        draw.text(int(round(x)), int(round(y)), self.info['character'])


# In[17]:


PARAMS_.symm = SymbolManager('..\\wand_assets\\symbols.xlsx')


# In[18]:


# PARAMS_.symm.debug_draw_element(PARAMS_, 'Tremor', output_png='tmp_symm_debug.png')


# ## [B1] 音符绘制

# ### [B1a] 下放 `underline` 信息，以便绘制 `dot_bottom`

# In[19]:


# 根据存储在 mono_data 中的 underlines 信息，下放到 notes 中
def update_underline_for_notes(mono_data):
    
    mono_data = copy.deepcopy(mono_data)
    if 'underlines' in mono_data.keys():
        for underline in mono_data['underlines']:
            for note in mono_data['notes'][underline['begin_id']:underline['end_id']+1]:
                if 'underline' not in note.keys():
                    note['underline'] = 1
                else:
                    note['underline'] += 1
    
    return mono_data


# ### [B1a'] 下放 `hints` 信息为 `note_decos` 信息

# In[20]:


# 根据存储在 mono_data 中的 underlines 信息，下放到 notes 中
def update_hint_for_notes(mono_data):
    
    mono_data = copy.deepcopy(mono_data)
    if 'hints' in mono_data.keys():
        for hint in mono_data['hints']:
            
            # deco_top
            if hint['hint_type'] in ['Fermata', 'Circle', 'Tremor', 'DownTriangle']:
                target = mono_data['notes'][hint['align_id']]
                if 'deco_top' not in target.keys():
                    target['deco_top'] = [hint['hint_type'], ]
                else:
                    target['deco_top'].append(hint['hint_type'])
                    
            # deco_left
            if hint['hint_type'] in ['Sharp', 'Flat', 'Natural']:
                target = mono_data['notes'][hint['align_id']]
                if 'deco_left' not in target.keys():
                    target['deco_left'] = [hint['hint_type'], ]
                else:
                    target['deco_left'].append(hint['hint_type'])
    
    return mono_data


# ### [B1b] 音符字符元素 `ENoteCharacter`

# In[21]:


class ENoteCharacter(Element):
    def __init__(self, character, PARAMS, appog, note_type, position=None):
        super().__init__(position=position)
        
        self.character = character
        self.font_size = PARAMS.get_a('NOTE_FONT_SIZE', appog)
        self.font_info = PARAMS.get_active_font('number')
        self.note_type = note_type
        
        side_width = self.font_info['width(px_per_pt)'] * self.font_size / 2
        self.physical_box = Box(
            left = -side_width,
            right = side_width,
            up = -self.font_info['height(px_per_pt)'] * self.font_size,
            down = 0,
        )
        self.render_box = self.physical_box.expand(10)
        
        assert self.check_valid()
        
    def draw_inner(self, draw, offset, debug=False):
        
        draw.font = self.font_info['font_file_path']
        draw.font_size = self.font_size
        draw.text_alignment = 'center'
        draw.stroke_color = wand.color.Color('none')
        draw.fill_color = wand.color.Color('black')
        x = self.font_info['x_offset(px_per_pt)'] * self.font_size + offset.x
        y = self.font_info['y_offset(px_per_pt)'] * self.font_size + offset.y
        if self.note_type == 'bar':
            y += self.font_info['bar_offset(px_per_pt)'] * self.font_size
        x = int(round(x))
        y = int(round(y))
        print('draw NoteCharacter, x=%.2f, y=%.2f, font_size=%.2f, char="%s"' % (x, y, self.font_size, self.character)) if debug else None
        draw.text(x, y, self.character)


# In[22]:


# temp = ENoteCharacter("3", PARAMS_, True, note_type='number')
# temp.render_to_path("tmp.png", debug=True)


# ### [B1c] 音符点元素 `ENoteDot`

# In[23]:


class ENoteDot(Element):
    def __init__(self, PARAMS, appog, position=None):
        super().__init__(position=position)
        PARAMS.update_item('dot')
        
        self.dot_radius = PARAMS.get_a('DOT_RADIUS', appog)
        self.physical_box = Box(
            left = -self.dot_radius,
            right = self.dot_radius,
            up = -self.dot_radius,
            down = self.dot_radius,
        )
        self.render_box = self.physical_box.expand(self.dot_radius)
        
        assert self.check_valid()
        
    def draw_inner(self, draw, offset, debug=False):
        draw.stroke_color = wand.color.Color('none')
        draw.fill_color = wand.color.Color('black')
        x, y = offset.totuple()
        draw.circle((x, y), # Center point
                    (x + self.dot_radius, y)) # Perimeter point


# ### [B1d] 音符附加元素 `ENoteDecoration`

# In[24]:


# class END_Sharp(Element):
#     def __init__(self, PARAMS, appog, position=None):
#         super().__init__(position=position)
#         PARAMS.update_item('symbol')
        
#         self.deco_size = PARAMS.get_a('NOTE_DECO_VARSIZE', appog)
#         self.stroke_width = PARAMS.get_a('NOTE_DECO_STROKE_WIDTH', appog)
#         self.deco_aspect = PARAMS.get('NOTE_DECO_ASPECT')
        
#         self.up1 = 0.6 * self.deco_size * 16
#         self.up2 = 1.6 * self.deco_size * 16
#         self.right1 = 0.4 * self.deco_size * 16
#         self.right2 = 1.0 * self.deco_size * 16
#         self.sharp_tan = PARAMS.get('SHARP_SHEAR_TANGENT')
        
#         self.physical_box = Box(
#             left = -self.right2,
#             right = self.right2,
#             up = -(self.up2 + self.right1 * self.sharp_tan),
#             down = self.up2 + self.right1 * self.sharp_tan,
#         )            
#         self.render_box = self.physical_box.expand(5)
#         assert self.check_valid()
        
#     def draw_inner(self, draw, offset, debug=False):
#         draw.stroke_color = wand.color.Color('black')
#         draw.stroke_width = self.stroke_width
#         draw.fill_color = wand.color.Color('none')
#         x, y = offset.totuple()
#         # verticals
#         draw.line((x+self.right1, y-self.up2-self.right1*self.sharp_tan),
#                   (x+self.right1, y+self.up2-self.right1*self.sharp_tan))
#         draw.line((x-self.right1, y-self.up2+self.right1*self.sharp_tan),
#                   (x-self.right1, y+self.up2+self.right1*self.sharp_tan))
#         # horizontals
#         draw.line((x-self.right2, y-self.up1+self.right2*self.sharp_tan),
#                   (x+self.right2, y-self.up1-self.right2*self.sharp_tan))
#         draw.line((x-self.right2, y+self.up1+self.right2*self.sharp_tan),
#                   (x+self.right2, y+self.up1-self.right2*self.sharp_tan))


# In[25]:


# temp = END_Sharp(PARAMS_, appog=False)
# temp.render_to_path("tmp.png", debug=True)


# ### [B1d] 音符元素 `ENote`

# In[26]:


def note_parse_char(note_data):
    if note_data['note_type'] == 'bar':
        return '-'
    elif note_data['note_type'] == 'number':
        return str(note_data['number'])
    elif note_data['note_type'] == 'nopitch':
        return note_data['symbol']
    else:
        assert False
        
class ENote(Element):
    
    def __init__(self, note_data, PARAMS, appog, position=None):
        super().__init__(position=position)
        PARAMS.update_item('note')
        self.data = note_data
        self.note_character = ENoteCharacter(note_parse_char(note_data), PARAMS, appog, note_type=note_data['note_type'])
        
        # upper_dots
        y_top_pos = self.note_character.physical_box.up
        self.upper_dots = ElementContainer()
        if 'dot_top' in note_data.keys():
            y_top_pos -= PARAMS.get_a('DOT_TOP_YOFFSET_FROM_TOP', appog)
            self.upper_dots.position = Position(0, y_top_pos)
            off_y = 0
            for i in range(note_data['dot_top']):
                self.upper_dots.add(ENoteDot(PARAMS, appog, position=Position(0, off_y)))
                off_y -= PARAMS.get_a('DOT_STACKING_OFFSET', appog)
            y_top_pos += off_y
        
        # bottom_dots
        self.bottom_dots = ElementContainer()
        if 'dot_bottom' in note_data.keys():
            ypos = self.note_character.physical_box.down + PARAMS.get_a('UNDERLINE_OR_DOT_BOTTOM_YOFFSET_FROM_BPOS', appog)
            if 'underline' in note_data.keys():
                ypos += PARAMS.get_a('UNDERLINE_STACKING_OFFSET', appog) * note_data['underline']
            self.bottom_dots.position = Position(0, ypos)
            off_y = 0
            for i in range(note_data['dot_bottom']):
                self.bottom_dots.add(ENoteDot(PARAMS, appog, position=Position(0, off_y)))
                off_y += PARAMS.get_a('DOT_STACKING_OFFSET', appog)
                
        # right_dots
        self.right_dots = ElementContainer()
        if 'dot_right' in note_data.keys():
            xpos = self.note_character.physical_box.right + PARAMS.get_a('DOT_RIGHT_XOFFSET_FROM_SIDE', appog)
            ypos = self.note_character.physical_box.down - PARAMS.get_a('DOT_RIGHT_YOFFSET_FROM_BPOS', appog)
            self.right_dots.position = Position(xpos, ypos)
            off_x = 0
            for i in range(note_data['dot_right']):
                self.right_dots.add(ENoteDot(PARAMS, appog, position=Position(off_x, 0)))
                off_x += PARAMS.get_a('DOT_STACKING_OFFSET', appog)
                
        # deco_top
        self.deco_top = ElementContainer()
        if 'deco_top' in note_data.keys():
            y_top_pos -= PARAMS.get_a('NOTE_DECO_TOP_YOFFSET_FROM_TOP', appog)
            self.deco_top.position = Position(0, y_top_pos)
            off_y = 0
            for i in range(len(note_data['deco_top'])):
                PARAMS.update_item('deco_top')
                new_ele = PARAMS.symm.get_element(PARAMS, note_data['deco_top'][i], appog=appog, position=Position(0, off_y))
                self.deco_top.add(new_ele)
                off_y -= (new_ele.physical_box.down - new_ele.physical_box.up) + PARAMS.get_a('NOTE_DECO_TOP_STACKING_OFFSET', appog)
            y_top_pos += off_y
            
        # deco_left
        x_left_pos = self.note_character.physical_box.left
        self.deco_left = ElementContainer()
        if 'deco_left' in note_data.keys():
            x_left_pos -= PARAMS.get_a('NOTE_DECO_LEFT_XSPACE_FROM_SIDE', appog)
            y = -PARAMS.get_a('NOTE_DECO_LEFT_YOFFSET_FROM_SIDE', appog)
            self.deco_left.position = Position(x_left_pos, y)
            off_x = 0
            for i in range(len(note_data['deco_left'])):
                PARAMS.update_item('deco_left')
                new_ele = PARAMS.symm.get_element(PARAMS, note_data['deco_left'][i], appog=appog, position=None)
                off_x -= new_ele.physical_box.right
                new_ele.position = Position(off_x, 0)
                self.deco_left.add(new_ele)
                off_x += new_ele.physical_box.left - PARAMS.get_a('NOTE_DECO_LEFT_STACKING_SPACE', appog)
            x_left_pos += off_x
                
        elists = [self.note_character, self.upper_dots, self.bottom_dots, self.right_dots, self.deco_top, self.deco_left]
        self.physical_box = Box.union_boxes([ele.physical_box + ele.position for ele in elists])
        self.render_box = Box.union_boxes([ele.render_box + ele.position for ele in elists])
        assert self.check_valid()
        
    def draw_inner(self, draw, offset, debug=False):
        self.note_character.draw(draw, offset, debug=debug)
        self.upper_dots.draw(draw, offset, debug=debug)
        self.bottom_dots.draw(draw, offset, debug=debug)
        self.right_dots.draw(draw, offset, debug=debug)
        self.deco_top.draw(draw, offset, debug=debug)
        self.deco_left.draw(draw, offset, debug=debug)


# In[27]:


# temp = ENote(example_monophonic_seq_2['notes'][15], PARAMS_, appog=False)
# temp.render_to_path("tmp.png", debug=True)


# ## [B2] 歌词绘制

# ### [B2a] 歌词信息进行整合

# In[28]:


def compose_lyric_row(lyrics, note_count):
    
    composed = [{
        'left_brackets': 0,
        'lyric': "",
        'right_brackets': 0,
        'anno': "",
        'is_empty': True,
    } for _ in range(note_count)]
    last_align_at = None
    left_brackets = 0
    for lyric in lyrics:
        if lyric['lyric_type'] == 'Lyric':
            last_align_at = lyric['align_at']
            #print(type(composed[last_align_at]['left_brackets']))
            composed[last_align_at]['left_brackets'] += left_brackets
            composed[last_align_at]['lyric'] += lyric['lyric']
            composed[last_align_at]['is_empty'] = False
            left_brackets = 0
        elif lyric['lyric_type'] == 'Bracket':
            if lyric['orientation'] == 'forward':
                left_brackets += 1
            else: # lyric['orientation'] == 'backward'
                composed[last_align_at]['right_brackets'] += 1
        elif lyric['lyric_type'] == 'Anno':
            composed[last_align_at]['anno'] += lyric['anno']
        else:
            assert False
    return composed


# In[29]:


# temp_cll = compose_lyric_row(example_monophonic_seq_2['lyrics'][0], len(example_monophonic_seq_2['notes']))


# ### [B2b] 文本元素 `EText`

# In[30]:


class EText(Element):
    def __init__(self, text, PARAMS, position=None, align_type='first', space=0, bracket='not', size_ratio=1.0):
        """align_type 为 'first', 'left', 'center' 之一；space 的单位是相对于字符宽度的比例。"""
        super().__init__(position=position)
        
        PARAMS.update_item('text')
        self.text = text
        self.font_size = PARAMS.get('LYRIC_FONT_SIZE') * size_ratio
        self.font_info = PARAMS.get_active_font('hanzi')
        self.space = space
        
        self.font_xoffset = self.font_info['x_offset(px_per_pt)'] * self.font_size
        if bracket == 'not':
            self.font_width = self.font_info['width(px_per_pt)'] * self.font_size 
        elif bracket == 'left':
            self.font_width = self.font_info['lbracket_width(px_per_pt)'] * self.font_size
            self.font_xoffset += self.font_info['lbracket_xoffset(px_per_pt)'] * self.font_size
        elif bracket == 'right':
            self.font_width = self.font_info['rbracket_width(px_per_pt)'] * self.font_size
            self.font_xoffset += self.font_info['rbracket_xoffset(px_per_pt)'] * self.font_size
        else:
            raise ValueError("Invalid param 'bracket' provided: {}".format(bracket))
        self.text_width_sum = self.font_width * len(text)
        
        self.space_width = self.font_width * space
        self.space_width_sum = (0 if len(text) <= 1 else self.space_width * (len(text) - 1))
        
        if (align_type == 'first'):
            self.xmin = -(0 if len(text) == 0 else self.font_width * 0.5)
            self.xmax = self.xmin + self.text_width_sum + self.space_width_sum
        elif (align_type == 'left'):
            self.xmin = 0
            self.xmax = self.text_width_sum + self.space_width_sum
        elif (align_type == 'center'):
            half_width = (self.text_width_sum + self.space_width_sum) / 2
            self.xmin = -half_width
            self.xmax = half_width
        else:
            raise ValueError("Invalid param 'align_type' provided: {}".format(align_type))
        
        self.physical_box = Box(
            left = self.xmin,
            right = self.xmax,
            up = -self.font_info['height(px_per_pt)'] * self.font_size,
            down = 0,
        )
        self.render_box = self.physical_box.expand(18)
        if bracket != 'not':
            self.render_box = self.physical_box.expand_x(36)
        
        assert self.check_valid()
        
    def draw_inner(self, draw, offset, debug=False):
        
        draw.font = self.font_info['font_file_path']
        draw.font_size = self.font_size
        draw.text_alignment = 'left'
        draw.stroke_color = wand.color.Color('none')
        draw.fill_color = wand.color.Color('black')
        x = self.font_xoffset + offset.x + self.xmin
        y = self.font_info['y_offset(px_per_pt)'] * self.font_size + offset.y
        for char in self.text:
            print("draw Text '%s', x=%.2f, y=%.2f, font_size=%.2f" % (char, x, y, self.font_size)) if debug else None
            draw.text(int(round(x)), int(round(y)), char)
            x += self.font_width + self.space_width


# ### [B2c] 歌词项目元素 `ELyricItem`

# In[31]:


class ELyricItem(Element):
    def __init__(self, lyric_data, PARAMS, position=None, align_note_id=-1):
        super().__init__(position=position)
        
        self.data = lyric_data
        self.font_size = PARAMS.get('LYRIC_FONT_SIZE')
        self.font_info = PARAMS.get_active_font('hanzi')
        self.align_note_id = align_note_id
        
        if len(lyric_data['lyric']) > 0:
            self.lyric_main = EText(lyric_data['lyric'], PARAMS)
            self.left = self.lyric_main.physical_box.left
            self.right = self.lyric_main.physical_box.right
        else:
            self.lyric_main = None
            self.left = 0
            self.right = 0
            
        if len(lyric_data['anno']) > 0:
            off_x = self.right + PARAMS.get('LYRIC_ANNO_XOFFSET_FROM_BPOS')
            off_y = -PARAMS.get('LYRIC_ANNO_YOFFSET_FROM_BPOS')
            size_ratio = PARAMS.get('LYRIC_ANNO_FONT_RATIO')
            self.lyric_anno = EText(lyric_data['anno'], PARAMS, position=Position(off_x, off_y), align_type="left", size_ratio=size_ratio)
            self.right = off_x + self.lyric_anno.physical_box.right
        else:
            self.lyric_anno = None
            
        if lyric_data['left_brackets']:
            off_x = self.left + PARAMS.get('LYRIC_LBRACKET_XOFFSET')
            self.lyric_lbracket = EText('（', PARAMS, position=Position(off_x, 0), align_type="center", bracket='left')
            self.left = off_x + self.lyric_lbracket.physical_box.left
        else:
            self.lyric_lbracket = None
            
        if lyric_data['right_brackets']:
            off_x = self.right + PARAMS.get('LYRIC_RBRACKET_XOFFSET')
            self.lyric_rbracket = EText('）', PARAMS, position=Position(off_x, 0), align_type="center", bracket='right')
            self.right = off_x + self.lyric_rbracket.physical_box.right
        else:
            self.lyric_rbracket = None
        
        self.elements = [ele for ele in [self.lyric_main, self.lyric_anno, self.lyric_lbracket, self.lyric_rbracket] if not (ele is None)]
        boxes = [ele.physical_box + ele.position for ele in self.elements]
        self.physical_box = Box.union_boxes(boxes)
        boxes = [ele.render_box + ele.position for ele in self.elements]
        self.render_box = Box.union_boxes(boxes)
        
        assert self.check_valid()
        
    def draw_inner(self, draw, offset, debug=False):
        
        for ele in self.elements:
            ele.draw(draw, offset, debug=debug)


# In[32]:


# temp = ELyricItem(temp_cll[0], PARAMS_)


# In[33]:


# temp.render_to_path("tmp.png", debug=True)


# ## [B3] 其他元素绘制

# ### [B3a] 小节线元素 `EBarline`

# In[34]:


class EBarline(Element):
    def __init__(self, barline_data, PARAMS, position=None):
        super().__init__(position=position)
        PARAMS.update_item('barline')
        
        self.data = barline_data
        self.stroke_width = PARAMS.get('BARLINE_STROKE_WIDTH')
        self.top_y = PARAMS.get('BARLINE_TOP_Y')
        self.bottom_y = PARAMS.get('BARLINE_BOTTOM_Y')
        
        if (barline_data['barline_type'] == 'Simple') or (barline_data['barline_type'] == 'Fin'):
            self.physical_box = Box(
                left = -self.stroke_width / 2,
                right = self.stroke_width / 2,
                up = self.top_y,
                down = self.bottom_y,
            ).expand(self.stroke_width / 2)
        else:
            raise ValueError("Invalid param 'barline_type' provided: {}".format(barline_data['barline_type']))
            
        self.render_box = self.physical_box.expand(4)
        
        assert self.check_valid()
        
    def draw_inner(self, draw, offset, debug=False):
        draw.stroke_color = wand.color.Color('black')
        draw.stroke_width = self.stroke_width
        draw.fill_color = wand.color.Color('none')
        x, y = offset.totuple()
        if (self.data['barline_type'] == 'Simple') or (self.data['barline_type'] == 'Fin'):
            draw.line((x, y+self.top_y), (x, y+self.bottom_y))
        else:
            raise ValueError("Invalid param 'barline_type' provided: {}".format(self.data['barline_type']))


# In[35]:


# temp = EBarline({
#     'barline_type': 'Simple',
#     'pos_id': 19,
# }, PARAMS_)


# In[36]:


# temp.render_to_path("tmp.png", debug=False)


# ### [B3b] 下划线元素 `EUnderline`

# In[37]:


class EUnderline(Element):
    def __init__(self, length, PARAMS, appog, position=None):
        super().__init__(position=position)
        PARAMS.update_item('underline')
        
        self.stroke_width = PARAMS.get_a('UNDERLINE_STROKE_WIDTH', appog)
        self.length = length
        self.physical_box = Box(
            left = -length / 2,
            right = length / 2,
            up = 0,
            down = 0,
        ).expand(self.stroke_width / 2)
        self.render_box = self.physical_box.expand(2)
        
        assert self.check_valid()
        
    def draw_inner(self, draw, offset, debug=False):
        draw.stroke_color = wand.color.Color('black')
        draw.stroke_width = self.stroke_width
        x, y = offset.totuple()
        draw.line((x - self.length / 2, y), (x + self.length / 2, y))


# In[38]:


def create_underlines(underline_rows, note_eles, PARAMS, appog=False):
    result = ElementContainer()
    ypos = +PARAMS.get_a('UNDERLINE_OR_DOT_BOTTOM_YOFFSET_FROM_BPOS', appog)
    off_y = 0
    for i, row in enumerate(underline_rows):
        for begin_id, end_id in row:
            begin_ele = note_eles[begin_id]
            end_ele = note_eles[end_id]
            begin_x = begin_ele.position.x + begin_ele.physical_box.left
            end_x = end_ele.position.x + end_ele.physical_box.right
            center_x = (begin_x + end_x) / 2
            result.add(EUnderline(end_x - begin_x, PARAMS, appog, position=Position(center_x, ypos)))
        ypos += PARAMS.get_a('UNDERLINE_STACKING_OFFSET', appog)
    return result


# ### [B3c] 拍号数字 `ETimeSignatureNumber`、拍号元素 `ETimeSignature`

# In[39]:


class ETimeSignatureNumber(Element):
    def __init__(self, number, PARAMS, position=None, y_stretch_ratio=1.0):
        super().__init__(position=position)
        
        self.character = number
        self.font_size = PARAMS.get('TIMESIG_FRAC_FONT_SIZE')
        self.font_info = PARAMS.get_active_font('timesig_num')
        self.y_stretch_ratio = y_stretch_ratio
        
        side_width = self.font_info['width(px_per_pt)'] * self.font_size / 2
        self.physical_box = Box(
            left = -side_width,
            right = side_width,
            up = -self.font_info['height(px_per_pt)'] * self.font_size * self.y_stretch_ratio,
            down = 0,
        )
        self.render_box = self.physical_box.expand(10)
        
        assert self.check_valid()
        
    def draw_inner(self, draw, offset, debug=False):
        
        with Drawing() as inner_draw:
            
            side_width = self.font_info['width(px_per_pt)'] * self.font_size / 2
            temp_render_box = Box(
                left = -side_width,
                right = side_width,
                up = -self.font_info['height(px_per_pt)'] * self.font_size,
                down = 0,
            )
            size = temp_render_box.size().round()
            leftup = temp_render_box.get_upleft()
            
            inner_draw.font = self.font_info['font_file_path']
            inner_draw.font_size = self.font_size
            inner_draw.text_alignment = 'center'
            inner_draw.stroke_color = wand.color.Color('none')
            inner_draw.font_stetch = 'expanded'
            inner_draw.fill_color = wand.color.Color('black')

            x = self.font_info['x_offset(px_per_pt)'] * self.font_size - leftup.x
            y = self.font_info['y_offset(px_per_pt)'] * self.font_size - leftup.y
            x = int(round(x))
            y = int(round(y))
            print('draw TimeSignatureNumber, x=%.2f, y=%.2f, font_size=%.2f, char="%s"' % (x, y, self.font_size, self.character)) if debug else None
            inner_draw.text(x, y, self.character)

            with Image(width=size.x, height=size.y, background=wand.color.Color('none')) as inner_img:
                inner_draw(inner_img)
                inner_img.format = 'png'
                img_width = inner_img.width
                img_height = int(inner_img.height * self.y_stretch_ratio)
                inner_img.transform(resize='{}x{}!'.format(img_width, img_height))
                
                draw.composite('over',
                               offset.x + leftup.x,
                               offset.y + leftup.y * self.y_stretch_ratio,
                               inner_img.width,
                               inner_img.height,
                               inner_img,
                              )


# In[40]:


class ETimeSignature(Element):
    def __init__(self, timesig_data, PARAMS, position=None):
        super().__init__(position=position)
        PARAMS.update_item('timesig')
        
        self.data = timesig_data
        content_y = -PARAMS.get('TIMESIG_FRAC_YOFFSET')
        self.content = ElementContainer(position=Position(0, content_y))
        
        if (timesig_data['ts_type'] == 'fraction'):
            
            self.font_size = PARAMS.get('TIMESIG_FRAC_FONT_SIZE')
            self.font_info = PARAMS.get_active_font('timesig_num')
            
            y_stretch_ratio = PARAMS.get('TIMESIG_FRAC_FONT_STRETCH_RATIO')
            top_ypos = -PARAMS.get('TIMESIG_FRAC_TOP_SPACE')
            self.content.add(ETimeSignatureNumber(timesig_data['ts_top'], PARAMS, position=Position(0, top_ypos), y_stretch_ratio=y_stretch_ratio))
            
            y_stretch_ratio = PARAMS.get('TIMESIG_FRAC_FONT_STRETCH_RATIO')
            bottom_ypos = PARAMS.get('TIMESIG_FRAC_BOTTOM_SPACE') + self.font_info['height(px_per_pt)'] * self.font_size * y_stretch_ratio
            self.content.add(ETimeSignatureNumber(timesig_data['ts_bottom'], PARAMS, position=Position(0, bottom_ypos), y_stretch_ratio=y_stretch_ratio))
            
            width = self.font_info['width(px_per_pt)'] * self.font_size + PARAMS.get('TIMESIG_FRAC_SIDE_STICK_OUT') * 2
            self.content.add(EUnderline(width, PARAMS, False))
            
        elif (timesig_data['ts_type'] == 'rubato'):
            
            self.font_size = PARAMS.get('TIMESIG_RUBATO_FONT_SIZE')
            self.font_info = PARAMS.get_active_font('hanzi')
            
            ypos = PARAMS.get('TIMESIG_RUBATO_Y_POS')
            self.content.add(EText('艹', PARAMS, align_type='center', position=Position(0, ypos)))
            
        else:
            raise ValueError("Invalid param 'ts_type' provided: {}".format(timesig_data['ts_type']))
            
        self.physical_box = self.content.physical_box + self.content.position
        self.render_box = self.content.render_box + self.content.position
        
        assert self.check_valid()
        
    def draw_inner(self, draw, offset, debug=False):
        
        self.content.draw(draw, offset, debug=debug)


# In[41]:


# temp = ETimeSignature(example_monophonic_seq_0['time_signatures'][0], PARAMS_)


# In[42]:


# temp.content


# In[43]:


# temp.content.elements[0].font_info


# In[44]:


# temp.render_to_path("tmp.png", debug=True)


# ## [B4] 音符序列绘制（准备工作）

# ### [B4a] 将 `underline` 信息准备好，配置好各线所在行

# In[45]:


def bmes_to_intervals(bmes_list, offset=0):
    result = []
    start_index = None

    for i, mark in enumerate(bmes_list):
        if mark == 'B':
            start_index = i
        elif mark == 'E':
            if start_index is not None:
                result.append((start_index + offset, i + offset))
            start_index = None
        elif mark == 'S':
            result.append((i, i))
        elif mark == 'I':
            if start_index is not None:
                result.append((start_index + offset, i + offset))
            start_index = i

    return result

# interval -> layered BMES -> layered interval
def interval_layering_underline(underlines, note_number, debug=False, line_offset=0):
    '''
    line_offset 为行首音符下标，为多行曲目作偏移用。如果要作偏移，应该仅给入此行的 note_number 数目。
    '''
    underlines = sorted(underlines, key=lambda x: (x['end_id'] - x['begin_id']), reverse=True)
    
    if len(underlines) == 0:
        return []
    
    def get_underline_row(underline_lines, underline, line_offset):
        begin_id = underline['begin_id'] - line_offset
        end_id = underline['end_id'] - line_offset
        row = 0
        while row < len(underline_lines):
            next_row_flag = False
            for i in range(begin_id, end_id+1):
                if underline_lines[row][i] != '':
                    next_row_flag = True
                    break
            if next_row_flag:
                row += 1
            else:
                return row
                
        return -1
    
    underline_lines = [['' for i in range(note_number)]]
    for underline in underlines:
        begin_id = underline['begin_id'] - line_offset
        end_id = underline['end_id'] - line_offset
        #print(begin_id, end_id)
        row = get_underline_row(underline_lines, underline, line_offset)
        if row < 0:
            #print('*')
            underline_lines.append(['' for i in range(note_number)])
            row = -1
        if begin_id == end_id:
            underline_lines[row][begin_id] = 'S'
        else:
            underline_lines[row][begin_id] = 'B'
            for i in range(begin_id+1, end_id):
                underline_lines[row][i] = 'M'
            underline_lines[row][end_id] = 'E'
            
    print(underline_lines)  if debug else None
        
    return [bmes_to_intervals(underline_line) for underline_line in underline_lines]


# In[46]:


# interval_layering_underline(example_monophonic_seq_2['underlines'], len(example_monophonic_seq_2['notes']), debug=True)


# ### [B4b] 将 `curve` 信息准备好，配置好各线所在行

# In[47]:


# interval -> layered BMESI -> layered interval
def interval_layering_curve(curves, note_number, debug=False, line_offset=0):
    '''
    line_offset 为行首音符下标，为多行曲目作偏移用。如果要作偏移，应该仅给入此行的 note_number 数目。
    '''
    curves = sorted(curves, key=lambda x: (x['end_id'] - x['begin_id']), reverse=False)
    
    if len(curves) == 0:
        return []
    
    def get_curve_row(curve_lines, curve, line_offset):
        begin_id = curve['begin_id'] - line_offset + 1
        end_id = curve['end_id'] - line_offset + 1
        # print("innder, len(curve_lines)=%d, begin_id=%d, end_id=%d" % (len(curve_lines), begin_id, end_id))
        row = 0
        while row < len(curve_lines):
            next_row_flag = False
            for i in range(begin_id, end_id+1):
                # print('1st', len(curve_lines), row)
                # print('2nd', len(curve_lines[row]), i)
                if curve_lines[row][i] != '':
                    if i == begin_id and curve_lines[row][i] == 'E':
                        continue
                    elif i == end_id and curve_lines[row][i] == 'B':
                        continue
                    next_row_flag = True
                    break
            if next_row_flag:
                row += 1
            else:
                return row
                
        return -1
    
    curve_lines = [['' for i in range(note_number + 2)]]
    for curve in curves:
        begin_id = curve['begin_id'] - line_offset + 1
        end_id = curve['end_id'] - line_offset + 1
        if ('open_type' in curve.keys()):
            if curve['open_type'] == 'right_open':
                end_id = note_number + 1
            elif curve['open_type'] == 'left_open':
                begin_id = 0
        # print("note_number=%d, begin_id=%d, end_id=%d" % (note_number, begin_id, end_id))
        row = get_curve_row(curve_lines, curve, line_offset)
        if row < 0:
            #print('*')
            curve_lines.append(['' for i in range(note_number + 2)])
            row = -1
        if begin_id == end_id:
            assert False
        else:
            curve_lines[row][begin_id] = 'I' if curve_lines[row][begin_id] == 'E' else 'B'
            for i in range(begin_id+1, end_id):
                curve_lines[row][i] = 'M'
            curve_lines[row][end_id] = 'I' if curve_lines[row][end_id] == 'B' else 'E'
            
    print(curve_lines) if debug else None
        
    return [bmes_to_intervals(curve_line, offset=-1) for curve_line in curve_lines]


# In[48]:


# interval_layering_curve(example_monophonic_seq_2['curves'], len(example_monophonic_seq_2['notes']), debug=True)


# ### [B4c] 连音线元素 `ECurve`

# In[49]:


class ECurve(Element):
    def __init__(self, length, PARAMS, appog, position=None, open_type='full'):
        '''
        type in ['full', 'left_open', 'right_open']
        '''
        super().__init__(position=position)
        PARAMS.update_item('curve')
        
        self.stroke_width = PARAMS.get_a('CURVE_STROKE_WIDTH', appog)
        self.height = PARAMS.get_a('CURVE_HEIGHT', appog)
        length = length if open_type == 'full' else length * 2
        self.length = length
        self.open_type = open_type
        
        if (length < self.height):
            self.curve_type = 'min_arc'
            self.curve_param = {
                'r': self.length / 2
            }
        else:
            flag = False
            if length < PARAMS.get_a('CURVE_LONG_LENGTH_THRESHOLD', appog):
                flag = True
            else:
                phi_deg = PARAMS.get('CURVE_LONG_PHI')
                phi = np.radians(phi_deg)
                d = self.height * np.tan((np.pi - phi) / 2)

                if 2 * d >= length:
                    flag = True
            if flag:
                self.curve_type = 'arc'
                r = (length * length + 4 * self.height * self.height) / (8 * self.height)
                self.curve_param = {
                    'r': r,
                    'center_y': r - self.height,
                    'theta': np.degrees(np.arcsin((length) / (2 * r)))
                }
            else:
                self.curve_type = 'long_arc'
                phi_deg = PARAMS.get('CURVE_LONG_PHI')
                phi = np.radians(phi_deg)
                d = self.height * np.tan((np.pi - phi) / 2)
                r = d / np.sin(phi)
                self.curve_param = {
                    'r': r,
                    'center_x_l': -length / 2 + d,
                    'center_x_r': length / 2 - d,
                    'center_y': r - self.height,
                    'phi': phi_deg,
                }
        self.physical_box = Box(
            left = -length / 2 if open_type == 'full' or open_type == 'right_open' else 0,
            right = length / 2 if open_type == 'full' or open_type == 'left_open' else 0,
            up = -self.height,
            down = 0,
        ).expand(self.stroke_width / 2)
        self.render_box = self.physical_box.expand(2)
        
        assert self.check_valid()
        
    def draw_inner(self, draw, offset, debug=False):
        draw.stroke_color = wand.color.Color('black')
        draw.stroke_width = self.stroke_width
        draw.fill_color = wand.color.Color('none')
        x, y = offset.totuple()
        if self.curve_type == 'min_arc':
            r = self.curve_param['r']
            if self.open_type == 'full':
                draw.arc((x-r, y-r), (x+r, y+r), (-90 - 90, -90 + 90))
            elif self.open_type == 'left_open':
                draw.arc((x-r, y-r), (x+r, y+r), (-90 - 0, -90 + 90))
            elif self.open_type == 'right_open':
                draw.arc((x-r, y-r), (x+r, y+r), (-90 - 90, -90 + 0))
            else:
                raise ValueError("Invalid param 'open_type' provided: {}".format(self.open_type))
        elif self.curve_type == 'arc':
            r = self.curve_param['r']
            y_c = self.curve_param['center_y']
            theta = self.curve_param['theta']
            if self.open_type == 'full':
                draw.arc((x-r, y+y_c-r), (x+r, y+y_c+r), (-90 - theta, -90 + theta))
            elif self.open_type == 'left_open':
                draw.arc((x-r, y+y_c-r), (x+r, y+y_c+r), (-90 - 0, -90 + theta))
            elif self.open_type == 'right_open':
                draw.arc((x-r, y+y_c-r), (x+r, y+y_c+r), (-90 - theta, -90 + 0))
            else:
                raise ValueError("Invalid param 'open_type' provided: {}".format(self.open_type))
        elif self.curve_type == 'long_arc':
            r = self.curve_param['r']
            x_cl = self.curve_param['center_x_l']
            x_cr = self.curve_param['center_x_r']
            y_c = self.curve_param['center_y']
            phi_deg = self.curve_param['phi']
            if self.open_type == 'full':
                draw.arc((x+x_cl-r, y+y_c-r), (x+x_cl+r, y+y_c+r), (-90 - phi_deg, -90))
                draw.line((x+x_cl, y-self.height), (x+x_cr, y-self.height))
                draw.arc((x+x_cr-r, y+y_c-r), (x+x_cr+r, y+y_c+r), (-90, -90 + phi_deg))
            elif self.open_type == 'left_open':
                draw.line((x+0, y-self.height), (x+x_cr, y-self.height))
                draw.arc((x+x_cr-r, y+y_c-r), (x+x_cr+r, y+y_c+r), (-90, -90 + phi_deg))
            elif self.open_type == 'right_open':
                draw.arc((x+x_cl-r, y+y_c-r), (x+x_cl+r, y+y_c+r), (-90 - phi_deg, -90))
                draw.line((x+x_cl, y-self.height), (x+0, y-self.height))
            else:
                raise ValueError("Invalid param 'open_type' provided: {}".format(self.open_type))
        else:
            raise ValueError("Invalid param 'curve_type' provided: {}".format(self.curve_type))


# In[50]:


def create_curves(curve_rows, note_eles, PARAMS, appog=False, left_x=20, right_x=1500):
    result = ElementContainer()
    for i, row in enumerate(curve_rows):
        for begin_id, end_id in row:
            
            if begin_id < 0 and end_id >= len(note_eles):
                assert False
            elif begin_id < 0: # and end_id < len(note_eles)
                
                end_ele = note_eles[end_id]
                end_x = end_ele.position.x + PARAMS.get_a('CURVE_XOFFSET_END', appog)
                
                section_up = float('inf')
                for note_ele in note_eles[:end_id+1]:
                    if note_ele.physical_box.up < section_up:
                        section_up = note_ele.physical_box.up
                ypos = section_up - PARAMS.get_a('CURVE_YOFFSET_FROM_TOP', appog) - i * PARAMS.get_a('CURVE_STACKING_OFFSET', appog)
                
                result.add(ECurve(end_x - left_x, PARAMS, appog, position=Position(left_x, ypos), open_type='left_open'))
                
            elif end_id >= len(note_eles): # and begin_id >= 0
                
                begin_ele = note_eles[begin_id]
                begin_x = begin_ele.position.x + PARAMS.get_a('CURVE_XOFFSET_BEGIN', appog)
                
                section_up = float('inf')
                for note_ele in note_eles[begin_id:]:
                    if note_ele.physical_box.up < section_up:
                        section_up = note_ele.physical_box.up
                ypos = section_up - PARAMS.get_a('CURVE_YOFFSET_FROM_TOP', appog) - i * PARAMS.get_a('CURVE_STACKING_OFFSET', appog)
                
                result.add(ECurve(right_x - begin_x, PARAMS, appog, position=Position(right_x, ypos), open_type='right_open'))
                
            else:
            
                begin_ele = note_eles[begin_id]
                end_ele = note_eles[end_id]
                begin_x = begin_ele.position.x + PARAMS.get_a('CURVE_XOFFSET_BEGIN', appog)
                end_x = end_ele.position.x + PARAMS.get_a('CURVE_XOFFSET_END', appog)

                section_up = float('inf')
                for note_ele in note_eles[begin_id:end_id+1]:
                    if note_ele.physical_box.up < section_up:
                        section_up = note_ele.physical_box.up
                ypos = section_up - PARAMS.get_a('CURVE_YOFFSET_FROM_TOP', appog) - i * PARAMS.get_a('CURVE_STACKING_OFFSET', appog)

                center_x = (begin_x + end_x) / 2
                result.add(ECurve(end_x - begin_x, PARAMS, appog, position=Position(center_x, ypos)))
                
    return result


# ### [B4d] 倚音序列 `EAppog`、倚音尾巴元素 `EAppogTail`

# In[51]:


def update_underline_appog(appog_notes):
    appog_notes = copy.deepcopy(appog_notes)
    for appog_note in appog_notes['notes']:
        appog_note['underline'] = 2
    return appog_notes

def appog_make_layout(notes):
    right = 0
    for note in notes.elements:
        note.position = Position(right, 0)
        right = right + note.physical_box.size().x + 4
    notes._recalculate_boxes()


# In[52]:


class EAppogTail(Element):
    def __init__(self, PARAM, position=None):
        super().__init__(position=position)
        self.radius = PARAM.get('APPOG_TAIL_RADIUS')
        self.stroke_width = PARAM.get('APPOG_TAIL_STROKE_WIDTH')
        self.physical_box = Box(
            left=0,
            up=0,
            right=self.radius,
            down=self.radius,
        )
        self.render_box = self.physical_box.expand(5)
        assert self.check_valid()
        
    def draw_inner(self, draw, offset, debug=False):
        draw.stroke_color = wand.color.Color('black')
        draw.stroke_width = self.stroke_width
        draw.fill_color = wand.color.Color('none')
        x, y = offset.totuple()
        r = self.radius
        draw.arc((x, y-r), (x+2*r, y+r), (90, 180))


# In[53]:


class EAppog(Element):
    
    def __init__(self, appog_data, PARAMS, position=None):
        super().__init__(position=position)
        PARAMS.update_item('appog')
        appog_data = update_underline_appog(appog_data)
        self.data = appog_data
        
        self.notes = ElementContainer()
        for note_data in appog_data['notes']:
             self.notes.add(ENote(note_data, PARAMS, True))
                
        appog_make_layout(self.notes)
        
        self.width = self.notes.physical_box.size().x
        if not self.notes.is_empty():
            self.notes.position = Position(-self.width / 2 - self.notes.elements[0].note_character.physical_box.left, 0)
        
        y1 = PARAMS.get_a('UNDERLINE_OR_DOT_BOTTOM_YOFFSET_FROM_BPOS', True)
        self.underline1 = EUnderline(self.width, PARAMS, True, position=Position(0, y1))
        y2 = y1 + PARAMS.get_a('UNDERLINE_STACKING_OFFSET', True)
        self.underline2 = EUnderline(self.width, PARAMS, True, position=Position(0, y2))
        self.appog_tail = EAppogTail(PARAMS, position=Position(0, y2))
                
        boxes = [ele.physical_box + ele.position for ele in [self.notes, self.underline1, self.underline2, self.appog_tail]]
        self.physical_box = Box.union_boxes(boxes)
        boxes = [ele.render_box + ele.position for ele in [self.notes, self.underline1, self.underline2, self.appog_tail]]
        self.render_box = Box.union_boxes(boxes)
        #self.render_box = self.render_box.expand(1000)
        assert self.check_valid()
        
    def draw_inner(self, draw, offset, debug=False):
        self.notes.draw(draw, offset, debug=debug)
        self.underline1.draw(draw, offset, debug=debug)
        self.underline2.draw(draw, offset, debug=debug)
        self.appog_tail.draw(draw, offset, debug=debug)


# In[54]:


# temp = EAppog(example_monophonic_seq_0['appogs'][0], PARAMS_)


# In[55]:


# temp.render_to_path("tmp.png", debug=False)


# In[56]:


# temp.appog_tail


# ## [B5] 单声部乐谱 `EMonophonic` 及其布局方法

# ### [B5a] 间隔对象 `Spacing`

# In[57]:


class Spacing(object):
    '''
    一个「间隔」对象。
    '''
    
    def __init__(self, width_l=0.0, width_r=0.0, k_l=1.0, k_r=1.0, w_l=0.7, w_r=0.7, v_l=0, v_r=0, c=1):
        '''
        初始化一个间隔。默认值为典型值。
        '''
        # k and w
        self.k_l = k_l
        self.k_r = k_r
        self.k = k_l * k_r
        self.w_l = w_l
        self.w_r = w_r
        self.w = w_l * (w_r ** 0.2)
        # b
        self.width_l = width_l
        self.width_r = width_r
        self.v_l = v_l
        self.v_r = v_r
        self.default_b = width_l + width_r + v_l + v_r
        self.current_b = self.default_b
        # c
        self.c = c
        
    @staticmethod
    def zero(width_l=0.0, width_r=0.0, k_l=0.0, k_r=0.0, w_l=0.0, w_r=0.0, v_l=0, v_r=0, c=1):
        '''
        得到一个零间隔（但仍然可以单独指定各项）
        '''
        return Spacing(width_l=width_l, width_r=width_r,
                       k_l=k_l, k_r=k_r, w_l=w_l, w_r=w_r, v_l=v_l, v_r=v_r, c=c)
        
    def calc_full(self, x_k, x_w):
        '''
        使用完整方法计算实际间隔，不使用内部的 c。
        '''
        dynamic = self.k * x_k + self.w * x_w
        if dynamic > self.current_b:
            return dynamic
        else:
            return self.current_b
        
    def calc(self, x_k):
        '''
        使用单变量方法计算实际间隔，使用内部的 c。
        '''
        dynamic = (self.k + self.w * self.c) * x_k 
        if dynamic > self.current_b:
            return dynamic
        else:
            return self.current_b
        
    def get_d(self):
        '''
        计算启发点。
        '''
        return self.current_b / (self.k + self.w * self.c)
    
    def copy(self):
        '''
        浅复制。
        '''
        return copy.copy(self)
        
    @staticmethod
    def group_require_b(spacing_sliced, b_target):
        '''
        对一组 Spacing 更新 current_b，以整体达到 b_target 间隔的要求。
        '''
        b_total = 0
        for spacing in spacing_sliced:
            b_total += spacing.current_b
        if b_total < b_target:
            avr_b_required = (b_target - b_total) / len(spacing_sliced)
            for spacing in spacing_sliced:
                spacing.current_b += avr_b_required
            
    @staticmethod
    def group_reset_b(spacing_sliced):
        '''
        对一组 Spacing 重置 current_b 为 default_b。
        '''
        for spacing in spacing_sliced:
            spacing.current_b = spacing.default_b
        
    @staticmethod
    def group_sort_by_d(spacing_sliced):
        '''
        对一组 Spacing 求它们按 d 增序排序后的 argsorted、sorted、和 d 结果。
        '''
        sorted_d_with_index = sorted((spacing.get_d(), idx) for idx, spacing in enumerate(spacing_sliced))
        sorted_ds, sorted_indices = zip(*sorted_d_with_index)
        sorted_spacings = [None for _ in range(len(sorted_indices))]
        for i, idx in enumerate(sorted_indices):
            sorted_spacings[i] = spacing_sliced[idx]

        return sorted_indices, sorted_spacings, sorted_ds
    
    def __str__(self):
        """字符串化"""
        return "Spacing(k=%.4f, w=%.4f, width_l=%.4f, width_r=%.4f, default_b=%.4f, current_b=%.4f, c=%.4f)" % \
            (self.k, self.w, self.width_l, self.width_r, self.default_b, self.current_b, self.c)
    def __repr__(self):
        return self.__str__()


# ### [B5b] 切片视图 SpacedSequenceView

# In[58]:


class SpacedSequenceView(object):
    '''
    包括对 MonophonicLayoutSequence 的切片及其「间隔」对象，及其对应在 matrix 中的切片。
    '''
    
    def __init__(self, layout_sequence, begin_seq_id, end_seq_id, y_target):
        '''
        初始化，会复制一份 spacings 并更新针对 lyric_rows 的 require 需求。
        '''
        self.layout_sequence = layout_sequence
        
        self.begin_seq_id = begin_seq_id
        self.end_seq_id = end_seq_id
        assert begin_seq_id >= 0 and end_seq_id < len(layout_sequence.sequence) and begin_seq_id <= end_seq_id
        self.spacings = [s.copy() for s in layout_sequence.raw_spacings[begin_seq_id:end_seq_id]]
        
        self.left_solid_b = max([-ele.physical_box.left for ele in layout_sequence.matrix[:, begin_seq_id] if ele is not None])
        self.right_solid_b = max([ele.physical_box.right for ele in layout_sequence.matrix[:, end_seq_id] if ele is not None])
        
        # do require b by lyrics
        for row_id in range(len(layout_sequence.lyric_rows)): # do require every lyric row
            last_lyric = None
            last_lyric_col_id = -1
            for col_id, lyric in enumerate(layout_sequence.matrix[row_id+1][begin_seq_id:end_seq_id]):
                
                if lyric is not None:
                    
                    if last_lyric_col_id >= 0:
                        spacing_slice = self.spacings[last_lyric_col_id:col_id]
                        required_b = last_lyric.physical_box.right - lyric.physical_box.left
                        Spacing.group_require_b(spacing_slice, required_b)
                        last_lyric = lyric
                        last_lyric_col_id = col_id
                    else:
                        last_lyric = lyric
                        last_lyric_col_id = col_id
                
        # print("SpacedSequenceView(begin_seq_id=%d, end_seq_id=%d)" % (begin_seq_id, end_seq_id))
        if begin_seq_id < end_seq_id:
            self.spacings_argsorted, self.spacings_sorted, self.sorted_ds = Spacing.group_sort_by_d(self.spacings)
            self.solved_x = self.solve_spacing_x(y_target)
        else:
            self.spacings_argsorted, self.spacings_sorted, self.sorted_ds = [], [], []
            self.solved_x = 0
        
        self.sequence_view = layout_sequence.sequence[begin_seq_id:end_seq_id+1]
        self.matrix_view = layout_sequence.matrix[:, begin_seq_id:end_seq_id+1]
        self.length = len(self.sequence_view)
            
    def solve_spacing_x(self, y_target):
        '''
        解出对于实际间隔 y_target 而言的 x 值。如果不存在这样的 x（这时候 b 之和大于 0），将返回 -1。
        不需要从外部调用，__init__中会自己调用。
        '''
        
        y_target -= self.left_solid_b + self.right_solid_b
        
        # undersized part
        b_sum = sum([s.current_b for s in self.spacings_sorted])
        if b_sum > y_target: # not found
            return -1
        elif b_sum == y_target: # y_target = b_sum
            return 0
        
        # curve part
        for i in range(len(self.sorted_ds)):
            x = self.sorted_ds[i]
            y_b_part = sum([s.current_b for s in self.spacings_sorted[i:]])
            y_x_muls = sum([s.k + s.w * s.c for s in self.spacings_sorted[:i]])
            y = y_b_part + y_x_muls * x
            if y >= y_target: # section locked
                return (y_target - y_b_part) / y_x_muls
            
        # oversized part
        y_x_muls_max = sum([s.k + s.w * s.c for s in self.spacings_sorted])
        return y_target / y_x_muls_max
        
    def debug_draw_line_chart(self, desired_y=None, x_extention=50, output_name='tmp_line_chart.png'):
        '''
        调试用，输出实际间隔折线图。
        '''
        extended_d = self.sorted_ds[-1] + x_extention
        xs = [0,]
        ys = [sum([s.current_b for s in self.spacings_sorted]),]
        for i, x in enumerate(self.sorted_ds + (extended_d,)):
            b_part = sum([s.current_b for s in self.spacings_sorted[i:]])
            x_part = sum([s.calc(x) for s in self.spacings_sorted[:i]])
            y = b_part + x_part
            xs.append(x)
            ys.append(y)
        fig, ax = plt.subplots(figsize=(5, 1.5))
        ax.plot(xs[:-1], ys[:-1], '-o', markersize=3)
        ax.plot(xs[-2:], ys[-2:])
        ax.set_ylim(bottom=0, top=desired_y * 2) if desired_y is not None else ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=150)
        xmin, xmax = ax.get_xlim()
        ax.hlines([desired_y], xmin, xmax, linestyle='--', color='r') if desired_y is not None else None
        fig.savefig(output_name)
        print("SpacedSequenceView's line chart saved to: %s" % output_name)
                
    def get_line_ready(self, PARAMS):
        '''
        计算并求解此视图中正确的位置，并输出带有 notes, appogs, barlines, timesigs 的视图。
        '''
        
        # 求解正确的 x 位置
        off_x = self.left_solid_b
        for matrix_item in self.matrix_view[:, 0]:
            if matrix_item is not None:
                matrix_item.position += Position(off_x, 0)
        for i, spacing in enumerate(self.spacings):
            off_x += spacing.calc(self.solved_x)
            for matrix_item in self.matrix_view[:, i+1]:
                if matrix_item is not None:
                    matrix_item.position += Position(off_x, 0)
        
        # 求解正确的 y 位置，以及打包 lyric_lines
        PARAMS.update_item('lyric_line')
        off_y = max([seq_ele.physical_box.down for seq_ele in self.matrix_view[0, :] if seq_ele.physical_box.is_valid()]) + PARAMS.get('LYRIC_LINE_SPACE')
        lyric_lines = ElementContainer()
        for lyric_row_in_matrix in self.matrix_view[1:]:
            PARAMS.update_item('lyric_line')
            lyric_line = ElementContainer()
            for lyric_item in lyric_row_in_matrix:
                if lyric_item is not None:
                    lyric_line.add(lyric_item)
            if lyric_line.physical_box.is_valid():
                off_y += -lyric_line.physical_box.up
            else:
                off_y += PARAMS.get('LYRIC_LINE_EMPTY_ROW_HEIGHT') / 2
            lyric_line.position += Position(0, off_y)
            lyric_lines.add(lyric_line)
            if lyric_line.physical_box.is_valid():
                off_y += lyric_line.physical_box.down + PARAMS.get('LYRIC_LINE_STACKING_SPACE')
            else:
                off_y += PARAMS.get('LYRIC_LINE_EMPTY_ROW_HEIGHT') / 2 + PARAMS.get('LYRIC_LINE_STACKING_SPACE')
        
        # 打包输出
        result = {
            'note': ElementContainer(),
            'appog': ElementContainer(),
            'barline': ElementContainer(),
            'timesig': ElementContainer(),
            'lyric_line': lyric_lines,
        }
        for seq_item in self.sequence_view:
            result[seq_item['type']].add(self.layout_sequence.sequence_data[seq_item['type']][seq_item['id']])
        
        return result


# ### [B5c] 整体布局 MonophonicLayoutSequence

# In[59]:


class MonophonicLayoutSequence(object):
    '''
    包含的内容：
    sequence 主序列，以及配套的 sequence_data 可按类别查询元素、note_pos_in_seq 可查询各个 note 在 sequence 中的下标；
    lyric_rows 各歌词行的歌词元素；
    matrix 矩阵，第一行是主序列，下面各行是对应的歌词。
    '''
    
    def __init__(self, mono_data, PARAMS):
        '''
        从 mono_data 初始化 MonophonicLayoutSequence。
        '''
        
        self.mono_data = mono_data
        
        self.sequence = []
        self.sequence_data = {key: [] for key in ['note', 'appog', 'barline', 'timesig']}
        self.note_pos_in_seq = np.arange(len(mono_data['notes']))
        
        # note
        for i, note_data in enumerate(mono_data['notes']):
            self.sequence_data['note'].append(ENote(note_data, PARAMS, False))
            self.sequence.append({'type': 'note', 'id': i, 'underline': note_data['underline'] if 'underline' in note_data.keys() else 0})
        # appog
        for i, appog_data in enumerate(mono_data['appogs']):
            ypos = PARAMS.get('APPOG_OFFSET_Y')
            if abs(ypos) > 9999999:
                print("WARNING MonophonicLayoutSequence", file=sys.stderr)
            self.sequence_data['appog'].append(EAppog(appog_data, PARAMS, position=Position(0, -ypos)))
            pos = self.note_pos_in_seq[appog_data['align_id']]
            if pos != -1:
                if appog_data['orientation'] == 'before':
                    self.sequence.insert(pos, {'type': 'appog', 'id': i, 'orientation': 'before'})
                    self.note_pos_in_seq[appog_data['align_id']:] += 1
                else: # appog_data['orientation'] == 'after':
                    self.sequence.insert(pos + 1, {'type': 'appog', 'id': i, 'orientation': 'after'})
                    self.note_pos_in_seq[appog_data['align_id']+1:] += 1
            else:
                raise ValueError("Invalid id of 'appogs['align_id']' provided, cannot find corresponding note: {}".format(appog_data['align_id']))
        # barline
        for i, barline_data in enumerate(mono_data['barlines']):
            self.sequence_data['barline'].append(EBarline(barline_data, PARAMS))
            if barline_data['pos_id'] == len(self.sequence_data['note']):
                self.sequence.append({'type': 'barline', 'id': i})
            else:
                pos = self.note_pos_in_seq[barline_data['pos_id']]
                if pos != -1:
                    # while pos < len(self.sequence) and self.sequence[pos]['type'] == 'appog' and self.sequence[pos]['orientation'] == 'after':
                    #     pos += 1
                    while pos > 0 and self.sequence[pos - 1]['type'] == 'appog' and self.sequence[pos - 1]['orientation'] == 'before':
                        pos -= 1
                    self.sequence.insert(pos, {'type': 'barline', 'id': i})
                    self.note_pos_in_seq[barline_data['pos_id']:] += 1
                else:
                    raise ValueError("Invalid id of 'barline['pos_id']' provided, cannot find corresponding note: {}".format(barline_data['pos_id']))
        # timesig
        for i, timesig_data in enumerate(mono_data['time_signatures']):
            self.sequence_data['timesig'].append(ETimeSignature(timesig_data, PARAMS))
            pos = self.note_pos_in_seq[timesig_data['pos_id']]
            if pos != -1:
                # while pos < len(self.sequence) and self.sequence[pos]['type'] == 'appog' and self.sequence[pos]['orientation'] == 'after':
                #     pos += 1
                while pos > 0 and self.sequence[pos - 1]['type'] == 'appog' and self.sequence[pos - 1]['orientation'] == 'before':
                    pos -= 1        
                self.sequence.insert(pos, {'type': 'timesig', 'id': i})
                self.note_pos_in_seq[timesig_data['pos_id']:] += 1
            else:
                raise ValueError("Invalid id of 'timesig['pos_id']' provided, cannot find corresponding note: {}".format(timesig_data['pos_id']))
        
        # lyric_rows, matrix
        composed_lyric_rows = [compose_lyric_row(lyric_row, len(mono_data['notes'])) for lyric_row in mono_data['lyrics']]
        self.matrix = np.empty(shape=(len(composed_lyric_rows) + 1, len(self.sequence)), dtype=object)
        self.matrix[0] = np.array([self.sequence_data[seq_item['type']][seq_item['id']] for seq_item in self.sequence], dtype=object)
        self.lyric_rows = []
        for row_i, composed_lyric_row in enumerate(composed_lyric_rows):
            cur_row = []
            for align_note_id, composed_lyric in enumerate(composed_lyric_row):
                if not composed_lyric['is_empty']:
                    ele = ELyricItem(composed_lyric, PARAMS, align_note_id=align_note_id)
                    cur_row.append(ele)
                    self.matrix[row_i + 1][self.note_pos_in_seq[align_note_id]] = ele
            self.lyric_rows.append(cur_row)
            
        self.raw_spacings = self.assamble_default_spacings(PARAMS)
        
    def assamble_default_spacings(self, PARAMS):
        '''
        装配初始的 spacings 配置，现在还没有考虑 lyric_rows 的 require 需求，因为在后面实际切割起来才会有。
        '''
        spacings = []

        last_item = self.sequence[0]
        last_ele = self.sequence_data[self.sequence[0]['type']][self.sequence[0]['id']]
        for item in self.sequence[1:]:
            ele = self.sequence_data[item['type']][item['id']]
            width_l=last_ele.physical_box.right
            width_r=-ele.physical_box.left

            PARAMS.update_item('spacing')
            if last_item['type'] == 'appog' and last_item['orientation'] == 'before':
                k_l = PARAMS.get('LAYOUT_APPOG_K')
                v_l = PARAMS.get('LAYOUT_APPOG_B')
                w_l = PARAMS.get('LAYOUT_GENERAL_W') * 0.1
            elif last_item['type'] == 'barline':
                k_l = PARAMS.get('LAYOUT_BARLINE_K')
                v_l = PARAMS.get('LAYOUT_BARLINE_B')
                w_l = PARAMS.get('LAYOUT_GENERAL_W') * 0.9
            elif last_item['type'] == 'timesig':
                k_l = PARAMS.get('LAYOUT_BARLINE_K')
                v_l = PARAMS.get('LAYOUT_BARLINE_B')
                w_l = PARAMS.get('LAYOUT_GENERAL_W') * 0.7
            else:
                k_l = PARAMS.get('LAYOUT_GENERAL_K')
                v_l = PARAMS.get('LAYOUT_GENERAL_B')
                if last_item['type'] == 'note':
                    w_l = 1 / (2 ** last_item['underline'])
                else:
                    w_l = PARAMS.get('LAYOUT_GENERAL_W')

            PARAMS.update_item('spacing')
            if item['type'] == 'appog' and item['orientation'] == 'after':
                k_r = PARAMS.get('LAYOUT_APPOG_K')
                v_r = PARAMS.get('LAYOUT_APPOG_B')
                w_r = PARAMS.get('LAYOUT_GENERAL_W') * 0.1
            elif item['type'] == 'barline':
                k_r = PARAMS.get('LAYOUT_BARLINE_K')
                v_r = PARAMS.get('LAYOUT_BARLINE_B')
                w_r = PARAMS.get('LAYOUT_GENERAL_W') * 0.9
            elif item['type'] == 'timesig':
                k_r = PARAMS.get('LAYOUT_BARLINE_K')
                v_r = PARAMS.get('LAYOUT_BARLINE_B')
                w_r = PARAMS.get('LAYOUT_GENERAL_W') * 0.7
            else:
                k_r = PARAMS.get('LAYOUT_GENERAL_K')
                v_r = PARAMS.get('LAYOUT_GENERAL_B')
                if item['type'] == 'note':
                    w_r = 1 / (2 ** item['underline'])
                else:
                    w_r = PARAMS.get('LAYOUT_GENERAL_W')

            c = np.exp(PARAMS.get('LAYOUT_LN_C'))
            spacings.append(Spacing(
                width_l=width_l, width_r=width_r,
                k_l=k_l, v_l=v_l, k_r=k_r, v_r=v_r, w_l=w_l, w_r=w_r, c=c
            ))
            last_item = item
            last_ele = ele
            
        return spacings
    
    def search_next_barline(self, begin_seq_id):
        '''
        找到 begin_seq_id 之后第一个 barline。
        如果到结尾还没有 barline，如果 begin_seq_id 有效就返回最后一个元素，否则返回 -1。
        '''
        for i in range(begin_seq_id, len(self.sequence)):
            if self.sequence[i]['type'] == 'barline':
                return i
        return len(self.sequence)-1 if begin_seq_id <= len(self.sequence)- 1 else -1
    
    def solve_line_views_greedily(self, line_split_info, debug=False):
        '''
        根据 line_split_info 给入的参数，贪心地解出最优的乐谱行布局。
        '''
        temp = 1
        
        result = []
        current_slice_begin = 0
        cached_pos = 0 # if cached_pos > current_slice_begin, means there is cache
        cached_slice = None
        while True: # check bar-wise
            next_barline_pos = self.search_next_barline(cached_pos)
            if next_barline_pos == -1:
                break
            next_barline_slice = SpacedSequenceView(self, current_slice_begin, next_barline_pos, line_split_info['width_target'])
            x_k = next_barline_slice.solved_x
            next_barline_slice.debug_draw_line_chart(desired_y = line_split_info['width_target'],
                                                     output_name = 'line_chart_%04d(barline)' % temp) if debug else None
            temp += 1
            if (x_k < 0) or (x_k <= line_split_info['x_k_threshold']): # retreat and fetch line
                if (x_k < 0) and (cached_pos <= current_slice_begin):
                    # single group with oversized b, then check item-wise
                    next_item_pos = current_slice_begin
                    while next_item_pos <= next_barline_pos:
                        next_item_slice = SpacedSequenceView(self, current_slice_begin, next_item_pos, line_split_info['width_target'])
                        x_k_i = next_item_slice.solved_x
                        next_item_slice.debug_draw_line_chart(desired_y = line_split_info['width_target'],
                                                              output_name = 'line_chart_%04d(item)' % temp) if debug else None
                        temp += 1
                        if (x_k_i < 0) or (x_k_i <= line_split_info['x_k_threshold']): # retreat and fetch line
                            result.append(cached_slice) if cached_pos > current_slice_begin else None
                            current_slice_begin = cached_pos
                            next_item_pos += 1
                        else:
                            cached_pos = next_item_pos + 1
                            cached_slice = next_item_slice
                            next_item_pos += 1
                    cached_pos = next_barline_pos + 1
                    cached_slice = next_item_slice
                elif (cached_pos <= current_slice_begin):
                    # and (x_k >= 0) and (x_k <= x_k_threshold)
                    result.append(next_barline_slice)
                    current_slice_begin = cached_pos = next_barline_pos + 1
                else:
                    result.append(cached_slice)
                    current_slice_begin = cached_pos
            else: # keep going
                cached_pos = next_barline_pos + 1
                cached_slice = next_barline_slice

        if (cached_pos > current_slice_begin):
            result.append(cached_slice)
        
        return result

    def compile_line_views_to_ele(self, views, line_split_info, PARAMS):
        '''
        将各个行视图中的所有物件整理成渲染元素。
        '''
        
        def get_necessary_info_for_underlines_and_curves(views):
            '''
            为 underlines and curves 的处理准备必须的数据。
            请注意，整理后 note_id 的含义仍未分行，目前还是整个 mono 中的。
            '''
            notes_to_line_i = []
            lines_to_first_note_i = [-1 for _ in range(len(views))]
            lines_to_last_note_i = [-1 for _ in range(len(views))]
            for line_i, view in enumerate(views):
                for item_i, seq_item in enumerate(view.sequence_view):
                    if seq_item['type'] == 'note':
                        notes_to_line_i.append(line_i)
                        if lines_to_first_note_i[line_i] < 0:
                            lines_to_first_note_i[line_i] = seq_item['id']
                        lines_to_last_note_i[line_i] = seq_item['id']
            return notes_to_line_i, lines_to_first_note_i, lines_to_last_note_i

        def update_underline_to_lines(underlines, notes_to_line_i, lines_to_first_note_i, lines_to_last_note_i, num_lines):
            '''
            对于 underlines 的各行拆解，遇到隔行情况拆解成前中后段落即可。
            请注意，整理后 begin_id end_id 的含义仍未分行，目前还是整个 mono 中的。
            '''
            underlines_by_rows = [[] for _ in range(num_lines)]
            for underline in underlines:
                begin_id = underline['begin_id']
                end_id = underline['end_id']
                # print("begin=%d, end=%d" % (begin_id, end_id))
                begin_line_i = notes_to_line_i[begin_id]
                end_line_i = notes_to_line_i[end_id]
                if begin_line_i == end_line_i:
                    underlines_by_rows[begin_line_i].append(underline)
                else:
                    copy_begin = underline.copy()
                    copy_begin['end_id'] = lines_to_last_note_i[begin_line_i]
                    underlines_by_rows[begin_line_i].append(copy_begin)
                    for line_i in range(begin_line_i+1, end_line_i):
                        copy_inner = underline.copy()
                        copy_inner['begin_id'] = lines_to_first_note_i[line_i]
                        copy_inner['end_id'] = lines_to_last_note_i[line_i]
                        underlines_by_rows[line_i].append(copy_inner)
                    copy_end = underline.copy()
                    copy_end['begin_id'] = lines_to_first_note_i[end_line_i]
                    underlines_by_rows[end_line_i].append(copy_end)
            return underlines_by_rows

        def update_curve_to_lines(curves, notes_to_line_i, lines_to_first_note_i, lines_to_last_note_i, num_lines):
            '''
            对于 curves 的各行拆解，遇到隔行时需要标记 open_type。
            请注意，整理后 begin_id end_id 的含义仍未分行，目前还是整个 mono 中的。
            '''
            curves_by_rows = [[] for _ in range(num_lines)]
            for curve in curves:
                begin_id = curve['begin_id']
                end_id = curve['end_id']
                begin_line_i = notes_to_line_i[begin_id]
                end_line_i = notes_to_line_i[end_id]
                if begin_line_i == end_line_i:
                    curves_by_rows[begin_line_i].append(curve)
                else:
                    copy_begin = curve.copy()
                    copy_begin['end_id'] = lines_to_last_note_i[begin_line_i]
                    copy_begin['open_type'] = 'right_open'
                    curves_by_rows[begin_line_i].append(copy_begin)
                    copy_end = curve.copy()
                    copy_end['begin_id'] = lines_to_first_note_i[end_line_i]
                    copy_end['open_type'] = 'left_open'
                    curves_by_rows[end_line_i].append(copy_end)
            return curves_by_rows

        # 预先处理 underlines and curves
        notes_to_line_i, lines_to_first_note_i, lines_to_last_note_i = get_necessary_info_for_underlines_and_curves(views)
        updated_underlines = update_underline_to_lines(self.mono_data['underlines'], notes_to_line_i, lines_to_first_note_i, lines_to_last_note_i, len(views))
        updated_curves = update_curve_to_lines(self.mono_data['curves'], notes_to_line_i, lines_to_first_note_i, lines_to_last_note_i, len(views))
        # print(updated_curves)
        # print(updated_curves)
        
        # 生成 lines_ready（放在这里是因为要先计算各个音符的 position，才能具体生成 underlines and curves）
        lines_ready = [view.get_line_ready(PARAMS) for view in views]
        
        # 具体生成 underlines and curves
        underlines_eles = []
        curves_eles = []
        for line_i in range(len(views)):
            first = lines_to_first_note_i[line_i]
            last = lines_to_last_note_i[line_i]
            # print("first=%d, last=%d" % (first, last))
            if first >= 0 and last >= 0 and first <= last:
                underline_rows = interval_layering_underline(updated_underlines[line_i], last - first + 1, line_offset=first)
                curve_rows = interval_layering_curve(updated_curves[line_i], last - first + 1, line_offset=first)
                underlines_eles.append(create_underlines(underline_rows, self.sequence_data['note'][first:last+1], PARAMS, appog=False))
                curves_eles.append(create_curves(curve_rows, self.sequence_data['note'][first:last+1], PARAMS, appog=False,
                                                 left_x=-PARAMS.get('CURVE_CROSSPAGE_LEFT_XOFFSET'),
                                                 right_x=line_split_info['width_target']+PARAMS.get('CURVE_CROSSPAGE_RIGHT_XOFFSET')))

        # 打包同时求解 y
        off_y = 0
        result = ElementContainer()
        for view, line_ready, underlines_ele, curves_ele in zip(views, lines_ready, underlines_eles, curves_eles):
            ele = EMonophonicSingleLine(line_ready, underlines_ele, curves_ele, PARAMS)
            if ele.physical_box.is_valid():
                off_y += -ele.physical_box.up
            else:
                off_y += PARAMS.get('LAYOUT_LINE_EMPTY_ROW_HEIGHT') / 2
            ele.position += Position(0, off_y)
            result.add(ele)
            if ele.physical_box.is_valid():
                off_y += ele.physical_box.down + PARAMS.get('LAYOUT_LINE_SPACE')
            else:
                off_y += PARAMS.get('LAYOUT_LINE_EMPTY_ROW_HEIGHT') / 2 + PARAMS.get('LAYOUT_LINE_SPACE')
                
        return result


# ### [B5d] `EMonophonicSingleLine`, `EMonophonic` 渲染元素

# In[60]:


class EMonophonicSingleLine(Element):
    def __init__(self, data_ready, underlines_ele, curves_ele, PARAMS, position=None):
        super().__init__(position=position)
        
        self.data_ready = data_ready
        self.data_ready['underline'] = underlines_ele
        self.data_ready['curve'] = curves_ele
        
        self.physical_box = Box.union_boxes([ele.physical_box + ele.position for ele in data_ready.values()])
        self.render_box = Box.union_boxes([ele.render_box + ele.position for ele in data_ready.values()])
        assert self.check_valid()
        
    def draw_inner(self, draw, offset, debug=False):
        for ele in self.data_ready.values():
            if ele.physical_box.is_valid():
                ele.draw(draw, offset, debug=debug)
                
    def __str__(self):
        result = super().__str__() + "\n"
        for i, key in enumerate(list(self.data_ready.keys())):
            result += "**" + str(key) + "**: " + str(self.data_ready[key])
            if i < len(self.data_ready.keys()) - 1:
                result += '\n'
        return result
    def __repr__(self):
        return str(self)

class EMonophonic(Element):
    
    def __init__(self, mono_data, PARAMS, appog, position=None, line_split_info=None):
        super().__init__(position=position)
        # PARAMS.update_item('monophonic')
        mono_data = update_hint_for_notes(update_underline_for_notes(mono_data))
        self.data = mono_data
        
        self.layout = MonophonicLayoutSequence(mono_data, PARAMS)
        self.views = self.layout.solve_line_views_greedily(line_split_info, debug=False)
        
        self.lines = self.layout.compile_line_views_to_ele(self.views, line_split_info, PARAMS)
        
        self.physical_box = self.lines.physical_box + self.lines.position
        self.render_box = self.lines.render_box + self.lines.position
        assert self.check_valid()
        
    def draw_inner(self, draw, offset, debug=False):
        self.lines.draw(draw, offset, debug=debug)


# In[61]:


# # with open('temp_gen_sheet.pickle', 'rb') as f:
# #     temp_gen_sheet = pickle.load(f)

# temp = EMonophonic(example_monophonic_seq_0_haha, PARAMS_, appog=False, line_split_info={
#     'width_target': 2400,
#     'w_to_k_ratio': 0.25,
#     'x_k_threshold': 40,
# })


# In[62]:


# temp.render_to_path("tmp.png", debug=True, padding=100)


# In[ ]:




