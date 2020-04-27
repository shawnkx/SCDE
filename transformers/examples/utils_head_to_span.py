# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
import json

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, head_pos, span_start, span_end):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.head_pos = head_pos
        self.span_start = span_start
        self.span_end = span_end


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, left_span_ids,
                right_span_ids, span_start, span_end, left_span_mask, right_span_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.left_span_ids = left_span_ids
        self.right_span_ids = right_span_ids
        self.span_start = span_start
        self.span_end = span_end
        self.left_span_mask = left_span_mask
        self.right_span_mask = right_span_mask


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    guid_index = 1
    examples = []
    with open(file_path) as f:
        for line in f:
            line = json.loads(line)
            words = line['tokens']
            head_pos = line['head']
            span_start = head_pos - line['start'] 
            span_end = line['end'] - head_pos
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                        words=words,
                                        head_pos=head_pos,
                                        span_start=span_start,
                                        span_end=span_end))
            guid_index += 1
    return examples


def convert_examples_to_features(examples,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-1,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        left_span = []
        right_span = []
        left_span_mask = []
        right_span_mask = []
        head_pos = example.head_pos
        if cls_token_at_end:
            tokens += [cls_token]
            # left_span += [pad_token_label_id]
            # right_span += [pad_token_label_id]
            #segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            # left_span = [pad_token_label_id] + left_span
            # right_span = [pad_token_label_id] + right_span
            #segment_ids = [cls_token_segment_id] + segment_ids
        # since cls is zero
        idx = 1
        for widx, word in enumerate(example.words):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            if -1 < head_pos - widx < 5:
                left_span = [idx] + left_span
            # else:
            #     left_span.append([0] * len(word_tokens))
            if -1 < widx - head_pos < 5:
                right_span.append(idx)
            # else:
            #     right_span.append([0] * len(word_tokens))
            idx += len(word_tokens)
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        
        special_tokens_count = 3 if sep_token_extra else 2
        wp_head_pos = left_span[0]
        if len(tokens) > max_seq_length - special_tokens_count:
            truncated_tokens = []
            left = wp_head_pos
            right = wp_head_pos + 1
            while len(truncated_tokens) < max_seq_length - special_tokens_count:
                if left >= 0:
                    #print(left, len(tokens))
                    truncated_tokens = [tokens[left]] + truncated_tokens
                    left -= 1
                if right < len(tokens):
                    truncated_tokens.append(tokens[right])
            # left = (max_seq_length - special_tokens_count) // 2
            # right = max_seq_length - special_tokens_count - left - 1
            # tokens = tokens[wp_head_pos-left:wp_head_pos] + [tokens[wp_head_pos]] + tokens[wp_head_pos+1:wp_head_pos+right+1]
            tokens = truncated_tokens
            left_span = [ls - left - 1  for ls in left_span]
            right_span = [rs - left - 1 for rs in right_span]
        left_span_mask = [1] * len(left_span) + [float('-inf')] * (5 - len(left_span))
        right_span_mask = [1] * len(right_span) + [float('-inf')] * (5 - len(right_span))
        left_span = left_span + [0] * (5 - len(left_span))
        right_span = right_span + [0] * (5 - len(right_span))
        
            #tokens = tokens[:(max_seq_length - special_tokens_count)]
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        # left_span += [0]
        # right_span += [0]
        if sep_token_extra:
            tokens += [sep_token]
        #     # roberta uses an extra separator b/w pairs of sentences
        #     left_span += [0]
        #     right_span += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            # left_span = ([pad_token_label_id] * padding_length) + left_span
            # right_span = ([pad_token_label_id] * padding_length) + right_span
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            # left_span += ([pad_token_label_id] * padding_length)
            # right_span += ([pad_token_label_id] * padding_length)
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # assert len(left_span) == max_seq_length
        # assert len(right_span) == max_seq_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("left_span: %s", " ".join([str(x) for x in left_span]))
            logger.info("right_span: %s", " ".join([str(x) for x in right_span]))
            logger.info("head_pos: %s", str(example.head_pos))
            logger.info("span_start: %s", str(example.span_start))
            logger.info("span_end: %s", str(example.span_end))
            logger.info('left_span_mask: %s', " ".join([str(m) for m in left_span_mask]))
            logger.info('right_span_mask: %s', " ".join([str(m) for m in right_span_mask]))


        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              left_span_ids=left_span,
                              right_span_ids=right_span,
                              span_start=example.span_start,
                              span_end=example.span_end,
                              left_span_mask=left_span_mask,
                              right_span_mask=right_span_mask))
    return features
