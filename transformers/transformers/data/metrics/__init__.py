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

import csv
import sys
import logging
from itertools import combinations, permutations

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False

def is_sklearn_available():
    return _has_sklearn

if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


    # def sentence_cloze_acc_maskdis(preds, datas):
    #     global_idx = 0
    #     total = 0.
    #     correct = 0
    #     all_correct = 0.
    #     all_total = 0.
    #     for data in datas:
    #         eid = data['eid']
    #         passage = data['passage']
    #         candidates = data['candidates']
    #         answers = data['answer_sequence']
    #         num_blanks = data['number_of_blanks']
    #         num_candidates = data['candidate_length']
    #         blank_indexes = [passage.index('<' + str(i + 1) + '>') for i in range(num_blanks)]
    #         cur_ans = []
    #         golden_ans = [ans[1] for ans in answers]
    #         dis = [ans for ans in list(range(num_candidates)) if ans not in golden_ans]
    #         for _ in range(num_blanks):
    #             cur_pred = preds[global_idx: global_idx + num_candidates]
    #             for d in dis:
    #                 cur_pred[d] = 0.
    #             sort_pred = sorted(range(len(cur_pred)), key=lambda k: cur_pred[k], reverse=True)
    #             for i in range(num_candidates):
    #                 if sort_pred[i] not in cur_ans:
    #                     cur_ans.append(sort_pred[i])
    #                     break
    #             global_idx += num_candidates
    #         for p, t in zip(cur_ans, golden_ans):
    #             if p == t:
    #                 correct += 1
    #             total += 1
    #         if cur_ans == golden_ans:
    #             all_correct += 1
    #         all_total += 1
    #     return correct / total, all_correct / all_total

    def single_exh(preds, datas, has_mask=True):
        global_idx = 0
        total = 0.
        correct = 0
        all_correct = 0.
        all_total = 0.
        dis_num = 0.
        for data in datas:
            eid = data['eid']
            passage = data['passage']
            candidates = data['candidates']
            answers = data['answer_sequence']
            num_blanks = data['number_of_blanks']
            num_candidates = data['candidate_length']
            blank_indexes = [passage.index('<' + str(i + 1) + '>') for i in range(num_blanks)]
            cur_ans = []
            golden_ans = [ans[1] for ans in answers]
            dis = [ans for ans in list(range(num_candidates)) if ans not in golden_ans]
            for _ in range(num_blanks):
                cur_pred = preds[global_idx: global_idx + num_candidates]
                if not has_mask:
                    for d in dis:
                        cur_pred[d] = 0.
                sort_pred = sorted(range(len(cur_pred)), key=lambda k: cur_pred[k], reverse=True)
                for i in range(num_candidates):
                    if sort_pred[i] not in cur_ans:
                        cur_ans.append(sort_pred[i])
                        break
                global_idx += num_candidates
            
            for p, t in zip(cur_ans, golden_ans):
                if p not in golden_ans:
                    dis_num += 1
                if p == t:
                    correct += 1
                total += 1
            if cur_ans == golden_ans:
                all_correct += 1
            all_total += 1
        return correct / total, all_correct / all_total, dis_num / all_total

    def bi_inc(preds, datas, has_mask=True):
        global_idx = 0
        total = 0.
        correct = 0
        all_correct = 0.
        all_total = 0.
        dis_num = 0.
        for data in datas:
            eid = data['eid']
            passage = data['passage']
            candidates = data['candidates']
            answers = data['answer_sequence']
            num_blanks = data['number_of_blanks']
            num_candidates = data['candidate_length']
            blank_indexes = [passage.index('<' + str(i + 1) + '>') for i in range(num_blanks)]
            cur_ans = []
            golden_ans = [ans[1] for ans in answers]
            dis = [ans for ans in list(range(num_candidates)) if ans not in golden_ans]
            for _ in range(num_blanks):
                cur_pred = preds[global_idx: global_idx + num_candidates * 2]
                cur_pred = [sum(cur_pred[i:i+2]) for i in range(0, len(cur_pred), 2)]
                if not has_mask:
                    for d in dis:
                        cur_pred[d] = 0.
                sort_pred = sorted(range(len(cur_pred)), key=lambda k: cur_pred[k], reverse=True)
                for i in range(num_candidates):
                    if sort_pred[i] not in cur_ans:
                        cur_ans.append(sort_pred[i])
                        break
                global_idx += num_candidates * 2
            
            for p, t in zip(cur_ans, golden_ans):
                if p not in golden_ans:
                    dis_num += 1
                if p == t:
                    correct += 1
                total += 1
            if cur_ans == golden_ans:
                all_correct += 1
            all_total += 1
        return correct / total, all_correct / all_total, dis_num / all_total

    def single_exh(preds, datas, has_dis=True):
        global_idx = 0
        total = 0.
        correct = 0
        exact_correct = 0.
        all_total = 0.
        dis_num = 0.
        for data in datas:
            eid = data['eid']
            passage = data['passage']
            candidates = data['candidates']
            answers = data['answer_sequence']
            num_blanks = data['number_of_blanks']
            num_candidates = data['candidate_length']
            blank_indexes = [passage.index('<' + str(i + 1) + '>') for i in range(num_blanks)]
            candi_range = list(range(num_candidates))
            max_score = 0
            rst = []
            golden_ans = [ans[1] for ans in answers]
            dis = [ans for ans in list(range(num_candidates)) if ans not in golden_ans]
            for permutation in permutations(candi_range, len(answers)):
                if not has_dis:
                    flag = False
                    for d in dis:
                        if d in permutation:
                            flag = True
                            break
                    if flag:
                        continue
                cur_score = 0.
                for idx, ans in enumerate(permutation):
                    cur_score += preds[global_idx + num_candidates * idx + ans]
                if cur_score > max_score:
                    max_score = cur_score
                    rst = permutation
            global_idx += num_candidates * len(answers)
            golden_ans = [ans[1] for ans in answers]
            single_total = 0
            single_correct = 0.
            if list(rst) == golden_ans:
                exact_correct += 1
            for p, t in zip(rst, golden_ans):
                if p not in golden_ans:
                    dis_num += 1
                if p == t:
                    correct += 1
                total += 1
            all_total += 1
        return correct / total, exact_correct / all_total, dis_num / all_total 


    def bi_exh(preds, datas, has_dis=True):
        level_array = []
        machine_array = []
        convert_dict = {}
        global_idx = 0
        total = 0.
        all_total = 0.
        correct = 0
        dis_num = 0.
        exact_correct = 0.
        bad = 0.
        eid_list = []
        for data in datas:
            eid = data['eid']
            passage = data['passage']
            candidates = data['candidates']
            answers = data['answer_sequence']
            num_blanks = data['number_of_blanks']
            num_candidates = data['candidate_length']
            blank_indexes = [passage.index('<' + str(i + 1) + '>') for i in range(num_blanks)]
            candi_range = list(range(num_candidates))
            max_score = 0
            rst = []
            golden_ans = [ans[1] for ans in answers]
            dis = [ans for ans in list(range(num_candidates)) if ans not in golden_ans]
            for permutation in permutations(candi_range, len(answers)):
                cur_score = 0.
                if not has_dis:
                    flag = False
                    for d in dis:
                        if d in permutation:
                            flag = True
                            break
                    if flag:
                        continue
                for idx, ans in enumerate(permutation):
                    cur_score += preds[global_idx + 2 * num_candidates * idx + ans * 2] + preds[global_idx + 2 * num_candidates * idx + ans * 2+1]
                if cur_score > max_score:
                    max_score = cur_score
                    rst = permutation
            global_idx += num_candidates * len(answers) * 2
            eid_list.append(eid)
            if list(rst) == golden_ans:
                exact_correct += 1
            assert len(rst) > 0 
            for p, t in zip(rst, golden_ans):
                if p not in golden_ans:
                    dis_num += 1
                total += 1
                if p == t:
                    correct += 1
            all_total += 1
        return correct / total, exact_correct / all_total, dis_num / all_total

    # def bi_sentence_cloze_acc_maskdis(preds, datas):
    #     global_idx = 0
    #     total = 0.
    #     correct = 0
    #     all_correct = 0.
    #     all_total = 0.
    #     data_accuracy = {}
    #     for data in datas:
    #         eid = data['eid']
    #         passage = data['passage']
    #         candidates = data['candidates']
    #         answers = data['answer_sequence']
    #         num_blanks = data['number_of_blanks']
    #         num_candidates = data['candidate_length']
    #         blank_indexes = [passage.index('<' + str(i + 1) + '>') for i in range(num_blanks)]
    #         cur_ans = []
    #         golden_ans = [ans[1] for ans in answers]
    #         dis = [ans for ans in list(range(num_candidates)) if ans not in golden_ans]
    #         for _ in range(num_blanks):
    #             cur_pred = preds[global_idx: global_idx + num_candidates * 2]
    #             cur_pred = [sum(cur_pred[i:i+2]) for i in range(0, len(cur_pred), 2)]
    #             for d in dis:
    #                 cur_pred[d] = 0.
    #             sort_pred = sorted(range(len(cur_pred)), key=lambda k: cur_pred[k], reverse=True)
    #             for i in range(num_candidates):
    #                 if sort_pred[i] not in cur_ans:
    #                     cur_ans.append(sort_pred[i])
    #                     break
    #             global_idx += num_candidates * 2
    #         single_correct = 0
    #         single_total = 0
    #         for p, t in zip(cur_ans, golden_ans):
    #             if p == t:
    #                 correct += 1
    #                 single_correct += 1
    #             single_total += 1
    #             total += 1
    #         if cur_ans == golden_ans:
    #             all_correct += 1
    #         all_total += 1
    #         data_accuracy[eid] = single_correct / float(single_total)
    #     return correct / total, all_correct / all_total


    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }


    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }


    def glue_compute_metrics(task_name, preds, labels):
        if task_name != 'senclz':
            assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "senclz":
            inc_ba, inc_pa, inc_de = bi_inc(preds, labels)
            exh_ba, exh_pa, exh_de = bi_exh(preds, labels)
            return {"inc_ba": inc_ba,
                    "inc_pa": inc_pa,
                    "inc_de": inc_de,
                    "exh_ba": exh_ba,
                    "exh_pa": exh_pa,
                    "exh_de": exh_de
                    }
        elif task_name == "senta-dis":
            return {"acc": acc_and_f1(preds, labels)}
        else:
            raise KeyError(task_name)
