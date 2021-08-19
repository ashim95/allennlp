import logging
import math
import os
import time
import datetime
import traceback
import torch
import torch.optim.lr_scheduler

from copy import deepcopy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
device = 'cuda:0'
model_id = 'gpt2'
gpt_tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
gpt_model = GPT2LMHeadModel.from_pretrained(model_id).to(device)


class SimpleCurriculum:

    def __init__(self, cl_metric,  train_data, augment_data, reverse=False, metric_reverse=False, num_stages=10):

        """
        reverse (default=False): if True, start from augment data and move to train data
        Sort each of train_data and augment_data w.r.t. the cl_metric
        First apply curriculum training on train_data, then augment data
        """
        if reverse:
            self.train_data = deepcopy(augment_data)
            self.augment_data = deepcopy(train_data)
        else:
            self.train_data = deepcopy(train_data)
            self.augment_data = deepcopy(augment_data)
        self.cl_type = cl_metric
        self.stage_num = -1
        self.num_stages = num_stages
        self.reverse = reverse
        self.metric_reverse = metric_reverse

        logger.info('Curriculum Type is %s', self.cl_type)

        logger.info('Sorting the datasets ')
        self.sort_dataset(self.cl_type, metric_reverse=self.metric_reverse)

        self.itr_on_first = True # is current iterator on first set or second set

        self.init_curriculum()

    def sort_dataset(self, cl_metric, metric_reverse=False):

        sort_function = function_mapping[cl_metric]
        # First sort train_data, then augment_data

        self.train_data = sort_function(self.train_data, reverse=metric_reverse)
        self.augment_data = sort_function(self.augment_data, reverse=metric_reverse)

    def init_curriculum(self):
        # First sort the examples by the ease of their difficulty/easiness based on cl_type
        self.size_to_add_each_stage = len(self.train_data) // self.num_stages
        self.complete = False

    def next_stage(self):
        logger.info('Stage %s for current Curriculum', str(self.stage_num))
        self.stage_num +=1

        if not self.itr_on_first and self.complete:
            # return all examples
            logger.info('All stages of curriculum done, now training till convergence')
            returned= self.concatenate_data(self.train_data, self.augment_data)
            logger.info('Size of data returned at stage %s of full data set is %s', str(self.stage_num), str(len(returned)))
            return returned

        if self.itr_on_first:
            self.current_data = self.train_data
        else:
            self.current_data = self.augment_data



        if self.stage_num == self.num_stages:
            if self.itr_on_first:
                logger.info('Completed initial stage of curriculum, now moving to next stage with second set of data (augmented data)')
                self.stage_num = 0
                self.itr_on_first = False
                self.complete = False
            else:
                logger.info('Completed both stages of curriculum, i.e. with both sets of data.')
                self.complete = True

        end = min((self.stage_num+1)*self.size_to_add_each_stage, len(self.current_data))

        sampled = self.current_data[:end]
        if self.itr_on_first:
            logger.info('Size of data returned at stage %s of first data set is %s', str(self.stage_num), str(len(sampled)))
            return deepcopy(sampled)
        else:
            returned =  self.concatenate_data(self.train_data, sampled)
            logger.info('Size of data returned at stage %s of second data set is %s', str(self.stage_num), str(len(returned)))
            return returned

    def concatenate_data(self, data1, data2):
        data = deepcopy(data1)
        data2 = deepcopy(data2)
        data.extend(data2)
        return data

    def is_complete(self):
        return self.complete



def len_between_entities(subj_0, subj_1, obj_0, obj_1):

    if subj_0 < obj_0:
        return obj_0 - subj_1
    else:
        return subj_0 - obj_1


def group_by_length_of_sents(examples, reverse=False, return_met=False):

    len_tuples = []

    for ex in examples:
        l = len(ex.fields['text'].tokens)
        len_tuples.append((ex, l))

    sorted_list, met = zip(*sorted(len_tuples, key=lambda x: x[1], reverse=reverse))
    if return_met:
        return list(sorted_list), met
    return list(sorted_list)


def group_by_distance_between_entities(examples, reverse=False, return_met=False):

    len_tuples = []

    for ex in examples:
        l = len_between_entities(ex.fields['head'].span_start, ex.fields['head'].span_end, ex.fields['tail'].span_start, ex.fields['tail'].span_end)
        len_tuples.append((ex, l))

    sorted_list, met = zip(*sorted(len_tuples, key=lambda x: x[1], reverse=reverse))
    if return_met:
        return list(sorted_list), met
    return list(sorted_list)

def get_gpt2_perplexity(sentence):

    gpt_model.eval()
    input_ids = torch.tensor(gpt_tokenizer.encode(sentence)).unsqueeze(0)

    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = gpt_model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)


def group_by_gpt2_ppl(examples, reverse=False, return_met=False):

    ppl_tuples = []
    for ex in examples:
        sent = ' '.join([tok.text for tok in ex.fields['text'].tokens])
        ppl_tuples.append((ex, get_gpt2_perplexity(sent)))

    sorted_list, met = zip(*sorted(ppl_tuples, key=lambda x: x[1], reverse=reverse))
    if return_met:
        return list(sorted_list), met
    return list(sorted_list)



function_mapping = {
        'sort_by_length': group_by_length_of_sents,
        'sort_by_dist_entities': group_by_distance_between_entities,
        'sort_by_ppl': group_by_gpt2_ppl,
        }

