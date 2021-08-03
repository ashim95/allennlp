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


class Curriculum:

    def __init__(self, train_data=None, augment_data=None, cl_type=None, starting_percent=None, num_stages=10, init_stage_till_convergence=False):

        self.train_data = deepcopy(train_data)
        self.augment_data = deepcopy(augment_data)
        self.cl_type = cl_type
        self.starting_percent = starting_percent
        self.stage_num = -1
        self.num_stages = num_stages

        logger.info('Curriculum Type is %s', self.cl_type)

        self.init_curriculum()

        self.complete = False

        self.func_dict = {
                        'none': self.none_curriculum,
                        'naive': self.naive_curriculum,
                        'opposite_naive': self.opposite_naive_curriculum,
                        'sort_by_length': self.sort_curriculum,
                        'sort_by_dist_entities': self.sort_curriculum,
                        }

        self.sort_func_dict = {
                            'sort_by_length': self.group_by_length_of_sents,
                            'sort_by_dist_entities': self.group_by_distance_between_entities,

                            }

        self.init_stage_till_convergence = False  # Train the model of initial stage till convergence then start adding examples

        self.init_model_converged = False

        self.function = self.func_dict[cl_type.lower()]

    def init_curriculum(self):
        # First sort the examples by the ease of their difficulty/easiness based on cl_type
        self.size_to_add_each_stage = len(self.train_data) // self.num_stages

    def none_curriculum(self):

        self.complete = True

    def sort_curriculum(self):

        sort_function = self.sort_func_dict[self.cl_type]

        self.stage_num +=1
        self.all_examples = []
        self.all_examples.extend(self.augment_data)
        self.all_examples.extend(self.train_data)

        self.all_examples = sort_function(self.all_examples)

        if self.stage_num > self.num_stages:
            self.complete = True
            logger.info('For stage > %s, using all augmented data + all train data', str(self.num_stages))
            return self.all_examples

        self.size_to_add_each_stage = len(self.all_examples) // self.num_stages

        end = min((self.stage_num+1)*self.size_to_add_each_stage, len(self.all_examples))

        sampled = self.all_examples[:end]

        if self.stage_num == self.num_stages:
            self.complete = True

        return deepcopy(sampled)

    def len_between_entities(self, subj_0, subj_1, obj_0, obj_1):

        if subj_0 < obj_0:
            return obj_0 - subj_1
        else:
            return subj_0 - obj_1


    def group_by_length_of_sents(self, examples):

        len_tuples = []

        for ex in examples:
            l = len(ex.fields['text'].tokens)
            len_tuples.append((ex, l))

        sorted_list, _ = zip(*sorted(len_tuples, key=lambda x: x[1], reverse=True))
        return sorted_list


    def group_by_distance_between_entities(self, examples):

        len_tuples = []

        for ex in examples:
            l = self.len_between_entities(ex.fields['head'].span_start, ex.fields['head'].span_end, ex.fields['tail'].span_start, ex.fields['tail'].span_end)
            len_tuples.append((ex, l))

        sorted_list, _ = zip(*sorted(len_tuples, key=lambda x: x[1], reverse=True))
        return sorted_list

    def naive_curriculum(self):

        # if we have to train initial stage model till convergence and that model has not converged
        if self.init_stage_till_convergence and not self.init_model_converged:
            self.stage_num = 0
        else:
            self.stage_num +=1
        logger.info('Stage %s for naive Curriculum', str(self.stage_num))

        if self.init_stage_till_convergence and not self.init_model_converged:
            logger.info('For stage %s, using all initial data, training initial model till convergence'. str(self.stage_num))
            return deepcopy(self.augment_data)

        if self.stage_num == 0:
            logger.info('For stage 0, using all augmented data')
            return deepcopy(self.augment_data)

        if self.stage_num > self.num_stages:
            self.complete = True
            logger.info('For stage > %s, using all augmented data + all train data', str(self.num_stages))
            return self.concatenate_data(self.augment_data, self.train_data)


        end = min(self.stage_num*self.size_to_add_each_stage, len(self.train_data))

        sampled = self.train_data[:end]

        if self.stage_num == self.num_stages:
            self.complete = True

        return self.concatenate_data(self.augment_data, sampled)

    def opposite_naive_curriculum(self):

        # if we have to train initial stage model till convergence and that model has not converged
        if self.init_stage_till_convergence and not self.init_model_converged:
            self.stage_num = 0
        else:
            self.stage_num +=1
        logger.info('Stage %s for naive Curriculum', str(self.stage_num))

        if self.init_stage_till_convergence and not self.init_model_converged:
            logger.info('For stage %s, using all initial data, training initial model till convergence'. str(self.stage_num))
            return deepcopy(self.train_data)

        if self.stage_num == 0:
            logger.info('For stage 0, using all train data')
            return deepcopy(self.train_data)

        if self.stage_num > self.num_stages:
            self.complete = True
            logger.info('For stage > %s, using all augmented data + all train data', str(self.num_stages))
            return self.concatenate_data(self.train_data, self.augment_data)

        self.size_to_add_each_stage = len(self.augment_data) // self.num_stages
        end = min(self.stage_num*self.size_to_add_each_stage, len(self.augment_data))

        sampled = self.augment_data[:end]

        if self.stage_num == self.num_stages:
            self.complete = True

        return self.concatenate_data(self.train_data, sampled)

    def is_complete(self):
        return self.complete

    def concatenate_data(self, data1, data2):
        data = deepcopy(data1)
        data2 = deepcopy(data2)
        data.extend(data2)
        return data

    def get_next_stage_data(self):

        self.stage_num +=1




