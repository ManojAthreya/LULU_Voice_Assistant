import os
import torch
from tqdm import tqdm
import numpy as np
from collections import Counter
from torch.autograd import Variable

class Evaluator(object):
    @staticmethod
    def max_freq_predict(sample):
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
        return predict

    @staticmethod
    def exp_decay_predict(sample, decay_rate=0.8):
        predict = []
        for items in sample:
            item_dict = {}
            curr_weight = 1.0
            for item in items[::-1]:
                item_dict[item] = item_dict.get(item, 0) + curr_weight
                curr_weight *= decay_rate
            predict.append(sorted(item_dict.items(), key=lambda x_: x_[1])[-1][0])
        return predict

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items

def validate(model_path, dataset_path):
        """
        validation will write mistaken samples to files and make scores.
        """
        # Load the model and dataset components
        model = torch.load(model_path)
        dataset = torch.load(dataset_path)

        pred_slot, exp_pred_intent, pred_intent = prediction(
            model, dataset, "test"
        )

        return pred_slot, pred_intent, exp_pred_intent

def prediction(input_text, model, dataset, mode):
    # Put the model in eval mode
    model.eval()

    pred_slot =  []
    pred_intent = []

    # A sample text
    sample_str = input_text.split()
    text_batch = [sample_str]
    
    # Padd the dataset to appropiate length
    padded_text, seq_lens, sorted_index = dataset.add_padding(
        text_batch, None, digital=False
    )

    digit_text = dataset.word_alphabet.get_index(padded_text)
    var_text = Variable(torch.LongTensor(digit_text))

    if torch.cuda.is_available():
        var_text = var_text.cuda()

    slot_idx, intent_idx = model(var_text, seq_lens, n_predicts=1)
    nested_slot = Evaluator.nested_list([list(Evaluator.expand_list(slot_idx))], seq_lens)[0]
    
    if mode == 'test':
        tmp_r_slot = [[] for _ in range(len(sorted_index))]
        for i in range(len(sorted_index)):
            tmp_r_slot[sorted_index[i]] = nested_slot[i]
        nested_slot = tmp_r_slot
    
    pred_slot.extend(dataset.slot_alphabet.get_instance(nested_slot))
    nested_intent = Evaluator.nested_list([list(Evaluator.expand_list(intent_idx))], seq_lens)[0]
    
    if mode == 'test':
        tmp_intent = [[] for _ in range(len(sorted_index))]
        for i in range(len(sorted_index)):
            tmp_intent[sorted_index[i]] = nested_intent[i]
        nested_intent = tmp_intent
    
    pred_intent.extend(dataset.intent_alphabet.get_instance(nested_intent))

    return pred_slot, pred_intent


if __name__ == "__main__":
    # Get the predictions
    pred_slot, pred_intent = validate(
        model_path="/mnt/c/Ubuntu/virtual_assistant/StackPropagation-SLU/save/model/model.pkl",
        dataset_path="/mnt/c/Ubuntu/virtual_assistant/StackPropagation-SLU/save/model/dataset.pkl")
    
    print(f"Pred slot : {pred_slot}")
    print(f"Pred intent : {pred_intent}")
