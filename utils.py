import logging
import time
import pickle
import json
import prettytable as pt
from collections import defaultdict, deque


def get_logger(dataset):
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def decode(outputs, entities, length):
    ent_r, ent_p, ent_c = 0, 0, 0
    for index, (instance, ent_set, l) in enumerate(zip(outputs, entities, length)):
        forward_dict = {}
        head_dict = {}
        ht_type_dict = {}
        for i in range(l):
            for j in range(i + 1, l):
                if instance[i, j] == 1:
                    if i not in forward_dict:
                        forward_dict[i] = [j]
                    else:
                        forward_dict[i].append(j)
        for i in range(l):
            for j in range(i, l):
                if instance[j, i] > 1:
                    ht_type_dict[(i, j)] = instance[j, i]
                    if i not in head_dict:
                        head_dict[i] = {j}
                    else:
                        head_dict[i].add(j)

        predicts = []

        def find_entity(key, entity, tails):
            entity.append(key)
            if key not in forward_dict:
                if key in tails:
                    predicts.append(entity.copy())
                entity.pop()
                return
            else:
                if key in tails:
                    predicts.append(entity.copy())
            for k in forward_dict[key]:
                find_entity(k, entity, tails)
            entity.pop()

        for head in head_dict:
            find_entity(head, [], head_dict[head])

        predicts = set([convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts])

        ent_r += len(ent_set)
        ent_p += len(predicts)
        for x in predicts:
            if x in ent_set:
                ent_c += 1
    return ent_c, ent_p, ent_r


def decode_2(outputs, entities, length):
    ent_r, ent_p, ent_c = 0, 0, 0
    for index, (instance, ent_set, l) in enumerate(zip(outputs, entities, length)):
        predicts = []
        for i in range(l):
            for j in range(i, l):
                if instance[i, j] != 0:
                    temp = []
                    for k in range(i, j+1):
                        temp.append(k)
                    predicts.append(temp)

        forward_dict = {}
        head_dict = {}
        ht_type_dict = {}
        # for i in range(l):
        #     for j in range(i + 1, l):
        #         if instance[i, j] == 1:
        #             if i not in forward_dict:
        #                 forward_dict[i] = [j]
        #             else:
        #                 forward_dict[i].append(j)
        for i in range(l):   #修改
            for j in range(i, l):
                if instance[i, j] > 0:  ##修改  >1 >0
                    ht_type_dict[(i, j)] = instance[i, j]
                    if i not in head_dict:
                        head_dict[i] = {j}
                    else:
                        head_dict[i].add(j)

        # predicts = []
        #
        # def find_entity(key, entity, tails):
        #     entity.append(key)
        #     if key not in forward_dict:
        #         if key in tails:
        #             predicts.append(entity.copy())
        #         entity.pop()
        #         return
        #     else:
        #         if key in tails:
        #             predicts.append(entity.copy())
        #     for k in forward_dict[key]:
        #         find_entity(k, entity, tails)
        #     entity.pop()
        #
        # for head in head_dict:
        #     find_entity(head, [], head_dict[head])

        predicts = set([convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts])

        ent_r += len(ent_set)
        ent_p += len(predicts)
        for x in predicts:
            if x in ent_set:
                ent_c += 1
    return ent_c, ent_p, ent_r


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)


def api_decode(outputs, entities, length):
    class Node:
        def __init__(self):
            self.THW = []  # [(tail, type)]
            self.NNW = defaultdict(set)  # {(head,tail): {next_index}}

    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = []
    q = deque()
    for instance, ent_set, l in zip(outputs, entities, length):
        predicts = []
        nodes = [Node() for _ in range(l)]
        for cur in reversed(range(l)):
            heads = []
            for pre in range(cur + 1):
                # THW
                if instance[cur, pre] > 1:
                    nodes[pre].THW.append((cur, instance[cur, pre]))
                    heads.append(pre)
                # NNW
                if pre < cur and instance[pre, cur] == 1:
                    # cur node
                    for head in heads:
                        nodes[pre].NNW[(head, cur)].add(cur)
                    # post nodes
                    for head, tail in nodes[cur].NNW.keys():
                        if tail >= cur and head <= pre:
                            nodes[pre].NNW[(head, tail)].add(cur)
            # entity
            for tail, type_id in nodes[cur].THW:
                if cur == tail:
                    predicts.append(([cur], type_id))
                    continue
                q.clear()
                q.append([cur])
                while len(q) > 0:
                    chains = q.pop()
                    for idx in nodes[chains[-1]].NNW[(cur, tail)]:
                        if idx == tail:
                            predicts.append((chains + [idx], type_id))
                        else:
                            q.append(chains + [idx])

        predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in predicts])
        ent_r += len(ent_set)
        ent_p += len(predicts)
        ent_c += len(predicts.intersection(ent_set))
    return ent_c, ent_p, ent_r


def write_outputs(outputs, entities, length):
    predict_list =[]
    for index, (instance, ent_set, l) in enumerate(zip(outputs, entities, length)):
        predicts = []
        for i in range(l):
            for j in range(i, l):
                if instance[i, j] != 0:
                    temp = []
                    for k in range(i, j+1):
                        temp.append(k)
                    predicts.append(temp)

        forward_dict = {}
        head_dict = {}
        ht_type_dict = {}

        for i in range(l):  # 修改
            for j in range(i, l):
                if instance[i, j] > 0:  ##修改  >1 >0
                    ht_type_dict[(i, j)] = instance[i, j]
                    if i not in head_dict:
                        head_dict[i] = {j}
                    else:
                        head_dict[i].add(j)
        predicts = set([convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts])
        predict_list.append(predicts)
    return predict_list


def to_out(data_1, data_2, logger):
    # f = open('预测结果.json', 'w', encoding='utf-8')
    # f.write(json.dumps(data_1))
    # f2 = open('标准答案.json', 'w', encoding='utf-8')
    # f2.write(json.dumps(data_2))

    for i in range(1, 10):
        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        for one_batch_1, one_batch_2 in zip(data_1, data_2):
            ent_r, ent_p, ent_c = 0, 0, 0
            pred_one_batch = set()
            ground_truth_one_batch = set()
            for item_1, item_2 in zip(one_batch_1, one_batch_2):

                if item_1 != set():
                    for entites in item_1:
                        nums = len(entites.split('-'))
                        if nums == 3:
                            entity_length = 1
                        else:
                            entity_length = int(entites.split('-')[-3]) - int(entites.split('-')[0]) +1
                        if i <= 8 :
                            if entity_length == i:
                                pred_one_batch.add(entites)
                        else:
                            if entity_length >=9:
                                pred_one_batch.add(entites)


                if item_2 != set():
                    for entites in item_2:
                        nums = len(entites.split('-'))
                        if nums == 3:
                            entity_length = 1
                        else:
                            entity_length = int(entites.split('-')[-3]) - int(entites.split('-')[0]) +1
                        if i <= 8:
                            if entity_length == i:
                                ground_truth_one_batch.add(entites)
                        else:
                            if entity_length >= 9:
                                ground_truth_one_batch.add(entites)


                # for ent, ground in zip(pred_one_batch, ground_truth_one_batch):
                ent_r += len(ground_truth_one_batch)
                ent_p += len(pred_one_batch)
                for x in pred_one_batch:
                    if x in ground_truth_one_batch:
                        ent_c += 1
            total_ent_r += ent_r
            total_ent_p += ent_p
            total_ent_c += ent_c
        print('实体长度：', i)
        e_f1, e_p, e_r = cal_f1(total_ent_c, total_ent_p, total_ent_r)
        table = pt.PrettyTable(["Name", 'F1', "Precision", "Recall"])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])
        # print('P\tR\tF\t')
        # print(e_p,'\t',e_r, '\t', e_f1,'\t')

        logger.info("\n{}".format(table))
    print('done!')


def cal_f1(c, p, r):
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r
