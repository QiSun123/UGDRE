import os
import os.path
import json
import numpy as np




def to_official(meta,preds, features,entrs):
    rel2id = json.load(open(meta, 'r'))
    id2rel = {value: key for key, value in rel2id.items()}
    h_idx, t_idx, title = [], [], []
    title_inter_mask={}
    # print(preds.shape)
    # print(len(entrs))

    for f in features:
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]
        title_inter_mask[f["title"]]=f['inter_mask']


    res = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        j=0
        for p in pred:
            if p != 0:
                inter=title_inter_mask[title[i]][h_idx[i]][t_idx[i]]
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2rel[p],
                        'inter':inter,
                        'uncertainty':entrs[i][j]
                    }
                )
            j +=1
    return res


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        # print("gen fact: ",fact_file_name)
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate(tmp,train_path, truthPath ,path):
    '''
        Adapted from the official evaluation code
    '''

    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, train_path), truth_dir)
    fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, truthPath)))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    truth_inter=0
    truth_intra=0

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        inter_mask = []
        for i in range(0, len(x['vertexSet'])):
            inter_mask.append([1] * len(x['vertexSet']))
        for i in range(0, len(x['vertexSet'])):
            inter_mask[i][i] = 0

        for i, ent1 in enumerate(x['vertexSet']):
            for j, ent2 in enumerate(x['vertexSet']):
                if i != j:
                    breakFlag = 0
                    for men1 in ent1:
                        for men2 in ent2:
                            if men1['sent_id'] == men2['sent_id']:
                                inter_mask[i][j] = 0
                                inter_mask[j][i] = 0
                                breakFlag = 1
                                break
                        if breakFlag == 1:
                            break

        for label in x['labels']:
                r = label['r']
                h_idx = label['h']
                t_idx = label['t']
                if inter_mask[h_idx][t_idx]==1:
                    truth_inter+=1
                if inter_mask[h_idx][t_idx]==0:
                    truth_intra+=1

                std[(title, r, h_idx, t_idx)] = set(label['evidence'])
                tot_evidences += len(label['evidence'])

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_inter=0
    correct_intra=0

    predict_inter=0
    predict_intra=0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        inter=x['inter']

        if inter==1:
            predict_inter+=1
        if inter==0:
            predict_intra+=1

        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            if inter==1:
                correct_inter+=1
            if inter==0:
                correct_intra+=1
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1

    inter_p=1.0*correct_inter/predict_inter
    inter_r=1.0*correct_inter/truth_inter
    if inter_p+inter_r==0:
        inter_f1=0
    else:
        inter_f1=2.0*inter_p*inter_r/(inter_p+inter_r)
    # print('inter performance')
    # print(inter_p*100)
    # print(inter_r*100)
    # print(inter_f1*100)

    intra_p=1.0*correct_intra/predict_intra
    intra_r=1.0*correct_intra/truth_intra
    if intra_p+intra_r==0:
        intra_f1=0
    else:
        intra_f1=2.0*intra_p*intra_r/(intra_p+intra_r)
    # print('intra performance')
    # print(intra_p*100)
    # print(intra_r*100)
    # print(intra_f1*100)

    # print('other data')
    # print(correct_inter)
    # print(correct_intra)
    # print(correct_re)
    # print(predict_inter)
    # print(predict_intra)
    # print(len(submission_answer))
    # print(truth_inter)
    # print(truth_intra)
    # print(tot_relations)


    # print('tatal performance')

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    # print("precision")
    # print(re_p* 100)
    # print("recall")
    # print(re_r* 100)
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train
