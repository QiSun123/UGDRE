import copy
import json
import numpy as np
import matplotlib.pyplot as plt


def getTitleMap(predict):
    predictMap = {}
    for item in predict:
        if item['title'] not in predictMap.keys():
            predictMap[item['title']] = [item]
        else:
            predictMap[item['title']].append(item)
    return predictMap

def Avg_type_Uncertainty(predictMap):
    idmap = json.load(open('../meta/rel2id.json'))
    type_UList = {}
    Avg_type_UList = [0] * len(idmap)
    for ids in idmap:
        type_UList[idmap[ids]]=[]

    for item in predictMap:
        for it in predictMap[item]:
            type_UList[idmap[it['r']]].append(it['uncertainty']) 

    for i in range(0, len(type_UList)):
        if type_UList[i] == []:
            Avg_type_UList[i] = 0
        else:
            type_UList[i].sort()
            std=np.std(type_UList[i],ddof=1)
            mean=np.mean(type_UList[i])
            Avg_type_UList[i]=mean+std

    return Avg_type_UList


def Calculate_psudo(x, DS_label):
    idmap = json.load(open('../meta/rel2id.json'))
    psudo = {}
    for title in x:
        if title in DS_label:
            val = copy.deepcopy(DS_label[title])
            for it in val:
                val[it] = []
            for item in x[title]:
                val[str(item['h_idx']) + '|' + str(item['t_idx'])].append(idmap[item['r']])
            psudo[title] = val
    return psudo

def Calculate_psudo_uncertainty(xu, DS_label):
    uncertain_map = {}
    for title in xu:
        if title in DS_label:  
            val = copy.deepcopy(DS_label[title])
            for it in val:
                val[it] = []
            for item in xu[title]:
                val[str(item['h_idx']) + '|' + str(item['t_idx'])].append(item['uncertainty'])
            uncertain_map[title] = val
    return uncertain_map

def relabel(origin_path,predict_S_path, save_relabel_path):
    idmap = json.load(open('../meta/rel2id.json'))
    origin = json.load(open(origin_path))
    predict_S = json.load(open(predict_S_path))
    print(len(origin))

    predictMap_S = getTitleMap(predict_S)
    Avg_type_UList_S = Avg_type_Uncertainty(predictMap_S)

    DS_label = {}
    for item in origin:
        val = {}
        for i in range(0, len(item['vertexSet'])):
            for j in range(0, len(item['vertexSet'])):
                k = str(i) + '|' + str(j)
                val[k] = []
        for lab in item['labels']:
            k = str(lab['h']) + '|' + str(lab['t'])
            val[k].append(idmap[lab['r']])
        DS_label[item['title']] = val
    psudo_label_S = Calculate_psudo(predictMap_S, DS_label)
    psudo_uncertain_S = Calculate_psudo_uncertainty(predictMap_S, DS_label)

    static_same={}
    static_relabel={}
    new_label = {}
    static_pos={}
    static_filter = {}

    for title in DS_label:
        l1 = DS_label[title]
        static_same[title] = 0
        static_relabel[title] = 0
        static_pos[title] = 0
        static_filter[title] = 0

        if title not in psudo_label_S.keys():
            val=copy.deepcopy(l1)
            for pair in val:
                static_pos[title] += len(val[pair])
        else:
            l2 = psudo_label_S[title]
            u2 = psudo_uncertain_S[title]
            val = {}

            for pair in l1:
                l1[pair].sort()

                if l1[pair] == l2[pair]:
                    val[pair] = l1[pair]

                    static_pos[title] += len(val[pair])
                    static_same[title] += len(val[pair])
                else:

                    if l2[pair] == []:
                        val[pair] = l1[pair]
                        static_pos[title] += len(val[pair])
                    else:
                        relabeled=[]
                        for idx in range(0,len(l2[pair])):

                            if u2[pair][idx] <= Avg_type_UList_S[l2[pair][idx]]:
                                relabeled.append(l2[pair][idx])
                            else:
                                static_filter[title]+=1
                        relabeled.sort()
                        if relabeled!=[] and relabeled!=l1[pair]:

                            val[pair] = relabeled
                            static_pos[title] += len(val[pair])
                            for op in relabeled:
                                if op not in l1[pair]:
                                    static_relabel[title] += 1
                                else:
                                    static_same[title] += 1
                        else:
                            val[pair] = l1[pair]
                            static_pos[title] += len(val[pair])
                            static_same[title] += len(val[pair])
        new_label[title]=val

    #print("******************** statistic *********************")
    total_pos = 0
    total_same = 0
    total_relabel = 0
    total_filter = 0
    for title in static_pos:
        total_pos += static_pos[title]
        total_same += static_same[title]
        total_relabel += static_relabel[title]
        total_filter += static_filter[title]
    #print("pseudo label nums: ", len(predict_S))
    #print("relabed total_positive_instance: " + str(total_pos))
    #print("ratio of same as dis and pesudo: " + str((total_same / total_pos) * 100))
    #print("ratio of relabel: " + str((total_relabel / total_pos) * 100))
    #print("ratio of filter by high u: " + str((total_filter / total_pos) * 100))

    train_new=[]
    map2id={}
    for k in idmap:
        map2id[idmap[k]]=k
    for item in origin:
        new_doc=copy.deepcopy(item)
        labels=[]
        mix_labels=new_label[item['title']]
        for key in mix_labels:
            if mix_labels[key]!=[]:
                for it in mix_labels[key]:
                    one = {}
                    one['h'] = int(key.split('|')[0])
                    one['t'] = int(key.split('|')[1])
                    one['r'] =map2id[it]
                    one['evidence']=[]
                    labels.append(one)
        new_doc['labels']=labels
        train_new.append(new_doc)
    print('len of new train')
    print(len(train_new))
    train_new = json.dumps(train_new)

    with open(save_relabel_path, 'w') as f1:
        f1.write(train_new)
    print("end relabel")

# iterative re-label
origin_path = 'docred/train_distant.json'
predict_S_path = '../results/result_UGDRE-RE-finetune1.json'
save_relabel_path = "docred/train_UGDRE-RE-finetune1.json"
relabel(origin_path,predict_S_path,save_relabel_path)

