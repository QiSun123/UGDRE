import copy
from tqdm import tqdm
import ujson as json

def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res

def addEntitySentence(entities,curSent):
    vertexSentence = {}
    for enti in entities:
        for menti in enti:
            key = menti['sent_id'] * 10 + menti['pos'][0]
            vertexSentence[key] =copy.deepcopy(menti)
    vertexlists = sorted(vertexSentence.items(),key=lambda item:item[0])
    vertexsentencelist=[]
    for item in vertexlists:
        vertexsentencelist+=curSent[item[1]['sent_id']][item[1]['pos'][0]:item[1]['pos'][1]]
    return vertexsentencelist

def read_docred(meta,file_in, tokenizer, max_seq_length=1024):
    docred_rel2id = json.load(open(meta, 'r'))
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)
    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []
        words=[]
        token_map=[]
        lengthofPice=0
        mentions=[]
        inter_mask=[]
        for i in range(0, len(sample['vertexSet'])):
            inter_mask.append([1] * len(sample['vertexSet']))
        for i in range (0,len(sample['vertexSet'])):
            inter_mask[i][i]=0
        for i, ent1 in enumerate(sample['vertexSet']):
            for j, ent2 in enumerate(sample['vertexSet']):
                if i!=j:
                    breakFlag=0
                    for men1 in ent1:
                        for men2 in ent2:
                            if men1['sent_id'] == men2['sent_id']:
                                inter_mask[i][j] = 0
                                inter_mask[j][i] = 0
                                breakFlag=1
                                break
                        if breakFlag==1:
                            break
        entities = sample['vertexSet']
        vertexsentencelist = addEntitySentence(entities, sample['sents'])
        entity_start, entity_end = {}, {}
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                mentions.append(pos)

                entity_start[(sent_id, pos[0],)]="*"
                entity_end[(sent_id, pos[1] - 1,)]="*"

        newsents = sample['sents']
        newsents.append(vertexsentencelist)

        for i_s, sent in enumerate(newsents):
            new_map = {}
            for i_t, token in enumerate(sent):

                oneToken=[]
                words.append(token)
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start.keys():
                    tokens_wordpiece = [entity_start[(i_s, i_t)]] + tokens_wordpiece
                    oneToken.append(lengthofPice + 1)
                    lengthofPice += len(tokens_wordpiece)
                    oneToken.append(lengthofPice)

                elif (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + [entity_end[(i_s, i_t)] ]
                    oneToken.append(lengthofPice)
                    lengthofPice += len(tokens_wordpiece)
                    oneToken.append(lengthofPice-1)
                else:
                    oneToken.append(lengthofPice)
                    lengthofPice += len(tokens_wordpiece)
                    oneToken.append(lengthofPice)
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
                token_map.append(oneToken)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        sent_occur={}
        ei=0
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))
                if m["sent_id"] not in sent_occur.keys():
                    sent_occur[m["sent_id"]]=[]
                if ei not in sent_occur[m["sent_id"]]:
                    sent_occur[m["sent_id"]].append(ei)
            ei+=1
        relations, hts = [], []
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)
        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        A=[]
        for i in range(0,len(input_ids)):
            A.append([0]*len(input_ids))
        offset=1
        edges=0
        for token_s in token_map:
            start=token_s[0]+offset
            end=token_s[1]+offset
            for i in range(start,end):
                for j in range (start,end):
                    if i < (len(input_ids) - 1) and j < (len(input_ids) - 1):
                        if A[i][j] == 0:
                            A[i][j] = 1
                            edges += 1
        mentionsofPice=[]
        for ment in mentions:
            mentionsofPice.append([token_map[ment[0]][0],token_map[ment[1]-1][1]])
        for ment in mentionsofPice:
            start=ment[0]+offset
            end=ment[1]+offset
            for i in range(start, end):
                for j in range(start, end):
                    if i < (len(input_ids) - 1) and j < (len(input_ids) - 1):
                        if A[i][j] == 0:
                            A[i][j] = 1
                            edges += 1
        entityofPice=[]
        for ent in entity_pos:
            oneEntityP=[]
            for ment in ent:
                if (ment[0]+offset)==(ment[1]-offset):
                    oneEntityP.append(ment[0]+offset)
                for i in range(ment[0]+offset,ment[1]-offset):
                    oneEntityP.append(i)
            entityofPice.append(oneEntityP)
        predicted_Doc2=[]
        for h in range(0,len(entities)):
            item=[0,h]
            predicted_Doc2.append(item)

        predictedEntityPairPiece = []
        for item in predicted_Doc2:
            one_predicted=entityofPice[item[0]]+entityofPice[item[1]]
            predictedEntityPairPiece.append(one_predicted)

        for line in predictedEntityPairPiece:
            for i in line:
                for j in line:
                    if A[i + offset][j + offset] == 0:
                        A[i + offset][j + offset] = 1
                        edges += 1

        for i in range(0, len(A)):
            A[i][i]=1

        i_line += 1
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'title': sample['title'],
                   'Adj':A,
                   'inter_mask':inter_mask,
                   }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features
