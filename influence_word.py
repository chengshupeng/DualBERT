import torch
from transformers import BertTokenizer, BertModel
from torch.nn.functional import cosine_similarity,normalize
# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('../bert/')
model = BertModel.from_pretrained('../bert/')
# 获取模型输出
# outputs = model(inputs)
# 输出向量
# original_vector = outputs.last_hidden_state.mean(dim=1)  # 句子的原始向量
def forward_effect(index,sentence): #mask Xi-1对Xi影响
    # 对句子进行分词和编码
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))
    inputs = tokenizer.encode(sentence, return_tensors="pt")
    # 获取mask掉某个位置后的向量
    mask_index1 = index+1 # 假设要mask掉第一个词(xi)
    masked_inputs1 = inputs.clone()
    masked_inputs1[0, mask_index1] = tokenizer.mask_token_id
    masked_outputs1 = model(masked_inputs1)
    masked_vector1 = masked_outputs1.last_hidden_state.mean(dim=1)  # mask掉xi个位置后的向量

    mask_index2 = [mask_index1-1,mask_index1] #(xi-1,xi)
    masked_inputs2 = inputs.clone()
    masked_inputs2[0, mask_index2] = tokenizer.mask_token_id
    masked_outputs2 = model(masked_inputs2)
    masked_vector2 = masked_outputs2.last_hidden_state.mean(dim=1)  # mask掉xi-1,xi个位置后的向量

    #print(mask_index1,mask_index2)
    # 计算余弦相似度
    similarity = cosine_similarity(masked_vector1, masked_vector2)

    #print("余弦相似度:", similarity.item())

    # 标准化向量
    vector1_normalized = normalize(masked_vector1, p=2, dim=1)
    vector2_normalized = normalize(masked_vector2, p=2, dim=1)

    # 计算欧氏距离
    euclidean_distance = torch.norm(vector1_normalized - vector2_normalized, p=2)

    #print("欧氏距离:",(1 / (1 + euclidean_distance.item())))

    alpha = 0.1
    beta = 0.9

    # 计算综合相似度指标 W
    W = alpha * similarity + beta * (1-euclidean_distance)

    #print("归一化结果：",W.item())
    return W

def reverse_effect(index,sentence): #mask Xi对Xi-1影响
    # 对句子进行分词和编码
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))
    inputs = tokenizer.encode(sentence, return_tensors="pt")
    # 获取mask掉某个位置后的向量
    mask_index1 = index # 假设要mask掉第一个词(xi)
    masked_inputs1 = inputs.clone()
    masked_inputs1[0, mask_index1] = tokenizer.mask_token_id
    masked_outputs1 = model(masked_inputs1)
    masked_vector1 = masked_outputs1.last_hidden_state.mean(dim=1)  # mask掉xi-1个位置后的向量

    mask_index2 = [mask_index1,mask_index1+1] #(xi-1,xi)
    masked_inputs2 = inputs.clone()
    masked_inputs2[0, mask_index2] = tokenizer.mask_token_id
    masked_outputs2 = model(masked_inputs2)
    masked_vector2 = masked_outputs2.last_hidden_state.mean(dim=1)  # mask掉xi-1,xi个位置后的向量

    #print(mask_index1, mask_index2)
    # 计算余弦相似度
    similarity = cosine_similarity(masked_vector1, masked_vector2)

    #print("余弦相似度:", similarity.item())

    # 标准化向量
    vector1_normalized = normalize(masked_vector1, p=2, dim=1)
    vector2_normalized = normalize(masked_vector2, p=2, dim=1)

    # 计算欧氏距离
    euclidean_distance = torch.norm(vector1_normalized - vector2_normalized, p=2)

    #print("欧氏距离:",(1 / (1 + euclidean_distance.item())))

    alpha = 0.1
    beta = 0.9

    # 计算综合相似度指标 W
    W = alpha * similarity + beta * (1-euclidean_distance)

    #print("归一化结果：",W.item())
    return W

def effect(index,sentence):
    return (forward_effect(index,sentence)+reverse_effect(index,sentence))/2 #正+反向影响
index=("吕梁期岩浆活动主要为北西-近南北向展布的变辉绿岩脉(墙)，脉宽20m～40m，矿区西侧的一条变辉绿岩岩脉长2750m，两端延出区外。").find('岩')
print(effect(index,"吕梁期岩浆活动主要为北西-近南北向展布的变辉绿岩脉(墙)，脉宽20m～40m，矿区西侧的一条变辉绿岩岩脉长2750m，两端延出区外。").item())
