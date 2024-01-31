from itertools import groupby
import influence_word
# 初始化一个空的列表用于存储数据集
with open('res800bmes.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
tags = []
tokens=[]
i=0
for line in lines:
    j=0
    if line!='\n':
        token, tag = line.strip().split('\t')
        tokens.append(token)
        tags.append(tag)
    else:
        i=i+1
        tokens.append('')
        tags.append('')
# 使用空元素进行分割
results = [list(g) for k, g in groupby(tags, key=lambda x: x == '') if not k]

data_list = []
with open('res800org.txt', encoding='utf-8') as f:
    sentences = f.read().split('\n')
# 循环创建100组数据集
i=0
for sentence in sentences:
    sentence = f"{sentence}"
    char_array = [char for char in sentence]
    label_array = results[i]  # 示例的标签数组，可以根据实际情况更改
    i=i+1
    # 创建数据集字典
    data_set = {
        "sentence": sentence,
        "char_array": char_array,
        "label_array": label_array
    }

    # 将数据集添加到列表中
    data_list.append(data_set)

# 打印整体数据
#print(data_list[5]["sentence"][5],data_list[5]["char_array"][5],data_list[5]["label_array"][5])
with open('dict.txt', 'r',encoding='utf-8') as f:
    dictionary = [g for g in f.read().split('\n')]
unique_characters = set()
# 遍历列表中的每个字符串，将其中的字符加入集合
for string in dictionary:
    unique_characters.update(string)
def merge(Str):
    if score > 0.92:
        label_array[i] = Str
        label_array[i + 1] = 'E'
        if i + 2 < len(label_array):
            if label_array[i + 2] == 'M':
                label_array[i + 2] = 'B'
            elif label_array[i + 2] == 'E':
                label_array[i + 2] = 'S'
for data in data_list:
    sentence=data["sentence"]
    char_array=data["char_array"]
    label_array=data["label_array"]
    for i in range(len(char_array)):
        char=char_array[i]
        label=label_array[i]
        if char in unique_characters:
            if char_array[i+1] in unique_characters:
                if char!='。' and label=='E':
                    score=influence_word.effect(i,sentence).item()
                    merge('M')
                if char!='。' and label=='S':
                    score = influence_word.effect(i, sentence).item()
                    merge('B')

