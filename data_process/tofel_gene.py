'''
Verbs Adjectives Nouns Adverbs
DT 限定词 忽略
WDT WH开头的限定词 忽略
*NNP 专有名词 认作名词
*VBG 动名词 认作动词
PDT 前置限定词 如all 忽略
*NNPS 专有名词复数 认作名词
UH 感叹词 忽略
POS 以's 结束的词忽略
WP$ 以WH 开头的所有格 忽略
*VB 动词原形 当作动词
CC 并列连词 忽略
TO 单词to 忽略
*RBS 副词最高级 当作副词
*NN NNS 普通名词 当作名词
RP 小品词 忽略
*RBR 程度副词 当作副词
*JJ JJR JJS 形容词 当作形容词
*MD 情态动词 当作动词
*VBD 动词过去时态 当作动词
PRP PRP$ 代词 忽略
*RB 副词
*WRB W开头的副词 副词
EX there 忽略
*VBN VBP VBZ 动词
FW 外来词 理论上不能忽略，但是没法统计所以忽略
Verbs : VB VBD VBG VBN VBP VBZ
Adjectives : JJ JJR JJS
Nouns : NN NNS NNP NNPS
Adverbs : RB RBR RBS WRB
'''
verb = ['VB','VBD','VBG','VBN','VBP','VBZ']
adj = ['JJ','JJR','JJS']
nos = ['NN','NNS','NNP','NNPS']
adv = ['RB','RBR','RBS','WRB']
import csv
import ast
data = []
char2pos = {}
for i in verb:
    char2pos[i] = 'Verbs'
for i in adj:
    char2pos[i] = 'Adjectives'
for i in nos:
    char2pos[i] = 'Nouns'
for i in adv:
    char2pos[i] = 'Adverbs'
formated = []
g = open('../datasets/tofel/tofel_allpos_formatted.csv', 'w',newline='', encoding='latin-1')
csv_writer = csv.writer(g)
csv_writer.writerow(['sentence','word','pos','label','idx'])
with open('tofel_preformatted.csv',encoding= 'latin-1')as afile:
    a_reader = csv.reader(afile)
    for row in a_reader:
        st = row[0].split()
        pos_seq = ast.literal_eval(row[2])
        label_seq = ast.literal_eval(row[1])
        assert (len(pos_seq) == len(label_seq))
        assert (len(row[0].split()) == len(pos_seq))
        for i in range(len(pos_seq)):
            if char2pos.get(pos_seq[i]):
                formated.append(row[0])
                formated.append(st[i])
                formated.append(char2pos[pos_seq[i]])
                formated.append(label_seq[i])
                formated.append(i)
                csv_writer.writerow(formated)
                formated = []




