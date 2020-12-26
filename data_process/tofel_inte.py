import csv
from stanfordcorenlp import StanfordCoreNLP
import os
'''
filedir = os.getcwd() + '\\tofel_essays_train\\essays_train'
filenames=os.listdir(filedir)
f = open('tofel_sum.txt','w')
for filename in filenames:
    filepath = filedir + '/' + filename
    for line in open(filepath):
        f.writelines(line)
f.close()
'''
#model = StanfordCoreNLP(r"C:\Users\shuaixiaoduo\Desktop\stanford-corenlp-latest\stanford-corenlp-4.1.0",lang="en")
model = StanfordCoreNLP('/data/lmy/xbx/metaphor_detection/stanford-corenlp-4.1.0',lang="en")
f = open('tofel_sum.txt',encoding='latin-1')
g = open('tofel_preformatted.csv' , 'w', encoding='latin-1')
csv_writer = csv.writer(g)
for line in f:
    line = line.replace('dont', 'do not')
    line = line.replace('-',' ')
    line = line.replace('etc ..','etc .')
    line = line.replace('kg ', ' kg ')
    line = line.replace('i .e', ' ')
    line = line.replace('e .g', ' ')
    line = line.replace('gonna', 'go to')
    line = line.replace('doesnt', 'does not')
    line = line.replace('isnt', 'is not')
    line = line.replace('didnt', 'did not')
    formated = []
    label = []
    pos = []
    st = line.split()

    for i in range(len(st)):
        if st[i].find('M_') != -1:
            st[i] = st[i].replace('M_', '')
            label.append(1)
        else:
            label.append(0)
    line = line.replace('M_', '')
    line = line.replace('\n', '')
    result = model.pos_tag(line)
    for k in result:
        pos.append(k[1])
    assert len(label) == len(line.split())
    assert len(label) == len(pos), line
    formated.append(line)
    formated.append(label)
    formated.append(pos)
    csv_writer.writerow(formated)