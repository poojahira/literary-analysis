import re
import numpy as np
from nltk.tag.stanford import StanfordNERTagger
from nltk import pos_tag, word_tokenize
jar = 'apps/stanford-ner-2018-02-27/stanford-ner-3.9.1.jar'
model = 'apps/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz'
tagger = StanfordNERTagger(model, jar, encoding='utf8')
files = ['data/stories-adultery.txt','data/stories-death.txt','data/stories-children.txt','data/stories-immigrants.txt']
training_set = {}
dev_set = {}
test_set = {}
j = 0

def input_fields(content,structure):
	structure['title'] = content[0].strip()
	structure['text'] = re.sub('<[/]?span.*?>', '',content[1])
	structure['text'] = re.sub('\[cartoon.*?\]','',structure['text'])
	structure['text'] = re.sub('</p><p>',"\n",structure['text'])
	structure['text'] = re.sub('<[/]?p>','',structure['text'])
	structure['text'] = re.sub('♦','',structure['text'])
	structure['text'] = re.sub('<a.*?>.*?</a>','',structure['text'])
	structure['pos_text'] = pos_tag(structure['text'].split())
	structure['ner_text'] = tagger.tag(structure['text'].split())
	structure['author'] = content[2].strip()
	structure['issue'] = content[3].strip()
	structure['URL'] = content[4].strip()
	structure['tags'] = [content[5].strip(),content[6].strip(),content[7].strip(),content[8].strip(),content[9].strip(),content[10].strip()]
	structure['tags'] = list(filter(None, structure['tags']))
	structure['protagonist'] = content[11].strip()
	structure['voice'] = content[12].strip()
	structure['theme'] = content[13].strip()
	return structure

for file in files:
	i = 0
	print(file)
	with open(file) as f:
		for line in f:
			content = line.split("\t")
			if i < 18:
				training_set[j] = input_fields(content,{})
			elif i >= 18:
				test_set[j] = input_fields(content,{})
			i += 1
			j += 1
		f.close()

np.save('data/training_set.npy', training_set)
#np.save('dev_set.npy',dev_set)		
np.save('data/test_set.npy', test_set)
		