import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tabulate import tabulate
import string
import re
import pandas as pd
import operator
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def find_voice(dataset):
	for key,value in dataset.items():
		value['first_person'] = 0
		value['second_person'] = 0
		value['third_person_male'] = 0
		value['third_person_female'] = 0
		value['third_person_plural'] = 0
		
		#get counts of personal pronouns for voice identification
		for j,word in enumerate(value['pos_text']):
			if word[0] in ['I','we','We'] and value['pos_text'][j + 1][1] in ['VB','VBZ','VBG','VBP','VBN','VBD','MD']:
				value['first_person'] += 1
			elif word[0] in ['he','He'] and value['pos_text'][j + 1][1] in ['VB','VBZ','VBG','VBP','VBN','VBD','MD']:
				value['third_person_male'] += 1
			elif word[0] in ['she','She'] and value['pos_text'][j + 1][1] in ['VB','VBZ','VBG','VBP','VBN','VBD','MD']:
				value['third_person_female'] += 1
			elif word[0] in ['You','you'] and value['pos_text'][j + 1][1] in ['VB','VBZ','VBG','VBP','VBN','VBD','MD']:
				value['second_person'] += 1
		max_third_person = max(value['third_person_female'],value['third_person_male'])
		
		if value['first_person'] > max_third_person and value['first_person'] > value['second_person']:
			value['predicted_voice'] = 'First'
		elif max_third_person > value['first_person'] and max_third_person > value['second_person']:
			value['predicted_voice'] = 'Third'
		elif value['second_person'] > value['first_person'] and value['second_person'] > max_third_person:
			value['predicted_voice'] = 'Second'
	return dataset
	
def find_protagonist(dataset):
	for key,value in dataset.items():
		name_list = {}
		protagonist = ''
		
		#get list of all names in the story text
		for j,word in enumerate(value['ner_text']):
			if word[1] == 'PERSON' and j + 1 < len(value['ner_text']) and value['pos_text'][j + 1][1] in ['VB','VBZ','VBG','VBP','VBN','VBD','MD'] and value['pos_text'][j][1] in ['NNP']:	
				name_list.setdefault(word[0],0)
				name_list[word[0]] += 1
				
		#if predicted voice is third person, go through name list
		if value['predicted_voice'] == 'Third':
			if len(name_list) > 0:
				last_para = value['text'].split("\n")[-1]
				first_para = value['text'].split("\n")[0]
				
				#get two names with the most number of mentions
				protagonist_l = sorted(name_list.items(), key=operator.itemgetter(1), reverse=True)[:2]
				
				#check to see if either of the two names is in both the first and last paragraphs of the story text
				if len(protagonist_l) > 1 and protagonist_l[0][1] > 10 and protagonist_l[0][1] - protagonist_l[1][1] < 10:
					if protagonist_l[0][0] in first_para and protagonist_l[0][0] in last_para:
						protagonist = protagonist_l[0][0]
					elif protagonist_l[1][0] in first_para and protagonist_l[1][0] in last_para: 
						protagonist = protagonist_l[1][0]
					else:
						protagonist = protagonist_l[0][0]
				else:
					protagonist = protagonist_l[0][0]
			else:
				protagonist = 'Unnamed'
		elif value['predicted_voice'] == 'Second':
			protagonist = 'Reader'
		elif value['predicted_voice'] == 'First':
			protagonist = 'Unnamed'
			
		#get full name and title of the assumed protagonist
		for j,word in enumerate(value['pos_text']):
			if word[0] == protagonist:
				if value['pos_text'][j + 1][1] == 'NNP' and value['ner_text'][j + 1][1] == 'PERSON':
					last_name = re.sub(r'[^\w\s]','',value['pos_text'][j + 1][0])
					protagonist = protagonist + " " + last_name
				elif value['pos_text'][j - 1][1] == 'NNP' and value['ner_text'][j - 1][1] == 'PERSON' and not any(k in value['pos_text'][j - 1][0] for k in ('.',',','?')):
					protagonist = value['pos_text'][j - 1][0] + " " + protagonist
				
				if value['pos_text'][j - 1][0] in ['Mr.','Mrs.','Dr.','Ms.']:
					protagonist = value['pos_text'][j - 1][0] + " " + protagonist

		if any(k in protagonist for k in ('Mr.','Mrs.','Dr.','Ms.')):
			if value['text'].count(protagonist) == 1:
				p = protagonist.split(".")
				protagonist = p[1].strip()
		value['predicted_protagonist'] = protagonist
	return dataset

def find_style(dataset):
	for key,value in dataset.items():
		avg_word_length = 0
		avg_paragraph_length = 0
		value['long_words'] = 0
		value['adj_count'] = 0
		value['quotation_count'] = 0
		#clean up the text for extracting style features
		text = value['text'].replace('<em>','')
		text = text.replace('</em>','')
		translator = text.maketrans('', '', string.punctuation)
		text = text.translate(translator)
		text = text.replace("“",'')
		text = text.replace("”",'')
		value['wc'] = len(text.split())
		
		#extract long word count
		for word in text.split():
			avg_word_length += len(word)
			if len(word) > 7:
				value['long_words'] += 1
		
		#extract adjective count
		for word in value['pos_text']:
			if word[1] in ['JJ','JJR','JJS']:
				value['adj_count'] += 1
		
		#extract double quotation count
		for word in value['text']:
			if "“" in word or "”" in word or '"' in word:
				value['quotation_count'] += 1
		
		#get percentages of counts 
		value['avg_word_length'] = avg_word_length/value['wc']
		value['adj_count'] = value['adj_count']/value['wc']
		value['long_words'] = value['long_words']/value['wc']
	
	#convert dictionary to a dataframe and calculate percentile rank from 1-10 of every value
	df = pd.DataFrame.from_dict(dataset,orient='index')
	df['descriptiveness'] = pd.qcut(df.adj_count,10,duplicates="drop",labels=False)
	df['wordiness'] = pd.qcut(df.wc,10,duplicates="drop",labels=False)
	df['difficulty'] = pd.qcut(df.long_words,10,duplicates="drop",labels=False)
	df['dialogue-heavy'] = pd.qcut(df.quotation_count,10,duplicates="drop",labels=False)
	return df

def predict_theme(trainset,testset):
	trainset_stories = []
	testset_stories = []
	trainset_labels = []
	testset_labels = []
	i = 0
	
	tf = TfidfVectorizer(sublinear_tf=True,min_df=5, max_df = 0.8, stop_words = 'english',ngram_range=(1, 3))
	model3 = MLPClassifier(random_state=0,activation='identity')
	for key,value in trainset.items():
		trainset_stories.append(value['text'])
		trainset_labels.append(value['theme'])
	for key,value in testset.items(): 
		testset_stories.append(value['text'])
		testset_labels.append(value['theme'])
	#get the tf-idf vector of the text of each story in training and test sets
	trainset_tfidf =  tf.fit_transform(trainset_stories)
	testset_tfidf = tf.transform(testset_stories)
	
	#fit the MLP classifier to the training data
	model3.fit(trainset_tfidf,trainset_labels)
	
	#get results
	results = model3.predict(testset_tfidf)
	for key,value in testset.items():
		value['predicted_theme'] = results[i]
		i += 1
	return testset
	
def print_accuracy(dataset,feature):
	y_pred,y_test = [],[]
	labels = []
	if feature == 'voice':
		for key,value in dataset.items():
			y_pred.append(value['predicted_voice'])
			y_test.append(value['voice'])
	elif feature == 'protagonist':
		for key,value in dataset.items():
			y_pred.append(value['predicted_protagonist'])
			y_test.append(value['protagonist'])
	elif feature == 'theme':
		for key,value in dataset.items():
			y_pred.append(value['predicted_theme'])
			y_test.append(value['theme'])
	#average for the calculation here is macro because for voice and theme, the labels are balanced
	if feature in ['voice','theme']:
		f1 = f1_score(y_test, y_pred, average="macro",labels=np.unique(y_pred))
		precision = precision_score(y_test, y_pred, average="macro",labels=np.unique(y_pred))
		recall = recall_score(y_test, y_pred, average="macro",labels=np.unique(y_pred))
	else:
		#average for the calculation here is weighted because for protagonist the labels are not balanced
		f1 = f1_score(y_test, y_pred, average="weighted")
		precision = precision_score(y_test, y_pred, average="weighted")
		recall = recall_score(y_test, y_pred, average="weighted")
	print(feature + " scores:")
	print("Precision score: " + str(precision))
	print("Recall score: " + str(recall))
	print("F1 score: "  + str(f1))
	print("\n")

def print_style_statistics(df):
	df1 = df[['title','wc','descriptiveness','wordiness','difficulty','dialogue-heavy']]
	print(tabulate(df1, headers='keys', tablefmt='psql'))

training_set = find_voice(np.load('data/training_set.npy').item())
test_set = find_voice(np.load('data/test_set.npy').item())

training_set = find_protagonist(training_set)
test_set = find_protagonist(test_set)
test_set = predict_theme(training_set,test_set)

whole_set = {**training_set,**test_set}
df = find_style(whole_set)
df1 = df[['title','text','URL','author','issue','tags','voice','protagonist','theme']]
df1.to_csv('whole_dataset.csv')
df2 = df[['title','author','URL','descriptiveness','wordiness','difficulty','dialogue-heavy']]
df2.to_csv('dataset_style.csv')
#print f1,recall,precision scores
print("Training set scores:")
print_accuracy(training_set,'voice')
print_accuracy(training_set,'protagonist')
print("Test set scores:")
print_accuracy(test_set,'voice')
print_accuracy(test_set,'protagonist')
print_accuracy(test_set,'theme')
print("Whole set scores:")
print_accuracy(whole_set,'voice')
print_accuracy(whole_set,'protagonist')
print_style_statistics(df)
	