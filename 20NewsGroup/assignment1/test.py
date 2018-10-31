#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:47:43 2018

@author: rasika
"""
import os,os.path
import shutil
import glob
import random 
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk import word_tokenize
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import words
from nltk.corpus import stopwords 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')



classes = ['computer','recreational','science', 'talk']

root = './classes/'

files= [name for name in os.listdir('./classes/computer') if os.path.isfile(os.path.join('./classes/computer', name))]
def split_datafiles(files):
    random.Random(101).shuffle(files)
    train_split = int(0.7 * len(files))
    test_split = int(0.3 * len(files))
    train_files=files[:train_split]
    test_files=files[-test_split:]
    return train_files,test_files # return the list of files names train and test


stop=["a", "about", "above", "above", "across", "after", 
       "afterwards", "again", "against", "all", "almost", "alone", 
       "along", "already", "also","although","always","am","among", 
       "amongst", "amoungst", "amount",  "an", "and", "another", "any",
       "anyhow","anyone","anything","anyway", "anywhere", "are", "around",
       "as",  "at", "back","be","became", "because","become","becomes", 
       "becoming", "been", "before", "beforehand", "behind", "being", 
       "below", "beside", "besides", "between", "beyond", "bill", "both",
       "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", 
       "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", 
       "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", 
       "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
       "everything", "everywhere", "except", "few", "fifteen", "fify", "fill",
       "find", "fire", "first", "five", "for", "former", "formerly", "forty", 
       "found", "four", "from", "front", "full", "further", "get", "give", "go", 
       "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", 
       "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
       "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest",
       "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", 
       "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", 
       "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", 
       "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", 
       "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", 
       "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", 
       "or", "other", "others", "otherwise", "our", "ours", "ourselves", 
       "out", "over", "own","part", "per", "perhaps", "please", "put",
       "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", 
       "serious", "several", "she", "should", "show", "side", "since", 
       "sincere", "six", "sixty", "so", "some", "somehow", "someone", 
       "something", "sometime", "sometimes", "somewhere", "still", "such", 
       "system", "take", "ten", "than", "that", "the", "their", "them", 
       "themselves", "then", "thence", "there", "thereafter", "thereby",
       "therefore", "therein", "thereupon", "these", "they", "thickv",
       "thin", "third", "this", "those", "though", "three", "through", 
       "throughout", "thru", "thus", "to", "together", "too", "top", 
       "toward", "towards", "twelve", "twenty", "two", "un", "under",
       "until", "up", "upon", "us", "very", "via", "was", "we", "well",
       "were", "what", "whatever", "when", "whence", "whenever", "where",
       "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", 
       "whether", "which", "while", "whither", "who", "whoever", "whole", "whom",
       "whose", "why", "will", "with", "within", "without", "would", "yet", "you",
       "your", "yours", "yourself", "yourselves", "the"] 


stop_words = set(stopwords.words('english') + list(string.punctuation) + stop)
tokenizer = nltk.tokenize.TreebankWordTokenizer
word_list = words.words()
stemmer = nltk.stem.WordNetLemmatizer()



def getFileContent(path,file_name):
    with open(path+"/"+file_name, 'r', encoding = "ISO-8859-1") as theFile:
        data = theFile.read().replace('\n', '')
        return data
#text = getFileContent('./classes/computer',files[0])
#words = [stemmer.stem(i) for i in word_tokenize(text.lower()) if i not in stop_words and i in word_list]


def getFileWordsList(p,files_name):
    words=[]
    
    for file_name in files_name[:2]:
        text = getFileContent(p,file_name)
        words+= [stemmer.lemmatize(i) for i in word_tokenize(text.lower()) if i not in stop_words and i in word_list] 
    
    return list(set(words))


def getUniqueWordList():
    uWords_train = []
    uWords_test = []
    
    for class_file in classes:
        p = os.path.join(root,class_file)
        files= [name for name in os.listdir(p) if os.path.isfile(os.path.join(p, name))]
        train_files,test_files = split_datafiles(files)
        
        print("progress "+ class_file)
        
        uWords_train += getFileWordsList(p,train_files)
        print(' number of unique words '+ class_file +' train is :', len(set(uWords_train)))
        
        uWords_test += getFileWordsList(p,test_files)
        print(' number of unique words '+ class_file +' test is : ', len(set(uWords_test)))
      
    return list(set(uWords_train)), list(set(test_files))


uniq_words_train, uniq_words_test = getUniqueWordList()
print(uniq_words_test)

#cv_train = CountVectorizer(vocabulary=uniq_words_train)
#cv_test = CountVectorizer(vocabulary=uniq_words_test)
#
#def create_feature_vector(root,class_file):
#    
#    p = os.path.join(root,class_file)
#    files= [name for name in os.listdir(p) if os.path.isfile(os.path.join(p, name))]
#    train_files,test_files = split_datafiles(files)
#    text_files=[]
#    for file in train_files:
#        text_files.append(getFileContent(p,file))
#        
#    X = cv_train.fit_transform(text_files)
#    Y = cv_train.get_feature_names()
#    df = pd.DataFrame(data=X.toarray(), columns=Y)
#    df.to_csv('./csv/'+class_file+'_train.csv')
#    print('created '+class_file+' train csv')
#    
#    text_files=[]
#    for file in test_files:
#        text_files.append(getFileContent(p,file))
#            
#    X = cv_test.fit_transform(text_files)
#    Y = cv_test.get_feature_names()
#    df = pd.DataFrame(data=X.toarray(), columns=Y)
#    df.to_csv('./csv/'+class_file+'_test.csv')
#    print('created '+class_file+' test csv')
#
#
#    
#create_feature_vector(root,classes[0])
#create_feature_vector(root,classes[1])
#create_feature_vector(root,classes[2])
#create_feature_vector(root,classes[3])



















# =============================================================================
# def create_list(files,path):
#     text_files=[]
#     #working on train or test files
#     for file in train_files:
#         f = open("temp.txt", "w") 
#         content=open(path+"/"+file, encoding = "ISO-8859-1").read()
#         words = re.split('[^a-zA-Z]',content)
#         for w in words:
#             if len(w) > 1 and not w in stop_words:
#                 f.write(w+" ") 
#         #write on txt file temperory and append to list
#         temp_file = open("./temp.txt", encoding = "ISO-8859-1").read()
#         text_files.append(temp_file)
#     
#     return text_files
# 
# vectorizer = TfidfVectorizer(min_df=2,max_df=.5,ngram_range=(1,2),lowercase=True)
# 
# def create_csv(name):
#     csv_file = open("./csv/"+name+"_train.csv", "w")
#     return csv_file
# 
# for class_file in  classes:
#     p = os.path.join(root,class_file)
#     files= [name for name in os.listdir(p) if os.path.isfile(os.path.join(p, name))]
#     train_files,test_files = split_datafiles(files)
#     
#     train_process_list = create_list(train_files,p)
#     print("create list train :"+class_file)
#     test_process_list = create_list(test_files,p)
#     print("create list test :"+class_file)
#     
#     features_train = vectorizer.fit_transform(train_process_list)
#     features_test = vectorizer.fit_transform(test_process_list)
#     
#     train_csv = create_csv(class_file)
#     test_csv = create_csv(class_file)
#     
#     df_train = pd.DataFrame(features_train.todense(),columns=vectorizer.get_feature_names())
#     df_test = pd.DataFrame(features_test.todense(),columns=vectorizer.get_feature_names())
#     
#     print(class_file+" dataframe for train\n")
#     print(df_train)
#     print(class_file+" dataframe for test\n")
#     print(df_test)
#     print("\n===================================================================================\n")
# =============================================================================



#train_files,test_files = split_datafiles(files)
#print(len(train_files))
#stop_words = set(stopwords.words('english')) 
#for file in train_files:
#    f = open("temp.txt", "w")        
#    content=open("./classes/computer/"+file, encoding = "ISO-8859-1").read()
#    words = re.split('[^a-zA-Z]',content)
#    for w in words:
#        if len(w) > 1 and not w in stop_words:
#            f.write(w+" ") 
#    temp_file = open("./temp.txt", encoding = "ISO-8859-1").read()
#    lines.append(temp_file)
#
#print(len(lines))
#vectorizer = TfidfVectorizer(min_df=2,max_df=.5,ngram_range=(1,2),lowercase=True)
#features = vectorizer.fit_transform(lines)
#
#csv_file = open("computer_train.csv", "w")
#df = pd.DataFrame(features.todense(),columns=vectorizer.get_feature_names())
#df.to_csv(csv_file)

#TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False).fit_transform(documents)



