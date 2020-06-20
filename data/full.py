import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
corona=pd.read_csv('corona_fake.csv')

test['label']  = '2'

test=test.fillna(' ')
train=train.fillna(' ')
corona=corona.fillna(' ')
test['full']=test['title']+' '+test['author']+test['text']
train['full']=train['title']+' '+train['author']+train['text']

for i in range(len(corona['label'])):
  if corona['label'][i] == "Fake":
    corona['label'][i] = np.int64(1)
  else:
    corona['label'][i] = np.int64(0)

corona_label=np.array(corona['label'],dtype='int')
train_label=np.array(train['label'])

ps= PorterStemmer()
stopWords = set(stopwords.words('english'))
test['pro']=' '
for i in range(len(test['full'])):
    words = word_tokenize(test['full'][i])
    stem=[]
    for w in words:
      if w not in stopWords:
        stem.append(ps.stem(w))
    test.loc[i,'pro']=' '.join(stem)

ps= PorterStemmer()
stopWords = set(stopwords.words('english'))
corona['pro']=' '
for i in range(len(corona['text'])):
    words = word_tokenize(corona['text'][i])
    stem=[]
    for w in words:
      if w not in stopWords:
        stem.append(ps.stem(w))
    corona.loc[i,'pro']=' '.join(stem)

ps= PorterStemmer()
stopWords = set(stopwords.words('english'))
train['pro']=' '
for i in range(len(train['full'])):
    words = word_tokenize(train['full'][i])
    stem=[]
    for w in words:
      if w not in stopWords:
        stem.append(ps.stem(w))
    train.loc[i,'pro']=' '.join(stem)


tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3),smooth_idf=True)
tfidf_train = tfidf_vectorizer.fit(train['pro'].values)
tfidf_train_ans = train['label'].values
tfidf_train = tfidf_vectorizer.transform(train['pro'].values)
corona_train= tfidf_vectorizer.transform(corona['pro'].values)
#tfidf_test = tfidf_vectorizer.fit_transform(test['full'].values)

X_train, X_test, y_train, y_test = train_test_split(train['pro'].values, train_label, random_state=0)   

X_train=tfidf_vectorizer.transform(X_train)
X_test=tfidf_vectorizer.transform(X_test)

logreg = LogisticRegression(max_iter=200,random_state=0)
logreg.fit(X_train, y_train)
print('Accuracy of LogisticRegress classifier on training set: {:.3f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of LogisticRegress classifier on test set: {:.3f}'.format(logreg.score(X_test, y_test)))


print('Accuracy of LogisticRegress classifier on COVID dataset: {:.3f}'.format(logreg.score(corona_train, corona_label)))

article=input("Enter article")
words = word_tokenize(article)
stem=[]
for w in words:
  if w not in stopWords:
    stem.append(ps.stem(w))
finalart=' '.join(stem)
final=[]
final.append(finalart)
print(final)
test1=tfidf_vectorizer.transform(final)
if logreg.predict(test1)==0:
  print("News is reliable")
else:
    print("News is unreliable")
#print(logreg.predict(test1))
