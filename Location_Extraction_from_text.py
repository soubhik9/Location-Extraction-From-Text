# -*- coding: utf-8 -*- 




import nltk 
from nltk.collocations import * 
from nltk import * 
from geotext import GeoText 

from nltk.corpus import stopwords 
import re 
import io 

from nltk.stem import WordNetLemmatizer 


bigram_measures = nltk.collocations.BigramAssocMeasures() 
stop_words = set(stopwords.words('english')) 

filename = "Report.txt" 
with open(filename,encoding ="utf8") as my_file: 
    text = my_file.read() 




#Tokenizing the words and Tagging them according to Parts-of-Speech 
tokens = nltk.wordpunct_tokenize(text) 
posTagged=pos_tag(tokens) 
#print(posTagged) 
tokens1=[] 

#Removing stopwords 
for w in tokens: 
        if w not in stop_words: 
            tokens1.append(w) 
            
posTagged1=pos_tag(tokens1)    
        
#Advanced Filtration in order to accumulate the meaningful data as per requirement and eliminate the noisy data # 
all_wrd = [wrd.lower() for (wrd,tags) in posTagged if tags not in ('NNP','NNPS')] 

#Extracting Named Entity and their action 
noun_vrb= [(wrd.lower(),tags) for (wrd,tags) in posTagged1 if tags in ('NNP','NNPS', 'VBZ','VBD')] 

#Implementing Bi-gram 
finder2=BigramCollocationFinder.from_words((wrd,tags) for (wrd,tags) in noun_vrb) 
best_bigrams2=finder2.nbest(bigram_measures.raw_freq, 10000) 


# This part can extract some useful information of the company e.g. locations (both countries and cities) of the company/branches, headquarter of the company, 
print("<<<Location/Branch Related Information>>>") 
places = GeoText(text) 
city=[city.lower() for city in places.cities] 
country=[c.lower() for c in places.countries] 

#Extracting sentences which contains City name 
cities=[city for city in places.cities] 
 
for x,y in best_bigrams2: 
   if  x[0] not in city and y[0] in city: 
       print(x[0],"->",y[0]) 
   if x[0] not in city and y[0] in country: 
       print(x[0],"->",y[0]) 



