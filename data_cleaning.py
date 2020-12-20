from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from numpy import quantile, array
from glob import glob
from re import sub
from random import shuffle
import string
import csv
from time import time
# # the 5 lines below need to be run once to download relevant nltk content
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk import tokenize
from nltk.corpus import stopwords
from pandas import Series, DataFrame



class BaseTextCleaner:
    STOPWORDS = set(stopwords.words('english'))
    PUNCTUATION = {i for i in string.punctuation}
    
    @staticmethod     
    # @timer 
    def split_sentence_to_list_of_words(sentence):
        return tokenize.word_tokenize(sentence)
    
    @classmethod
    # @timer
    def split_article_to_lists_of_words(cls, article):    
        article_sentences = tokenize.sent_tokenize(article)
        article_words = [cls.split_sentence_to_list_of_words(sentence) for sentence in article_sentences]
        return article_words   
    
    @classmethod
    # @timer        
    def lemmatize_and_denoise_sentence(cls, sentence):
        cleaned_sentence = []
        lemmatizer = WordNetLemmatizer()
        for word, tag in pos_tag(sentence):
            word = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', word)
            word = sub(r"(@[A-Za-z0-9_]+)","", word)

            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            word = lemmatizer.lemmatize(word, pos)

            if len(word) > 0:
                if word not in cls.PUNCTUATION:
                    if word.lower() not in cls.STOPWORDS:
                        cleaned_sentence.append(word.lower())
        return cleaned_sentence
    
    @classmethod
    # @timer
    def lemmatize_and_denoise_article(cls, article):
        return [cls.lemmatize_and_denoise_sentence(sent) for sent in article]
    
    @staticmethod
    # @timer
    def sentence_to_sequences(sentence, tokenizer):
        return tokenizer.texts_to_sequences([sentence])   
    
    @classmethod
    # @timer
    def article_to_sequences(cls, article, tokenizer):
        return [cls.sentence_to_sequences(sent, tokenizer) for sent in article]
    
    @staticmethod
    # @timer
    def pad_sentence(sentence, padding_type, max_length, trunc_type):
        return pad_sequences(sentence, padding=padding_type, maxlen=max_length, truncating=trunc_type)
    
    @classmethod
    # @timer
    def pad_article(cls, article, padding_type, max_length, trunc_type):
        return [cls.pad_sentence(sent, padding_type=padding_type, max_length=max_length, trunc_type=trunc_type) for sent in article]




# @timer
class TrainTextCleaner(BaseTextCleaner):
    
    def __init__(self, train_data_files, trunc_type='post', max_length=100, padding_type='post', oov_token='<OOV>'):
        self.train_data_files = train_data_files
        self.trunc_type = trunc_type
        self.max_length = max_length
        self.padding_type = padding_type  
        self.tokenizer = Tokenizer(oov_token=oov_token) # instantiate tokenizer
        self.tokenizer.fit_on_texts(texts = self.get_clean_train_data_generator(self.train_data_files, sentence_only=True)) # train tokenizer on train sentences   
        
    def get_tokenizer(self):
        return self.tokenizer
    
            
    # @timer
    def get_clean_train_data_generator(self, files_list, index=None, as_sequences=False, sentence_only=False, sentiment_only=False, sentiment_as_one_hot=False):
        
        if index != None: #if index was passed, make sure it's a set to efficiently check for membership
            index=set(index)

        counter=0 # if counter is present in index, yield value
        try:
            for file in files_list:
                with open(file, 'r') as f:
                    for row in f:
                        sentence, sentiment = row.split('@')
                        sentiment=sentiment.replace('\n','')
                                
                        # cleaning
                        sentence = self.split_sentence_to_list_of_words(sentence)
                        sentence = self.lemmatize_and_denoise_sentence(sentence)
                        if as_sequences == True:
                            sentence = ' '.join(sentence) # joining words back together because that's how the sequencing function accepts data
                            sentence = self.sentence_to_sequences(sentence, self.tokenizer)
                            sentence = self.pad_sentence(sentence, self.padding_type, self.max_length, self.trunc_type)

                        if sentiment_as_one_hot == True and sentence_only == False:
                            sentiment = array([[1,0,0]]) if sentiment=='positive' else array([[0,1,0]]) if sentiment=='neutral' else array([[0,0,1]])
                        
                        # yielding
                        if index: # allows to split the generated values into train and test (outputs a value only if it was assigned to the specified subset)
                            if counter in index:
                                counter+=1
                                if sentence_only:
                                    yield sentence
                                elif sentiment_only:
                                    yield sentiment
                                else:
                                    yield sentence, sentiment     
                        elif not index:
                            if sentence_only:
                                yield sentence
                            elif sentiment_only:
                                yield sentiment  
                            else:
                                yield sentence, sentiment
        except StopIteration:
            pass
         
         
         
            
class ArticleTextCleaner(BaseTextCleaner):
    
    def __init__(self, article_data_files, train_text_cleaner=None, trunc_type='post', max_length=100, padding_type='post', tokenizer=None):
        self.article_data_files = article_data_files
        if train_text_cleaner:
            self.train_text_cleaner = train_text_cleaner
            self.max_length = self.train_text_cleaner.max_length
            self.trunc_type = self.train_text_cleaner.trunc_type
            self.padding_type = self.train_text_cleaner.padding_type
            self.tokenizer = self.train_text_cleaner.tokenizer
        elif not train_text_cleaner:
            assert tokenizer != None
            self.tokenizer = tokenizer
            self.trunc_type = trunc_type
            self.max_length = max_length
            self.padding_type = padding_type
    
    def get_clean_article_data(self, as_sequences=False, yield_date_link=False):
        frame = DataFrame()
        for entry in self.get_clean_article_data_generator(as_sequences=as_sequences, yield_date_link=yield_date_link):
            data = dict(text = entry[0], date = entry[1], link = entry[2])
            frame = frame.append(DataFrame(data), ignore_index=True)
        return frame
                
            
    # @timer
    def get_clean_article_data_generator(self, as_sequences=False, yield_date_link=False):
        '''Generator that receives raw rows from file and outputs cleaned row data as lists of sequences
        '''
        try:
            for row in get_raw_article_generator(self.article_data_files):
                row_text = '. '.join([row[1], row[2], row[3]])
                date = row[4]
                link = row[5]
                row_sentences = self.split_article_to_lists_of_words(row_text)
                row_sequences = self.lemmatize_and_denoise_article(row_sentences)
                
                if as_sequences == True:
                    row_sentences = [' '.join(sent) for sent in row_sequences] # join words back into sentence after cleaning (sequencing works on sentence strings)
                    row_sequences = self.article_to_sequences(row_sentences, self.tokenizer)
                    row_sequences = self.pad_article(row_sequences, self.padding_type, self.max_length, self.trunc_type)
                
                if yield_date_link == True:
                    yield row_sequences, date, link
                else:
                    yield row_sequences
        except StopIteration:
            pass   
        
        
             

def timer(func):
    def wrapper(*args, **kwargs):
        print(f'Began   {func.__name__}')
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f'Finished   {func.__name__}')
        print(f'{round(end-start,2)} seconds\n')
        return result
    return wrapper


# @timer
def get_data_length(files):
    count = 0
    for file in files:
        with open(file) as f:
            for _ in f:
                count +=1
    return count


# @timer  
def train_test_split_index_for_generators(files, test_size=0.2):
    data_length = get_data_length(files)
    index = [i for i in range(data_length)]
    shuffle(index)
    
    cutoff = int(data_length*test_size)
    
    train_index = index[cutoff:]
    test_index = index[:cutoff]
    
    train_index = set(train_index)
    test_index = set(test_index)
    
    return train_index, test_index
  
    
# @timer
def get_csv_file_locations(folder):
    return glob(f'{folder}\\*.csv')


# @timer
def get_raw_train_data_generator(files_list):
    try:
        for file in files_list:
            with open(file, 'r') as f:
                for row in f:
                    sentence, sentiment = row.split('@')
                    sentiment=sentiment.replace('\n','')
                    yield sentence, sentiment
    except StopIteration:
        pass 


# @timer
def get_raw_article_generator(files_list, skip_header=True):
    try:
        for file in files_list:
            with open(file, 'r', encoding="utf8") as f:
                csv_reader = csv.reader(f, delimiter=',')
                if skip_header == True:
                    next(csv_reader) # skip header
                for row in csv_reader:
                    yield row
    except StopIteration:
        pass 
    
    
# def get_infinite_raw_train_data_generator(files_list):
#     while True:   
#         for i in get_raw_train_data_generator(files_list=files_list):
#             yield i
    
    
# def get_infinite_raw_article_generator(files_list, skip_header=True):
#     while True:
#         for i in get_raw_article_generator(files_list=files_list, skip_header=skip_header):
#             yield i
       
       
       
# @timer 
def remove_outliers(data, fraction=0.05):
    upper_frac = 1-fraction/2
    lower_frac = fraction/2
            
    perc_upper = quantile(data, upper_frac)
    perc_lower = quantile(data, lower_frac)

    data[data > perc_upper] = perc_upper
    data[data < perc_lower] = perc_lower
    
    return data
            
          
          
          
          
            
            
            
            
 
            
            
  




