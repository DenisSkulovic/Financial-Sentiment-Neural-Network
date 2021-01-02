import data_cleaning as dcl
from model_classes import NNModelHandler
from pathlib import Path
from tqdm import tqdm
from pandas import read_csv, DataFrame, to_datetime
import matplotlib.pyplot as plt
tqdm.pandas()

import pickle



if __name__ == "__main__":

    # get train and article data
    cwd = str(Path.cwd())
    article_folder = cwd + r'\scraped_articles'
    article_csv_files = dcl.get_csv_file_locations(article_folder)
    train_data_files = [cwd + r'\FinancialPhraseBank-v1.0\Sentences_AllAgree.txt']
    embedding_file = cwd + r'\glove.6B.100d.txt'

    # instantiate train and predict data handlers
    print('Instantiating text cleaners')
    train_cleaner = dcl.TrainTextCleaner(train_data_files)
    article_cleaner = dcl.ArticleTextCleaner(article_csv_files, train_text_cleaner=train_cleaner)

    # instantiate, fit and save/load best neural network
    # print('Loading model')
    print('Training & saving the model')
    model_filepath = str(Path.cwd()) + r'\best_NN_model.hdf5'
    m_handler = NNModelHandler(train_cleaner, article_cleaner, embedding_file, save_model_to_location=model_filepath) #, load_model_from_location=model_filepath
    
    print('Saving tokenizer')
    with open("tokenizer.pickle", 'wb') as handle:
        pickle.dump(train_cleaner.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print('Preparing article data for prediction')
    # pred_data_df = m_handler.pred_data_cleaner.get_clean_article_data(as_sequences=True, yield_date_link=True)
    # # pred_data_df.to_csv(f'{cwd}/pred_data_df.csv')
    # # pred_data_df = read_csv(f'{cwd}/pred_data_df.csv', index_col=0)
    
    
    # # I tried implementing multiprocessing to speed up prediction by using all cores, however research showed that Keras is not threading-safe and many Keras objects are not pickle-able. 
    # print('Predicting article sentiment')
    # pred_data_df['raw_predictions'] = pred_data_df['text'].progress_apply(lambda x: m_handler.model.predict(x))
    
    # def func(row):
    #     if (row[0][0] > 0.33) & (row[0][0] > row[0][2]): # positive > 0.33 and positive > negative
    #         return 1
    #     elif (row[0][2] > 0.33) & (row[0][2] > row[0][0]): # negative > 0.33 and negative > positive
    #         return -1
    #     else:
    #         return 0
    
    # # preparing data for plotting    
    # pred_data_df['predictions'] = pred_data_df['raw_predictions'].apply(func)
    # pred_data_df['date'] = to_datetime(pred_data_df['date'], errors='coerce')
    # pred_data_df = pred_data_df.dropna()
    # pred_data_df = pred_data_df[['date','predictions']]
    # pred_data_df = pred_data_df.groupby(['date']).mean()
    # pred_data_df = pred_data_df.resample('W').mean()
    # pred_data_df = pred_data_df.ewm(alpha=0.045).mean()
    
    # # plot the sentiment predictions
    # plt.figure(figsize=(10,6))
    # y1 = pred_data_df['predictions'][20:] # cutting off starting entries as they are distorted due to exponential averaging
    # y2 = 0
    # x = y1.index

    # plt.fill_between(x, y1, y2, where=(y1 <= y2), color='red', alpha=0.25)
    # plt.fill_between(x, y1, y2, where=(y1 > y2), color='green', alpha=0.25)
    
    # plt.show()