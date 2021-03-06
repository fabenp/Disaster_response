import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages =pd.read_csv(messages_filepath) 
    categories_data = pd.read_csv(categories_filepath)
    df=pd.merge(messages,categories_data, on='id')
    return df

def clean_data(df):
  """Expand the category column and transform all the values to 0 or 1 .
Drop the categories column from the df dataframe.
Concatenate df and categories data frames. Drop the duplicates"""
    categories = df.categories.str.split(';', expand=True)
    row = categories.loc[0,:]
    category_colnames = row.apply(lambda x: x[: -2])
    categories.columns = category_colnames
    for col in categories:
        categories[col]=categories[col].apply(lambda x : str(x)[-1])
        categories[col]=categories[col].astype(int)     
       
    df=df.drop(['categories'], axis=1)
    df = pd.concat([df,categories],axis=1)
    df=df.drop_duplicates()
    return df


def save_data(df, database_filepath):
    """ Use pandas to_sql method and SQLAlchemy library to save the cleaned data."""
 
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('msg', engine, index=False)


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath= sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()