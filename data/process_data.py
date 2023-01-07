import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function that loads the two dataframes and merges them
    :param messages_filepath: messages dataframe filepath
    :param categories_filepath: categories dataframe filepath
    :return: dataframe joined by id
    """
    # Importing the data
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    # # Joining tables by id
    df_merged = df_messages.merge(df_categories, how='inner', on='id')

    return df_merged


def clean_data(df):
    """
    Function that cleans the data, dropping unnecessary columns and
    duplicate values. In addition, it also creates new
    columns from the messed up category column
    :param df: joined dataframe
    :return: clean dataframe
    """

    # Dropping the original column
    df = df.drop(columns=['original', 'id'])

    # Separating column from dataframe
    categories = df.categories
    df = df.drop(columns='categories')

    # Dividing the values and checking the result
    categories = categories.str.split(';', expand=True)

    # Extracting column values and names
    new_values = {}
    new_columns_name = []
    for col in categories.columns:
        new_values[col] = [x[-1] for x in categories[col]]
        new_columns_name.append(np.unique([x[:-2] for x in categories[col]])[0])

    # Creating a dataframe with these values and checking
    new_categories = pd.DataFrame(new_values)
    new_categories.columns = new_columns_name

    # Fixing inconsistent data type and values
    new_categories = new_categories.astype('int')
    new_categories = new_categories.replace(2, 1)

    # Resetting the indexes and merging the dataframes
    df = df.reset_index().drop(columns='index')
    df = pd.concat([df, new_categories], axis=1)

    # # Dropping duplicated data
    df = df.drop_duplicates()
    df_cleaned = df.drop_duplicates(subset='message')

    return df_cleaned


def save_data(df, database_filename):
    """
    Save data in .db format
    :param df: dataframe
    :param database_filename: name to assign to the database
    :return: file in format .db
    """
    # exporting to database
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_categories', engine, index=False)


def main():
    """
    Perform the above steps, loading, clearing and saving the data
    :return: A cleaned database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
