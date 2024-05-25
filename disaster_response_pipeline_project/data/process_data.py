import sys
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    """
    Merges the messages and categories dataframes.
    Args:
    - messages_filepath: path to the messages CSV file
    - categories_filepath: path to the categories CSV file
    Returns:
    - merged_df: Pandas dataframe containing messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge dataframes
    merged_df = pd.merge(messages, categories, on="id") 

    return merged_df

def clean_data(df):
    """
    Removes duplicates in the dataframe and expands the categories column
    Args:
    df: merged dataframe
    Returns:
    cleaned_df: cleaned dataframe
    """
    categories = df['categories'].str.split(";", expand=True)

    # extract a list of new column names for categories
    row = categories.iloc[0].tolist()
    category_colnames = list(map(lambda cat:  cat.split('-')[0], row))

    # rename the columns in the categories dataframe
    categories.columns = category_colnames

    # convert category values to 1 or 0
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        categories[column] = categories[column].astype(int)
    
    # Replace categories column in df with new category columns
    df = df.drop(columns = ['categories'])
    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    df = df.drop_duplicates()

    # drop rows where the 'related' column contains values of 2
    df = df[df.related <= 1]

    return df


def save_data(df, database_filename):
    """
    Saves dataframe to a SQLLite database
    Args:
    - df: Pandas dataframe to save
    - database_filename: name for the database
    """
    from sqlalchemy import create_engine

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_response', engine, index=False)


def main():
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
