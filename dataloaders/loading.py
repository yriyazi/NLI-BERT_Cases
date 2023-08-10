import  pandas          as      pd

def transform_dtaframe(df:pd.DataFrame,
                       preprocess):
    df['INPUT_A'] = df['premise'].apply(preprocess.forward)
    df['INPUT_B'] = df['hypothesis'].apply(preprocess.forward)
    return df
