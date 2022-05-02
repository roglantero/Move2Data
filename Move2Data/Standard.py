import pandas as pd
def standard_len(df, file, length, path):
    '''
    This function standardizes all sequences into a standard length, making ovelarpping windows.

    Args:
    df: Pandas DataFrame you wish to standardize.
    file: column name from df identifying the different series.
    lenght: standard length of the series you want to achieve.
    path: folder in which you wish to save all series csvs separately.

    Returns:
    This function returns a pandas DataFrame per file and saves it in the specified path.
    '''
    for files in df[file].unique():
        if len(df.loc[df[file] == files]) != length:
            count=0
            final=pd.DataFrame([],columns=df.columns)
            for i in range(0,len(df.loc[df[file] == files])):
                if i+length <= len(df.loc[df[file] == files]):
                    new=df.loc[df[file] == files][i:i+length]
                    new["File_group"]= files+"_Group"+str(count)
                    count= count+1
                    final=pd.concat([final, new])
                else:
                    continue
            final.to_csv(path+file+".csv")
        else:
            final=pd.DataFrame([],columns=df.columns)
            new=df.loc[df[file] == files]
            new["File_group"]= files+"_Group0"
            final=pd.concat([final, new])
            final.to_csv(path+file+".csv")
    return True