
def fill_df_with_mean(df):
    means = df.mean(axis='index', skipna=True)
    return df.fillna(means)
