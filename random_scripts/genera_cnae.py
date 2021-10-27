import pandas as pd

def fixer(code:str):
    if len(code) == 2:
        code = code + '00'
    elif len(code) == 3:
        code = code + '0'
    else:
        code = code
    return code


cnae = 'C:\\..\\cnae2009.xls'

df = pd.read_excel(cnae)

df['len'] = df['CODINTEGR'].apply(lambda x: len(x))
df['aux'] = df['CODINTEGR'].apply(lambda x: x[0])

dc = df.loc[df['len'] == 1, ['aux', 'TITULO_CNAE2009']]

dft = pd.merge(df, dc, on = 'aux', how = 'left')
dft['COD_CNAE2009'] = dft['COD_CNAE2009'].apply(fixer)
# dft = dft.loc[dft['len'] == 5, :]
dft['COD_CNAE2009'] = dft['COD_CNAE2009'].astype(str)
dft = dft.drop_duplicates('COD_CNAE2009')
dft[['COD_CNAE2009', 'TITULO_CNAE2009_y']].to_csv('processed_cnae_codes.csv', sep = '|', index = False)


