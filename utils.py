import pandas as pd

def read_seq(fileobj, isPositive):
    data = []
    for line in fileobj:
        data.append(line.strip('\n'))
    
    data_dna = []
    for i,j in enumerate(data):
        if i%2 == 1:
            data_dna.append(j)
    
    df = pd.DataFrame(data_dna, columns=['DNA'])
    df  = df['DNA'].str.split('',expand=True).iloc[:,1:-2:]
    df['Class'] = 1 if isPositive == True else 0
    return df, pd.DataFrame(data_dna, columns=['DNA'])
