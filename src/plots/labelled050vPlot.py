from os import path, getcwd
import pandas as pd
from csv import reader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filePath = path.join(getcwd(), '../../data/testLabelledData.csv')

    headers = ('Image', 'Manual', 'Automatic', 'Distance')
    df = pd.read_csv(filePath, names=headers).iloc[1:]

    aggregate = df.groupby('Distance').agg('count')\
        .reset_index()\
        .apply(lambda x: x.astype(int)) \
        .sort_values('Distance')\

    plt.interactive(True)
    aggregate.plot(x='Distance', y='Image', kind='bar', legend=False)
