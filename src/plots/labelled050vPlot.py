from os import path, getcwd
import pandas as pd
from csv import reader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filePath = path.join(getcwd(), '../../data/testLabelledData.csv')

    headers = ('Image', 'Manual', 'Automatic', 'Distance')
    df = pd.read_csv(filePath, names=headers).iloc[1:]

    # aggregate = df.groupby('Distance', as_index=False).agg('count')
    aggregate = df.groupby('Distance').agg('count') \
        .reset_index()\
        .apply(lambda x: x.astype(int))

    """
    x = aggregate['Distance']
    y = aggregate['Image']
    xPositions = range(len(x))
    yPositions = range(0, max(y), 10)

    plt.bar(yPositions, y, align='center', alpha=0.5)
    plt.xticks(xPositions, x)
    plt.ylabel('Transcript')
    plt.xlabel('Distances')
    plt.title('Images to error transcription')

    plt.show()
    """

    aggregate.plot(x='Distance', y='Image')
