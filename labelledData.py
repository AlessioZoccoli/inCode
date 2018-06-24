from csv import writer
from os import getcwd, path, listdir

if __name__ == '__main__':
    labelledDir = path.join(getcwd(), 'data/050v')

    imagesNames = listdir(labelledDir)
    img2transcr = {
        im.split('.')[0]: open(path.join(labelledDir, im)).read().split()[0]
        for im in imagesNames}

    with open(path.join(getcwd(), 'data/labelledData.csv'), 'w') as lbls:
        out = writer(lbls)
        for img, txt in img2transcr.items():
            out.writerow((img, txt))
