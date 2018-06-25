from csv import writer
from os import getcwd, path, listdir


def reformat(x):
    pathPart = x.split('.')[0].split('_')
    return '050v/'+'_'.join([pathPart[1], pathPart[0]] + pathPart[2:]) + '.png'


if __name__ == '__main__':
    labelledDir = path.join(getcwd(), 'data/050v')

    imagesNames = listdir(labelledDir)
    img2transcr = {
        reformat(im): open(path.join(labelledDir, im)).read().split()[0]
        for im in imagesNames}

    with open(path.join(getcwd(), 'data/labelledData.csv'), 'w') as lbls:
        out = writer(lbls)
        for img, txt in img2transcr.items():
            out.writerow((img, txt))
