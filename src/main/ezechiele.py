from cv2 import imshow, waitKey, destroyAllWindows, imwrite
from os import path
import ezechieleData
from config import ezechieleMinace
from src.lib.createMinacesLittera import createLetter


def ezechiele():
    ccToTokens = ezechieleData.mapping
    phrase = ezechieleData.verse
    littera = createLetter(ccToTokens, phrase)

    imshow('lectera minaces', littera)
    waitKey(0)
    destroyAllWindows()

    if not path.exists(ezechieleMinace):
        print('writing image to ', ezechieleMinace)
        # imwrite(ezechieleMinace, littera)


if __name__ == '__main__':
    ezechiele()
