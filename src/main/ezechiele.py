from cv2 import imshow, waitKey, destroyAllWindows
import ezechieleData
from src.lib.createMinacesLittera import createLetter


if __name__ == '__main__':

    ccToTokens = ezechieleData.mapping
    phrase = ezechieleData.verse
    littera = createLetter(ccToTokens, phrase)

    imshow('lectera minaces', littera)
    waitKey(0)
    destroyAllWindows()

    # if not path.exists(ezechieleMinace):
    #    imwrite(ezechieleMinace, toWhitePaper)
