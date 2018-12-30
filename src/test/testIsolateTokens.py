from json import load
from os import path
from config import color_words, images2ColorsBBxesJSON, transcriptedWords_holesFree
from src.utils.imageProcessing import extractComponent
from numpy import zeros, flip
import cv2
from base64 import b64encode
from pprint import pprint

convertBack = {
    # this applies to transcriptions
    '<b_stroke>': 'b',
    '<con>': 'con',
    '<curl>': 'us',
    '<d_stroke>': 'd',
    '<l_stroke>': 'l',
    '<nt>': 'et',
    '<per>': 'per',
    '<pro>': 'pro',
    '<prop>': 'prop',
    '<qui>': 'qui',
    '<rum>': 'rum',
    '<s_ending>': 's',
    '<s_mediana>': 's',
    '<semicolon>': ';',  # q -> ue, b->us, _-> ;
    # the following mapping applies to tokens
    '$': 'pro',
    '%': 'per',
    '&': ';',
    '(': 'rum',
    '/': 'prop',
    '1': 's',
    '2': 's',
    '3': 'd',
    '4': 'l',
    '5': 'b',
    '6': 'us',
    '7': 'qui',
    '8': 'con',
    '9': 'et'
}


def testIsolateTokens(showTks=True, printOut=True):
    """
    Isolates each token in the word "di" in "040v/1011_653_23_38.png". Each token is repesented as a bw mask on the
    original image, where background and the other tokens are black while the target token is white.
    Transcription of "di" (the string itself) is included in the output
    :return: dict of list.

    "par<s_mediana>" -> "pars"

    output = {
                "040v/1203_168_31_64.png": {
                                                [
                                                    "pars",
                                                    [
                                                        ("p", b'...'),
                                                        ("a", b'...'),
                                                        ("r", b'...'),
                                                        ("s", b'...')
                                                    ]
                                                ]
                                            }
             }
    """
    # 050v/440_1426_33_78.png c<l_stroke>ico<s_ending>
    """
        048v/1236_636_25_46.png  ->  an
        040v/1203_168_31_64.png  ->  par<s_mediana>  ->  pars
        050v/440_1426_33_78.png  ->  c<l_stroke>ico<s_ending>  ->  clicos
        051r/175_1125_42_79.png  ->  a<b_stroke><b_stroke>a<s_ending>  -> abbas
    """
    ims = ["048v/1236_636_25_46.png", "040v/1203_168_31_64.png", "050v/440_1426_33_78.png", "051r/175_1125_42_79.png"]
    imgNames = [path.join(color_words, im) for im in ims]

    for _im in imgNames:
        assert path.exists(_im)

    # for testing
    imsTrascr = ["an", "pars", "clicos", "a<b_stroke><b_stroke>a<s_ending>"]
    imsTrascrClean = ["an", "pars", "clicos", "abbas"]

    out = {imname: [] for imname in ims}

    with open(images2ColorsBBxesJSON, 'r') as icb:
        colEtks = load(icb)
        colEtks = {i: colEtks[i] for i in ims}

        for indIm, im in enumerate(ims):
            cols = colEtks[im]["col"]
            tokens = colEtks[im]["tks"]

            #
            #   transcription
            #
            transcription = imsTrascr[indIm]
            indx = 0
            transcr = transcription
            for c in range(transcription.count("<")):
                prevTranscr = transcr[:indx]
                transcr = transcr[indx:]
                # print("\nindx   ", indx, transcr)
                try:
                    startSubst = transcr.index('<')
                    endSubst = transcr.index('>') + 1
                    isSemicolon = transcr[startSubst:endSubst] == "semicolon"
                    if isSemicolon:
                        if startSubst >= 1 and transcr[startSubst - 1] == 'b':
                            subst = "us"
                        elif startSubst >= 1 and transcr[startSubst - 1] == 'q':
                            subst = "ue"
                        else:
                            subst = ';'
                    else:
                        subst = convertBack[transcr[startSubst:endSubst]]
                    #         taken                              | substitution | old 'tail'
                    transcr = prevTranscr + transcr[:startSubst] + subst + transcr[endSubst:]
                    # future edits start at indx
                    indx = startSubst + len(subst)
                except ValueError:
                    pass

            assert transcr == imsTrascrClean[indIm]
            print("##### {} ####\n".format(transcr))
            out[im].append(transcr)

            # original image
            image = cv2.imread(imgNames[indIm])

            # encoding of the isolated tokens
            isolated = []
            for idx, bbxt in enumerate(tokens):
                bbx, t = bbxt
                #
                #   image
                #
                tImage = zeros(image.shape[:2], dtype="uint8")
                fromX, toX, fromY, toY = bbx[-4:]
                colors = flip(cols[t], axis=1)
                tMask = extractComponent(image, colors, fromX, toX, fromY, toY)
                tImage[fromY:toY, fromX:toX] = tMask

                if showTks:
                    cv2.imshow(t, tImage)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                # encoding
                buffer = cv2.imencode(".png", tImage)[1]
                encoding = b64encode(buffer)
                # token
                if idx > 0 and t == "semicolon":
                    if tokens[idx - 1][1] == "b":
                        t = "us"
                    elif tokens[idx - 1][1] == "q":
                        t = "ue"
                    else:
                        t = ";"
                elif t in convertBack:
                    t = convertBack[t]
                # token to encoded image
                isolated.append((t, encoding))

            out[im].append(isolated)

        if printOut:
            pprint(out)


if __name__ == "__main__":
    testIsolateTokens(showTks=True, printOut=False)
