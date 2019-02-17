from config import color_words, images2ColorsBBxesJSON, transcriptedWords_holesFree, isolatedTokensTranscription_txt
from os import path
from json import load
from numpy import zeros, flip
import cv2
from base64 import b64encode
from src.utils.imageProcessing import extractComponent


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


def isolatedTokensInstancesAndTranscription(lateralHoles=False):
    """
    Isolates each token in the word "pars" in "040v/1203_168_31_64.png".
    For each token T in a given word exists an image representing a mask of T on the original image, where T is white and every other
    pixel is black.
    :param lateralHoles: boolean. Consider words where there are no missing transcriptions at the ends.
    :return: dict of list. Key = image name, value = [transcription, [base64 representation of each token]]

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

    # output
    isolatedTranscrOut = []

    with open(images2ColorsBBxesJSON, 'r') as icb, open(transcriptedWords_holesFree, "r") as hf:
        colEtks = load(icb)
        innerHolesFree = load(hf)

        for im, ctks in colEtks.items():
            # hasLateralHoles = "^" in innerHolesFree[im] or "~" in innerHolesFree[im]
            if im in innerHolesFree and (not ("^" in innerHolesFree[im] or "~" in innerHolesFree[im]) or lateralHoles):
                colors = ctks["col"]
                tokens = ctks["tks"]

                #
                #   transcription
                #
                transcription = innerHolesFree[im]
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

                # original image
                imPath = path.join(color_words, im)
                assert path.exists(imPath)
                image = cv2.imread(imPath)
                # encoding of the isolated tokens
                isolated = []

                for idx, bbxt in enumerate(tokens):
                    bbx, t = bbxt
                    #
                    #   image
                    #
                    tImage = zeros(image.shape[:2], dtype="uint8")
                    fromX, toX, fromY, toY = bbx[-4:]
                    # colors
                    _t = t
                    if t not in colors:
                        if t.isupper():
                            _t = t.lower()
                        elif len(set(t)) == 1:
                            _t = t[0]
                        elif t in ('us', 'ue'):
                            _t = 'semicolon'
                    colorsT = flip(colors[_t], axis=1)

                    # token mask
                    tMask = extractComponent(image, colorsT, fromX, toX, fromY, toY)
                    tImage[fromY:toY, fromX:toX] = tMask
                    # encoding
                    buffer = cv2.imencode(".png", tImage)[1]
                    encoding = b64encode(buffer)

                    # token cleaning
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
                del image
                # imageName: transcription, encoding isolated image
                isolatedTranscrOut.append((im, transcr, isolated))

        return isolatedTranscrOut


if __name__ == "__main__":
    isolatedTkTranscription = isolatedTokensInstancesAndTranscription()

    with open(isolatedTokensTranscription_txt, "w") as outTxt:
        for entry in isolatedTkTranscription:
            outTxt.write(str(entry))
            outTxt.write('\n')

    print("\n#### Isolated tokens and transcription written to ", isolatedTokensTranscription_txt)
