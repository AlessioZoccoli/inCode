from json import load
from config import wordsClean

AREA, WIDTH, HEIGHT, A_W = 0, 1, 2, 3

# meanValue:         area                  width            height        a/w
stats = {'a': (149.67737079779226, 18.74862017059709, 17.03211239337682, 8.018377632215357),
         'b': (199.6801242236025, 16.782608695652176, 25.928571428571427, 11.785501921953635),
         'c': (112.54645814167434, 13.615455381784729, 16.515179392824287, 8.206220489923078),
         'd': (206.24590163934425, 20.19924337957125, 26.284993694829762, 10.411815138843432),
         'e': (126.51014791881666, 14.092879256965944, 17.197454420364636, 9.099686161456708),
         'f': (205.30188679245282, 18.132075471698112, 26.628930817610062, 10.670471378291499),
         'g': (299.66487935656835, 32.94101876675603, 29.621983914209114, 9.037677497699057),
         'h': (206.91489361702128, 18.822695035460992, 28.98581560283688, 10.552064438017707),
         'i': (86.44489935175707, 10.534288638689867, 16.985670419651996, 8.345398515174411),
         'l': (155.2521572387344, 12.698945349952062, 27.548418024928093, 12.604589342122651),
         'm': (222.54172560113153, 26.613861386138613, 16.671852899575672, 8.155562220085768),
         'n': (163.19897377423032, 19.693842645381984, 16.93386545039909, 8.238428671202602),
         'o': (141.49185667752442, 14.644408251900108, 17.237242128121608, 9.68076590714673),
         'p': (230.6528, 19.784, 29.1008, 11.564377608833956),
         'q': (197.96491228070175, 17.65497076023392, 26.432748538011698, 11.15434894783022),
         'r': (112.17748562037798, 15.767460969597371, 17.281840591618735, 7.10437702800906),
         's': (185.0555041628122, 17.825161887141537, 27.730804810360777, 10.488641184740938),
         't': (121.23846153846154, 16.623076923076923, 17.42, 7.373002168526813),
         'u': (156.87190332326284, 18.749244712990937, 16.732930513595168, 8.32739100873178),
         'x': (155.89312977099237, 22.87786259541985, 20.610687022900763, 6.5901844326640155)
         }

DOUBLE = 2
SINGLE = 1
DELETE = 0


class Annotations:
    words = None

    def __init__(self, words=None):
        self.words = words


def mysort(l):
    print(len(l))
    return sorted(l, key=lambda x: x[A_W])  # a/aMean


def calcMetricsInMiddle(ch, lowerW=1.2, upperW=3., areaThrs=0.9):
    """
    Returns some basic metrics for each bbxes of a given character if this is neither in the starting
    position nor in the ending one. Note: only considering single 'standard' character (e.g. no doubles checking if
    charachter is 'pro' or 'um')
    :param ch: String. Charachter
    :param lowerW: float. Width lower bound
    :param upperW: float. Width upper bound
    :param areaThrs: float. Area lower thrashold/bound
    :return: tuple.
    (image:String, widthRatio:float, heightRatio:float, areaRatio:float, a2wRatio:float, a2wMeanRatio:float)

        image: path to image.
        widthRatio: width to area ratio.    # not == height
        heightRatio: height to area ratio.
        a2wRatio: area to width ratio
        a2wMeanRatio: a2wRatio to mean a2wRatio

    """
    if not Annotations.words:
        with open(wordsClean, 'r') as w:
            Annotations.words = load(w)

    my = []
    print('WIDTH lowerbound: {}, upperbound: {}, AREA thrashold: {},'.format(lowerW, upperW, areaThrs))

    for im, comps in Annotations.words.items():
        if len(comps) > 1:
            for el in comps[1:-1]:
                widthRatio = el[0][3] / stats[ch][1]
                areaRatio = el[0][2] / stats[ch][0]
                heightRatio = el[0][4] / stats[ch][2]
                a2wRatio = el[0][2] / el[0][3]  # area to width ratio
                a2wMeanRatio = a2wRatio / stats[ch][3]
                if el[1] == ch and lowerW <= widthRatio <= upperW and 0.9 <= heightRatio <= 1.1 and areaRatio > areaThrs:
                    my.append((im,
                               widthRatio,
                               heightRatio,
                               areaRatio,
                               a2wRatio,
                               a2wMeanRatio
                               ))
    return mysort(my)


def isDouble(bbx, upperAreaThrs, middleIs, lowerAreaThrs=1.1, upperBound=1000.0, widthThrs=0.0, AW2MeanThrs=0.0):
    areaRatio = bbx[0][2] / stats[bbx[1]][AREA]
    # DEFAULTs
    out = SINGLE
    a2wMeanRatio = AW2MeanThrs
    #   width / mean width
    widthRatio = widthThrs if widthThrs <= 0.0 else bbx[0][3]/stats[bbx[1]][WIDTH]
    # bbx area to width ratio divided by its mean value
    if AW2MeanThrs > 0.0:
        area2width = bbx[0][2]/bbx[0][3]
        a2wMeanRatio = area2width/stats[bbx[1]][A_W]
    # width / mean width

    if areaRatio < lowerAreaThrs:
        out = SINGLE
    elif lowerAreaThrs <= areaRatio < upperAreaThrs:
        out = middleIs
    elif upperAreaThrs <= areaRatio < upperBound:
        if a2wMeanRatio >= AW2MeanThrs and widthRatio >= widthThrs:
            out = DOUBLE
        else:
            out = middleIs
    return out


def isUpperCase(bbx, areaThrs, areaUpperBound=1000.0, widthThrs=0.0, heightThrs=0.0, a2wThrs=0.0):
    areaRatio = bbx[0][2] / stats[bbx[1]][AREA]
    widthRatio = widthThrs if widthThrs == 0.0 else bbx[0][3] / stats[bbx[1]][WIDTH]
    heightRatio = heightThrs if heightThrs == 0.0 else bbx[0][4] / stats[bbx[1]][HEIGHT]
    a2wMeanRatio = a2wThrs if a2wThrs == 0.0 else (bbx[0][2] / bbx[0][3]) / stats[bbx[1]][A_W]

    return areaThrs <= areaRatio < areaUpperBound and widthRatio >= widthThrs and heightRatio >= heightThrs and a2wMeanRatio >= a2wThrs


#               OUTLIERS
#
#  everything uppercase 050v/695_535_50_91.png, 040v/614_472_54_95.png
#
#  'b' 049v/1062_1331_44_127.png is uppercase
# 'f' '048v/149_1460_34_193.png', 060r_165_264_1424_1835/256_973_36_26.png -> 'fi'
#     050v/197_710_30_71.png, 057v_542_281_1394_1819/447_1086_37_48.png => 'fu'
#     059v/235_1405_34_77.png => 'fa' remove
# 'e'   doubles: 054r/1118_620_31_40.png, 057r/1036_915_30_42.png, 055v_631_241_1360_1839/776_1312_33_45.png,
#               048v/637_357_32_46.png, 057r/1036_915_30_42.png, 051v/322_373_24_41.png
#              '059r/972_184_30_55.png', '049v/433_341_26_38.png',
#
#      - upper: 047r/882_1507_48_56.png, 056v/1229_1061_34_62.png
#       'eg' ' 049v_586_258_1366_1821/171_1592_37_129.png' edit anncolor_clean so that 'g' -> 'eg'
#       'ex' '059r/1193_195_33_38.png', '057r/148_314_35_34.png', 055r/632_1614_49_108.png, 055r/674_992_50_140.png,
#            058v/177_1545_46_192.png, 051r/252_627_50_126.png, 058v/523_695_52_124.png, 051r/1119_894_56_99.png,
#            051r/700_473_38_114.png, 054r/360_950_37_163.png, 054r/360_950_37_163.png, 051r/1119_894_56_99.png,
#            059r/1193_195_33_38.png, 051r/252_627_50_126.png
#       'E....' '046v_586_244_1348_1778/482_759_53_99.png'
#       'ep' 049v/1100_831_45_63.png
#       'e____' 049v_586_258_1366_1821/174_356_34_166.png
#       'de' '047r/599_510_44_112.png', '047r/1121_1338_33_155.png'
#      -lower: 054r/144_947_32_32.png, 056r_178_258_1393_1827/811_1307_27_35.png, 054r/360_950_37_163.png
#      -final but in the beginning: 050r/571_1721_54_117.png
#
#
# 'l' double on [0] 050v/936_926_34_63.png
#       055v_631_241_1360_1839/455_1254_37_69.png
#      053r/875_374_42_102.png
#      '046r/651_1113_51_93.png'
#     '048v/1187_131_47_86.png'
#     '050v/936_926_34_63.png',
#     '055v_631_241_1360_1839/455_1254_37_69.png'
#       final = 050r/251_386_40_115.png
#
#
# 't'    'l'! = 056r_178_258_1393_1827/146_1079_34_58.png,
#        056r_178_258_1393_1827/881_758_25_40.png lowercase
#       'tt' = 056r_178_258_1393_1827/1076_691_34_55.png, 057v_542_281_1394_1819/714_416_46_167.png
#        't' only 1 t is annotated 059v/353_170_46_131.png
#        separated from 'a' and single 054r/1191_1563_35_76.png
#
# 'ex' 054r/55_1281_40_87.png
# 'xx' 050v/325_549_36_84.png, 048r/458_954_25_70.png
# 'a'  057v_542_281_1394_1819/385_1696_34_117.png => 'fa' remove or merge it
#       056v/106_349_36_92.png => 'pa' remove or merge it
#       056r_178_258_1393_1827/398_249_46_133.png => 'pa', unmergable
#       052r/689_1514_52_93.png => 'ap'
#       058v/1263_918_34_43.png => 'ca
#       upper case = 051r/540_1571_43_122.png, 046r/433_837_38_111.png
#       lower case =      '055r/549_1559_30_49.png', '060r_165_264_1424_1835/1114_367_32_48.png',
#                         '055r/1117_1100_40_177.png',
#                         '054r/671_738_25_69.png', '048r/734_1332_37_90.png',
#      first 'a' is a small stain (the following 2 are legit) 055r/756_1499_35_166.png
# 'h'  046r/347_1620_38_131.png   small but upper case, 059r/776_776_50_133.png also upper case,
#        long 058v/104_261_31_191.png
#      lower: 059v/1212_785_46_49.png, 055r/867_1616_47_67.png,
#             054r/360_950_37_163.png

# 'g'
#     the only upper ones: 047r/1206_1281_35_56.png      059r/815_175_39_57.png
#     delete  in 057r/620_1475_48_76.png is 'gl', 056v/265_1176_46_79.png 'gl'
#                '053r/1172_973_38_86.png'
#     delete 055v_631_241_1360_1839/122_422_40_65.png -> 'eg' (in part)
#     delete 040v/309_1629_47_33.png,  049v_586_258_1366_1821/666_1586_45_143.png = 'oga'
#     double g '057r/86_368_34_151.png' <- here delete the third g

# 'i' double at [0] 047r/626_566_28_44.png
#       'ü' not i -> 050v/389_928_30_86.png
# 'o' double 051v/1210_572_53_84.png
#       050r/68_885_31_30.png is an 'M'
#
#  'm' uppercase 050r/734_1101_54_215.png
#
# 'p' upper case: '046v_586_244_1348_1778/1222_1096_46_47.png', 056v/254_902_37_91.png, 047r/1073_727_48_205.png
#     056r_178_258_1393_1827/398_249_46_133.png => 'pa', keep it
#     056v/106_349_36_92.png => 'pa'
#     057v_542_281_1394_1819/507_859_46_124.png => 'pe' remove it
#     057r/515_1139_42_118.png => pro
#     lower case: 058v/137_980_49_95.png, 058v/137_980_49_95.png
#          'prop':  040v/1251_1137_44_83.png , 053r/396_932_37_123.png, 059v/300_682_40_77.png
#           'pp': 050v/408_485_45_50.png, 060v/1239_172_33_66.png
#           'pp':  050v/804_541_42_45.png, 051v/665_209_35_57.png, 057r/521_591_37_133,
#                   059r/296_1069_38_47.png, 051v/811_205_39_59.png, 058v/1072_314_46_167.png,
#                  040v/731_487_36_37.png, 055r/969_1336_35_73.png, 055r/365_1398_39_52.png, 059r/785_1125_37_48.png
#           big ornaments: 048r/394_128_50_133.png
#     'ep'  049v/1100_831_45_63.png, 049v_586_258_1366_1821/1109_798_45_63.png
# 'r'      double in [0] 040v/1143_111_35_175.png
# 's'      double in [0] 059v/675_725_37_58.png, 055v_631_241_1360_1839/645_1370_36_37.png, 046r/924_238_39_78.png,
#                        046r/165_1673_36_33.png, 055v_631_241_1360_1839/735_1368_36_60.png,
#                        057v_542_281_1394_1819/854_1361_35_67.png, 050r/228_436_44_132.png, 060v/1126_268_34_49.png
#          'si'= 058r/805_1644_47_139.png
#          SEE sLowerCase
#          's' Upper case? 059r/924_838_28_25.png, 060v/580_108_28_26.png
#           ornaments: 050v/71_1092_40_176.png
#          single and final = 048v/680_364_35_108.png 049v_586_258_1366_1821/1096_351_42_185.png
#          FInal but in the middle: 050v/1003_493_37_264.png, remove!
#          final : 058v/1010_257_46_166.png
#
#  'u'      uppercase thrasholds: areaRatio >= 1.2 widthRatio >= 1.4
#           050v/129_648_36_86.png ending stroke of Q, 049v/806_1220_35_130.png ending stroke of C
#           lowercase = '057r/1000_426_25_58.png', '055r/678_942_31_35.png',
#                        '050r/1050_1719_45_164.png'
#           uppercase (but below thrashold) =  '050v/612_1088_42_133.png', '050v/907_924_36_31.png',
#                                               '049v/967_224_39_107.png'
#           single = '056r_178_258_1393_1827/995_1580_35_111.png'
#           double = '057r/235_591_24_86.png'
#
#  'd'     areaRatio >= 1.41
#           053r/443_1706_34_131.png => 'nd' remove
#
#          upper: 056v/1065_836_33_58.png, 050v/998_433_28_82.png, 055r/893_219_37_66.png, 053r/716_212_39_105.png
#
# 'n' upper: '046r/1027_1446_40_108.png', '055v_631_241_1360_1839/178_749_36_118.png', '058v/991_757_37_134.png'
#     lower: '054r/945_1615_38_95.png'
#  'q'     no 040v/747_1473_38_40.png
#   'eran'   in 050r/977_882_37_174.png is not a real cc =>  e r an
#
#    051v/306_1520_34_164.png secon 'm' is a stein
#    056r_178_258_1393_1827/488_856_51_217.png    [['l'], ['i'], ['i', 's'], ['ss'], ['p', 'e'], ['n'], ['n'], ['us']]
#
#   'qui' first occurrence is a stain 050r/792_449_42_103.png
#

doublesRules = {
    'a': (lambda _: SINGLE),
    'b': (lambda chBBX: isDouble(chBBX, upperAreaThrs=1.67, middleIs=DELETE)),
    'c': (lambda chBBX: isDouble(chBBX, upperAreaThrs=1.4, middleIs=DELETE)),
    'd': (lambda chBBX: isDouble(chBBX, upperAreaThrs=1.87, middleIs=SINGLE)),
    'e': (lambda chBBX: DELETE if chBBX[0][2] / stats[chBBX[1]][AREA] >= 1.2 else SINGLE),
    'f': (lambda chBBX: isDouble(chBBX, upperAreaThrs=1.2, middleIs=SINGLE)),
    'g': (lambda chBBX: isDouble(chBBX, upperAreaThrs=1.79, middleIs=DELETE, lowerAreaThrs=1.3)),
    'h': (lambda _: SINGLE),
    'i': (lambda _: SINGLE),
    'l': (lambda chBBX: isDouble(chBBX, upperAreaThrs=1.5, middleIs=DELETE, lowerAreaThrs=1.1)),
    'm': (lambda chBBX: isDouble(chBBX, upperAreaThrs=1.8, middleIs=DELETE, lowerAreaThrs=1.2)),
    'n': (lambda chBBX: isDouble(chBBX, upperAreaThrs=1.8, middleIs=SINGLE)),
    'o': (lambda _: SINGLE),
    'p': (lambda chBBX: isDouble(chBBX, upperAreaThrs=1.54, middleIs=DELETE, lowerAreaThrs=1.2)),
    'q': (lambda _: SINGLE),
    'r': (lambda chBBX: isDouble(chBBX, upperAreaThrs=1.47, middleIs=DELETE, lowerAreaThrs=1.3)),
    's': (lambda chBBX: isDouble(chBBX, upperAreaThrs=1.24, middleIs=SINGLE)),
    't': (lambda chBBX: isDouble(chBBX, upperAreaThrs=1.59, middleIs=DELETE, lowerAreaThrs=1, AW2MeanThrs=0.99)),
    'u': (lambda chBBX: isDouble(chBBX, upperAreaThrs=1.73, middleIs=DELETE, lowerAreaThrs=1.24, widthThrs=1.85)),
    'x': (lambda chBBX: isDouble(chBBX, upperAreaThrs=2.84, middleIs=SINGLE))
}

fUpperCase = ('057v_542_281_1394_1819/1067_65_73_111.png',
              '060r_165_264_1424_1835/1249_799_54_25.png',
              '060r_165_264_1424_1835/1050_1417_50_107.png')

lUpperCase = ('056v/105_52_44_98.png', '051r/215_1623_42_73.png', '058v/1104_1539_48_92.png')

pLowerCase = ('058r/194_657_37_137.png', '049v_586_258_1366_1821/706_1072_53_155.png', '050v/444_1370_46_140.png',
              '052r/68_1634_34_65.png', '058r/194_657_37_137.png', '059v/1136_1173_47_82.png',
              '058v/137_980_49_95.png')

gUpperCase = ('047r/1206_1281_35_56.png', '059r/815_175_39_57.png')

uLowerCase = ('057r/1000_426_25_58.png', '055r/678_942_31_35.png', '050r/1050_1719_45_164.png')

sLowerCase = {'046r/105_1228_39_27.png',
              '046r/1180_894_49_118.png',
              '048v/427_947_52_60.png',
              '048v/928_1342_47_63.png',
              '049v/228_1229_39_34.png',
              '049v_586_258_1366_1821/726_247_34_26.png',
              '050r/1206_995_39_42.png',
              '050r/445_997_34_25.png',
              '050v/851_1583_56_167.png',
              '051v/1254_1297_35_28.png',
              '051v/603_305_37_67.png',
              '054r/1144_452_36_40.png',
              '054r/1200_944_40_58.png',
              '054r/197_1502_45_149.png',
              '054r/396_677_39_108.png',
              '054r/441_1393_45_63.png',
              '054r/496_1446_38_48.png',
              '054r/563_893_36_109.png',
              '054r/789_1616_49_28.png',
              '054r/799_1562_40_106.png',
              '054r/815_1057_36_48.png',
              '054r/862_1392_38_70.png',
              '054r/866_951_37_56.png',
              '054r/905_1562_49_155.png',
              '054r/966_343_36_97.png',
              '054r/969_674_37_179.png',
              '055v_631_241_1360_1839/653_1489_35_82.png',
              '055v_631_241_1360_1839/779_1255_48_152.png',
              '055v_631_241_1360_1839/918_1599_37_59.png',
              '055v_631_241_1360_1839/918_1599_37_59.png058v/195_361_35_25.png',
              '056r_178_258_1393_1827/1168_856_37_119.png',
              '056r_178_258_1393_1827/736_1358_37_28.png',
              '057r/1193_971_36_51.png',
              '057r/1241_1189_36_54.png',
              '057r/835_1133_39_38.png',
              '057v_542_281_1394_1819/879_419_32_23.png',
              '058r/268_1203_42_51.png',
              '058v/139_862_49_159.png',
              '058v/195_361_35_25.png',
              '058v/381_1711_37_95.png',
              '058v/603_757_35_49.png',
              '059r/986_1457_34_26.png',
              '059v/1217_1285_41_40.png'}

upperCaseRules = {
    'a': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.27, heightThrs=1.29)),
    'b': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.43)),
    'c': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.32, heightThrs=1.15, a2wThrs=1.0)),
    'd': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.41)),
    'e': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.5, heightThrs=1.2, a2wThrs=0.9)),
    'f': (lambda _: False),
    'g': (lambda _: False),
    'h': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.37, widthThrs=1.0)),
    'i': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.5, widthThrs=1.7)),
    'l': (lambda _: False),
    'm': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.38, heightThrs=1.2)),
    'n': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.4, heightThrs=1.4)),
    'o': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.4, areaUpperBound=4.0)),
    'p': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.1, widthThrs=1.3, a2wThrs=0.93)),
    'q': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.41)),
    'r': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.54)),
    's': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.1)),
    't': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.69)),
    'u': (lambda chBBX: isUpperCase(chBBX, areaThrs=1.2, widthThrs=1.4)),
    'x': (lambda _: False)
}
