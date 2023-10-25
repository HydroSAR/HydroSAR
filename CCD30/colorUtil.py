from osgeo import gdal, ogr, osr
import numpy as np
from matplotlib import cm
import math


def colorToInt(color):
    red = color[0]
    green = color[1]
    blue = color[2]
    return (int(math.floor(255 * red)), int(math.floor(255 * green)), int(math.floor(blue * 255)))


def getSymbology(style, *args):
    if style == 'binary':
        return getBinaryColor(*args)

    if style == 'scaled':
        return getScaledColor(*args)

    else:
        print('Invalid symbology schema')


def getBinaryColor(stats, medians, values, noData_median, noData):

    gdalColors = gdal.ColorTable()

    negative = (201, 221, 240)
    positive = (240, 0, 0)

    for cl in stats:
        cn = cl[0]
        val = cl[1]
        if val != noData_median:
            if val > noData_median:
                color = positive
            elif val < noData_median:
                color = negative
            gdalColors.SetColorEntry(int(cn), color)

    gdalColors.SetColorEntry(int(noData), (0, 0, 0))

    return gdalColors


def getScaledColor(stats, medians, values, noData_median, noData):
    medians = np.array(stats).transpose()[1]
    gdalColors = gdal.ColorTable()

    maxMedian = abs(np.max(medians))
    minMedian = abs(np.min(medians))
    positiveChange = cm.get_cmap('seismic', 2 * maxMedian + 10)
    negativeChange = cm.get_cmap('Blues', 2 * minMedian + 10)
    for cl in stats:
        cn = cl[0]
        val = cl[1]
        if val != noData_median:
            if val > noData_median:
                color = colorToInt(positiveChange(
                    int(maxMedian + np.abs(val) + 10)))
            elif val < noData_median:
                color = colorToInt(negativeChange(
                    int(minMedian + np.abs(val) + 10)))
            gdalColors.SetColorEntry(int(cn), color)
    gdalColors.SetColorEntry(int(noData), (0, 0, 0))

    return gdalColors
