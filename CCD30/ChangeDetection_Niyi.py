#!/usr/bin/python
import argparse
import csv
import os
import shutil
import sys
import time
from dataclasses import dataclass
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
# import pyct
import pywt
from PIL import Image
from osgeo import gdal
from scipy import stats
from skimage import morphology
from skimage.restoration import inpaint
from osgeo import gdal, ogr, osr
from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import filters

import colorUtil

matplotlib.use('agg')

PaintNanSwitch = False  # turn on paintNaN; if false, replace NaNs with zeros


@dataclass(frozen=True)
class Georef:
    geo_transform: tuple
    projection: str


def readInputs():

    parser = argparse.ArgumentParser()

    parser.add_argument('PreImage', help='Pre-event image')
    parser.add_argument('PostImage', help='Post-event image')
    parser.add_argument('-a', '--adaptive', action='store_true',
                        help='Use a significance test of SSE to determine the optimal class number.')
    parser.add_argument('-c', '--classes', type=int, default=3,
                        help='Number of classes used, default=3')
    parser.add_argument('-d', '--dcom-level', type=int, default=4,
                        help='Number of decomposition steps for wavelet transform, default=4')
    parser.add_argument('-e', '--struct-elem', type=int, default=2,
                        help='Size of structuring element (Mathematical morphology), default=2')
    parser.add_argument('-s', '--sigma', type=float, default=1.5,
                        help='Noise standard deviation used for Non-Local Means filtering, default=1.5')
    parser.add_argument('-o', '--result-path', type=str, default=os.path.join(
        os.getcwd(), 'Results'), help='Directory for result files')
    parser.add_argument('-sym', '--symbology', type=str, default='scaled',
                        help='Select a symbology color schema: \'binary\' or \'scaled\'. If binary is selected, change classes will be combined into one for brightening and one for darkening.')
    parser.add_argument('--debug', action='store_true',
                        help='Create debug output')
    args = parser.parse_args()

    return args


def preprocess(path1, path2, sigma):
    print("Preprocessing")
    t0 = time.time()

    dataset1 = gdal.Open(path1)
    image1 = np.array(dataset1.ReadAsArray())

    dataset2 = gdal.Open(path2)
    image2 = np.array(dataset2.ReadAsArray())

    if dataset1.GetDriver().GetDescription() != 'GTiff' or dataset2.GetDriver().GetDescription() != 'GTiff':
        sys.exit('Images must be GeoTIFFs; Exiting')

    shape = image1.shape
    if shape != image2.shape:
        sys.exit('Images 1 and 2 have different sizes; Exiting')

    georef = Georef(dataset1.GetGeoTransform(), dataset1.GetProjection())
    if georef != Georef(dataset2.GetGeoTransform(), dataset2.GetProjection()):
        print('WARNING: Image 1 and 2 may have different georeferences')
        # sys.exit('Images 1 and 2 have different georeferences; Exiting')

    if np.array_equal(image1, image2):
        sys.exit('Images 1 and 2 are identical; Exiting')

    # replace any elements in image1 less than 0.0001 with 0.0001
    mask1 = ~np.isnan(image1)
    if np.all(mask1):     # if no nans are in image1
        image1[image1 < 0.0001] = 0.0001
    else:                 # if nans are in image1
        mask1[mask1] &= image1[mask1] < 0.0001
        image1[mask1] = 0.0001
    del mask1

    # replace any elements in image2 less than 0.0001 with 0.0001
    mask2 = ~np.isnan(image2)
    if np.all(mask2):     # if no nans are in image2
        image2[image2 < 0.0001] = 0.0001
    else:                 # if nans are in image2
        mask2[mask2] &= image2[mask2] < 0.0001
        image2[mask2] = 0.0001
    del mask2

    # image1 = nonLocalMeans(image1, sigma)
    # image2 = nonLocalMeans(image2, sigma)

    image1_dB = 10 * np.log10(image1)
    image2_dB = 10 * np.log10(image2)

    image1_dB = nonLocalMeans(image1_dB, sigma)
    image2_dB = nonLocalMeans(image2_dB, sigma)

    del image1, image2

    ratio_image = image2_dB - image1_dB

    # zero-pad image if needed to make it square and its side lengths an integer power of 2
    ypad = 0
    xpad = 0
    if shape[0] != shape[1] or shape[0] != int(2**np.ceil(np.log2(shape[0]))):
        SideLen = int(2**np.ceil(np.log2(np.max(shape))))
        ypad = int(np.floor((SideLen - shape[0]) / 2.))
        xpad = int(np.floor((SideLen - shape[1]) / 2.))
        imageZ = np.zeros((SideLen, SideLen), dtype=np.float32)
        imageZ[ypad: ypad + shape[0], xpad: xpad + shape[1]] = ratio_image
        ratio_image = imageZ.copy()
        del imageZ

    # Replaces infs with 2
    ratio_image[ratio_image == np.inf] = 2
    ratio_image[ratio_image == -np.inf] = 2

    # deal with NaNs by painting over them or by replacing them with zeros
    if PaintNanSwitch:
        ratio_image_fixed = paintNaN(ratio_image)
    else:
        ratio_image_fixed = ratio_image.copy()
        ratio_image_fixed[np.isnan(ratio_image_fixed)] = 0

    del ratio_image

    t1 = time.time()
    print('Total Preprocessing time: ' + str(t1 - t0))

    return ratio_image_fixed, georef, shape, ypad, xpad, image1_dB, image2_dB


def paintNaN(array):
    # Should be equivalent to the 3rd method in the MATLAB nanpaint function
    print("Painting NaNs")
    t0 = time.time()
    mask = np.zeros(array.shape)
    mask[np.isnan(array)] = 1
    fixed = inpaint.inpaint_biharmonic(array, mask)

    del mask

    t1 = time.time()

    print("Paint NaNs Time: " + str(t1 - t0))

    return fixed


def nonLocalMeans(unfilteredImg, sig):
    print('Performing Non Local Means denoising on an input image... ')
    denoise_fast = denoise_nl_means(
        unfilteredImg, h=0.6 * sig, sigma=sig, fast_mode=True)
    return denoise_fast


def swt_decomp(Filt, ShapeOrig, ypad, xpad, dcomLvl):
    ims = []
    # noinspection PyUnusedLocal
    for cA, (cH, cV, cD) in pywt.swt2(Filt, 'bior5.5', dcomLvl):
        ims.append(cA)
    wvFilt = np.stack(ims, axis=2)
    del ims

    # via Niyi's method
    wvFilt[:, :, dcomLvl - 1] = Filt.copy()
    del Filt

    # remove zero-padding, if needed
    if ShapeOrig[0] != ShapeOrig[1] or ShapeOrig[0] != int(2**np.ceil(np.log2(ShapeOrig[0]))):
        wvFilt = remove_padding(wvFilt, ShapeOrig, ypad, xpad)

    return wvFilt


def remove_padding(image, ShapeOrig, ypad, xpad):
    return image[ypad: ypad + ShapeOrig[0], xpad: xpad + ShapeOrig[1]]


def math_morphology(wvFilt, dcomLvl, struct):
    for lvl in range(dcomLvl):
        filteredImg = wvFilt[:, :, lvl].copy()

        elem = morphology.square(struct)

        erode = morphology.erosion(filteredImg, selem=elem)

        recon1 = morphology.reconstruction(
            erode, filteredImg).astype(np.float32)
        del erode, filteredImg

        dilate = morphology.dilation(recon1, selem=elem)
        del elem

        compRecon1 = complement(recon1)
        del recon1
        compDilate = complement(dilate)
        del dilate
        compRecon1[compRecon1 <=
                   compDilate] = compDilate[compRecon1 <= compDilate]
        recon2 = morphology.reconstruction(
            compDilate, compRecon1).astype(np.float32)
        del compDilate, compRecon1

        comp = complement(recon2)
        del recon2

        # save back into wvFilt (not the best idea, but conserving memory is key)
        wvFilt[:, :, lvl] = comp
        del comp

    return wvFilt


def complement(image):
    compImage = 1. - image
    return compImage


def scale(wvFilt):
    wvFilt = wvFilt - np.nanmin(wvFilt)
    wvFilt = (255 - np.spacing(1).astype(np.float32)) * \
        (wvFilt / np.nanmax(wvFilt))
    return wvFilt


def calc_SSE(A, pis, mus, vs):
    size = round(len(A) / 1000)
    observed, bins1 = np.histogram(A, bins=size,  density=True)

    model = np.zeros(bins1.shape)
    for p in range(len(pis)):
        variance = vs[p][0][0]
        weight = pis[p][0][0]
        mean = mus[p][0][0]
        gaussian = weight/(np.sqrt(2 * variance * np.pi)) * \
            np.exp(-(bins1 - mean)**2 / (2 * variance))
        model = model + gaussian

    # Calculate mean-square residual
    df = (model.size - (3 * pis.size))
    SSE = np.sum((model[0:-1] - observed)**2)

    return SSE, df


def get_classed_image(wvFilt, cl, dcomLvl):
    prod = np.ones((wvFilt.shape[0], wvFilt.shape[1], cl)).astype(np.float32)
    df = None
    for lvl in range(dcomLvl):
        pis, mus, vs = em_seg_opt(wvFilt[:, :, lvl], cl)
        c = distribution_2d(pis, mus, vs, wvFilt[:, :, lvl])
        if lvl == 0:
            base = wvFilt[:, :, 0].flatten()
            SSE, df = calc_SSE(base, pis, mus, vs)

        wvFilt_shape = wvFilt.shape
        posterior = c * np.tile(pis.reshape(1, 1, cl),
                                (wvFilt_shape[0], wvFilt_shape[1], 1))

        del c
        np.seterr(divide='ignore', invalid='ignore')
        prod = prod * (np.divide(posterior, np.tile(np.sum(posterior,
                                                           axis=2)[:, :, np.newaxis], (1, 1, cl))))  # conditional probabilities
        del posterior

    ch = prod.max(axis=2)  # map

    classed_image = np.zeros((prod.shape[0], prod.shape[1]), dtype=np.float32)
    for cn in range(cl):
        classed_image[prod[:, :, cn] == ch] = cn + 1

    return classed_image, SSE, df


def em_seg_opt(ima, k):
    # EM segmentation optimization

    Niter = 2000      # maximum number of allowed iterations
    Tol = 0.000001    # tolerance

    ima2 = ima.copy()
    max_ima = np.nanmax(ima)
    min_ima = np.nanmin(ima)
    Rima = max_ima - min_ima    # range of ima

    # change the NaNs to -1
    ima2[np.isnan(ima2)] = -1

    # Remove inf, -inf, and negatives; don't just turn them to -1
    ima2 = ima2[np.logical_and(np.logical_and(
        ima2 != np.inf, ima2 != -np.inf), ima2 >= 0)]

    # had to hard-wire the number of bins since max values of all images can vary
    (h, binEdges) = np.histogram(ima2, np.arange(256))
    h = h.astype(np.float32)

    del ima2
    del binEdges

    tri = np.array([1., 2., 3., 2., 1.], dtype=np.float32)
    h = np.convolve(h, tri.T)
    h = h[2:-2]
    h = h / h.sum()

    # turn any zero values in h to very small numbers
    h[h == 0] = np.spacing(1)

    pis = np.multiply(np.ones((k, 1), dtype=np.float32), 1. / k)
    mus = np.arange(1., k + 1., dtype=np.float32).reshape(k,
                                                          1) * Rima / (k + 1.) + min_ima
    vs = np.ones((k, 1), dtype=np.float32) * Rima

    pis, mus, vs = em_gmm_eins(h[:, None], pis, mus, vs, Tol, Niter)

    return pis, mus, vs


# TODO delete if not needed
def em_gmm_eins_orig(xs, pis, mus, sigmas, tol=0.000001, max_iter=2000):
    n, p = xs.shape
    k = len(pis)
    ll_old = 0
    for i in range(max_iter):
        # E-step
        ws = np.zeros((k, n))
        for j, (pi, mu, sigma) in enumerate(zip(pis, mus, sigmas)):
            ws[j, :] = pi * stats.multivariate_normal(mu, sigma).pdf(xs)
        ws /= ws.sum(0)

        # M-step
        pis = np.einsum('kn->k', ws) / n
        mus = np.einsum('kn,np -> kp', ws, xs) / ws.sum(1)[:, None]
        sigmas = np.einsum('kn,knp,knq -> kpq', ws,
                           xs - mus[:, None, :], xs - mus[:, None, :]) / ws.sum(axis=1)[:, None, None]

        # update complete log likelihood
        ll_new = 0
        for pi, mu, sigma in zip(pis, mus, sigmas):
            ll_new += pi * stats.multivariate_normal(mu, sigma).pdf(xs)
        ll_new = np.log(ll_new).sum()

        if np.abs(ll_new - ll_old) < tol:
            break

        ll_old = ll_new

    # noinspection PyUnboundLocalVariable
    return ll_new, pis, mus, sigmas


def em_gmm_eins(xs, pis, mus, vs, tol=0.000001, max_iter=2000):
    # Adapted from https://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html#vectorization-with-einstein-summation-notation

    n, p = xs.shape
    X = np.arange(1., n + 1)
    k = len(pis)
    prb = np.zeros((n, k))

    for j, (pi, mu, v) in enumerate(zip(pis, mus, vs)):
        prb[:, j] = pi * stats.multivariate_normal(mu, v).pdf(X)

    scal = np.sum(prb, axis=1) + np.spacing(1)

    ll_old = np.sum(xs.reshape(n, 1) * np.log(scal).reshape(n, 1))

    for i in range(max_iter):
        pp = np.tile(np.divide(xs.reshape(n, 1),
                               scal.reshape(n, 1)), (1, k)) * prb

        pis = np.sum(pp, axis=0).reshape(k, 1)
        mus = np.divide(np.sum(np.tile(X.reshape(n, 1), (1, k))
                               * pp, axis=0).reshape(k, 1), pis)

        vr = np.tile(X.reshape(n, 1), (1, k)) - \
            np.tile(mus.reshape(1, k), (n, 1))
        vs = np.divide(np.sum(vr * vr * pp, axis=0).reshape(k, 1), pis) + 0.001

        pis = pis + 0.001
        pis = pis / np.sum(pis)

        for j, (pi, mu, v) in enumerate(zip(pis, mus, vs)):
            prb[:, j] = pi * stats.multivariate_normal(mu, v).pdf(X)

        scal = np.sum(prb, axis=1) + np.spacing(1)
        ll_new = np.sum(xs.reshape(n, 1) * np.log(scal).reshape(n, 1))

        # return i
        if np.abs(ll_new - ll_old) < tol:
            break

        ll_old = ll_new

    # sort pis, mus, and vs, sort by increasing order of mus
    ind = np.argsort(mus, axis=0)
    pis = pis[ind]
    mus = mus[ind]
    vs = vs[ind]

    pis = pis.astype(np.float32)
    mus = mus.astype(np.float32)
    vs = vs.astype(np.float32)

    return pis, mus, vs


def distribution_2d(p, m, v, x):
    p2 = p.copy()
    m2 = m.copy()
    v2 = v.copy()
    x2 = x.copy()
    Nm = m2.size

    # apply formula for guassian probability density function on x, weighted by p
    # IE this produces the probability of the pixel belonging to each of the classes.

    if Nm == v2.size and Nm == p2.size:
        Sx = x2.shape

        d = np.tile(x2[:, :, np.newaxis], (1, 1, Nm)) - \
            np.tile(np.reshape(m2, (1, 1, Nm), 'F'), (Sx[0], Sx[1], 1))

        amp = 1 / np.sqrt(2 *
                          np.pi * np.reshape(v2, (Nm, 1), 'F'))

        y = np.tile(np.reshape(amp, (1, 1, Nm), 'F'), (Sx[0], Sx[1], 1)) * np.exp(
            -0.5 * d * d / np.tile(np.reshape(v2, (1, 1, Nm), 'F'), (Sx[0], Sx[1], 1)))
    else:
        y = np.empty((1, 1))
        print("Error in function 'distribution_2d':")
        print("  inputs (p, m, v) do not have the same number of elements.")

    return y


def write_geotiff(image, georef, outname, noData=None, dtype=gdal.GDT_Float32, colors='scaled', statistics=None):
    outdriver = gdal.GetDriverByName("GTiff")
    outdata = outdriver.Create(
        outname, image.shape[1], image.shape[0], 1, dtype)
    band = outdata.GetRasterBand(1)
    band.WriteArray(image)

    # Use an intuitive color scheme
    if colors and statistics is not None:

        classes = statistics.transpose()[0]
        medians = statistics.transpose()[3]
        values = np.unique(image)
        noData_median = statistics[noData - 1, 3]

        # Sort classes by their medians
        stats = [[classes[n], medians[n]] for n in range(len(classes))]
        stats = sorted(stats, key=lambda e: e[1])

        # Use only classes that actually have pixels attached to them
        stats = [cl for cl in stats if cl[0] in values]

        gdalColors = colorUtil.getSymbology(
            colors, stats, medians, values, noData_median, noData)

        band.SetRasterColorTable(gdalColors)
        band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    outdata.SetGeoTransform(georef.geo_transform)
    outdata.SetProjection(georef.projection)

    del outdata

    gdal.Warp(outname, outname, dstNodata=noData)

    # TODO NIYI-6: Include hyp3lib as a dependency, then uncomment this
    # line (and the import statement at the top of this file):
    # makeChangeBrowse(outname)


def write_image(image, outname):
    minVal = np.min(image)
    if minVal < 0:
        image = image + abs(np.min(image))
    image = (255 * (image / np.max(image))).astype(np.uint8)
    Image.fromarray(image).save(outname)


def get_KS_statistic(classed_image, ratio_image, classes):

    ratio_cl = []

    for ii in range(classes):
        classPopulation = ratio_image[classed_image == ii + 1].flatten()
        ratio_cl.append(classPopulation)


def write_histogram_files(classed_image, ratio_image, classes, output_dir):
    # make histogram that relates pixel values in ratio image to final classes

    # generate stacked histogram
    ratio_cl = []
    bins_n = 50
    classStatistics = np.zeros((classes, 6))
    for ii in range(classes):
        classPopulation = ratio_image[classed_image == ii + 1].flatten()
        weight = len(classPopulation) / len(ratio_image.flatten())
        ratio_cl.append(classPopulation)

        if weight == 0:
            print('This class does not exist.')
            continue
        classStatistics[ii] = [
            ii + 1,
            round(float(np.mean(classPopulation)), 2),
            round(float(stats.mode(np.round(classPopulation)).mode[0])),
            round(float(np.median(classPopulation)), 2),
            round(float(np.std(classPopulation)), 2),
            round(float(weight), 4)
        ]

    ratio_cl_len = [len(jj) for jj in ratio_cl]
    ratio_cl_numel = sum(ratio_cl_len)

    sig3 = 3 * np.std(ratio_image[classed_image > 0])
    mean1 = np.mean(ratio_image[classed_image > 0])
    x_bound = sig3 + abs(mean1)
    bins_range = (-x_bound, x_bound)

    # noinspection PyUnboundLocalVariable
    plt.figure(ii + 1)
    ax1 = plt.subplot(111)

    # for classPop in ratio_cl:
    hist_values1, bins1, patches1 = plt.hist(
        ratio_cl, bins=bins_n, range=bins_range, stacked=True, density=True)
    del patches1

    # save histogram values to csv file
    list_header = ['Bin Minimum', 'Bin Maximum']
    for ii in range(len(hist_values1)):
        list_header.append(f'Stacked Histogram Class {ii + 1}')
    with open(os.path.join(output_dir, 'StackedHistSummary_Data.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(list_header)
        writer.writerows(list(zip(bins1[0:-1], bins1[1:], *hist_values1)))
        f.close()
    del hist_values1, bins1, list_header, f

    # save Class Statistics
    list_header = ['Class', 'Mean (dB)', 'Mode',
                   'Median', 'Standard Deviation', 'Proportion']
    with open(os.path.join(output_dir, 'ClassStatistics.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(list_header)
        writer.writerows(classStatistics)
        f.close()

    # save histogram image
    ax1.set_xlim(bins_range)
    plt.title('Histogram of Class Assignments')
    plt.xlabel('Change in Pixel Intensity (dB)')
    plt.ylabel('Normalized Probability')
    hist_label = []
    for ii in range(classes):
        hist_label.append('Class {0}, ({1:.2f} %)'.format(
            ii + 1, 100 * float(ratio_cl_len[ii]) / ratio_cl_numel))
    ax1.legend(hist_label, framealpha=1)
    plt.savefig(os.path.join(output_dir, 'StackedHistSummary.png'))
    plt.close()
    del ratio_cl, ratio_cl_len, ratio_cl_numel, hist_label
    del ratio_image

    return classStatistics


def getNoChangeClass(statistics, ratio):
    statistics = np.array([stat for stat in statistics if stat[5] > .05])
    print('Selecting no change class that has median close to : ', np.median(ratio))
    statistics[:, 3] = np.abs(statistics[:, 3] - np.median(ratio))
    noChange = statistics[np.where(  # Find the class with the smallest difference to the total ratio median
        statistics[:, 3] == np.min(statistics[:, 3]))][0][0]
    print(f'No Change Class: {noChange}')
    # Return the class number (in this case classes start at 1)
    return noChange


def isSignificant(SSE1, SSE2, df):
    f_stat = ((SSE1 - SSE2) / (SSE2 / df))
    p = 1 - stats.f.cdf(f_stat, 1, df)
    return p < .01


def reclass(classed, statistics, noChange):
    def vecfunc(a):
        return statistics[int(a) - 1, 3] - statistics[int(noChange) - 1, 3]

    re = classed.copy()
    refunc = np.vectorize(vecfunc)
    re = refunc(classed)
    noChange = 2
    re = (re / np.abs(re))
    re[np.isnan(re)] = 0

    return re + noChange, noChange


def main():
    inputs = readInputs()
    if os.path.exists(inputs.result_path):
        assert os.path.isdir(inputs.result_path)
        print(f'Deleting existing results directory {inputs.result_path}')
        shutil.rmtree(inputs.result_path)

    os.mkdir(inputs.result_path)

    debug_path = os.path.join(inputs.result_path, 'debug')
    if inputs.debug:
        os.mkdir(debug_path)

    (ratioImg, georef, ShapeOrig, ypad, xpad, image1, image2) = preprocess(
        inputs.PreImage, inputs.PostImage, inputs.sigma)

    # Write filtered before and after images
    write_geotiff(remove_padding(image1, ShapeOrig, ypad, xpad), georef, os.path.join(
        inputs.result_path, 'before-filtered.tif'))

    write_geotiff(remove_padding(image2, ShapeOrig, ypad, xpad), georef, os.path.join(
        inputs.result_path, 'after-filtered.tif'))

    # Output Png of ratio image
    filtRatioImg = ratioImg

    write_image(remove_padding(filtRatioImg, ShapeOrig, ypad, xpad), os.path.join(
        inputs.result_path, 'filtered-ratio-image.png'))

    # filtRatioImg = 10 * np.log10(filtRatioImg)
    write_geotiff(remove_padding(filtRatioImg, ShapeOrig, ypad, xpad), georef, os.path.join(
        inputs.result_path, 'filtered-ratio-image.tif'))

    # Wavelet Decompositions
    print('Performing wavelet decompositions')
    wvFilt = swt_decomp(filtRatioImg, ShapeOrig, ypad, xpad, inputs.dcom_level)

    print('Performing math morphology stage')
    wvFilt = math_morphology(wvFilt, inputs.dcom_level, inputs.struct_elem)

    # Output a geotiff representation of the decomposed and morphologically filtered stack
    stdWvFilt = np.mean(wvFilt, axis=2)
    write_geotiff(stdWvFilt, georef, os.path.join(
        inputs.result_path, 'wvFilt-morph-stdev.tif'))

    wvFilt = scale(wvFilt)
    classed_image = None
    SSE = None
    classn = inputs.classes
    sse_list = []

    if inputs.adaptive:
        for cc in range(1, inputs.classes):  # Minimuze Sum Squared Error
            if cc >= 2:
                print(f'Trying with {cc} classes...')
                classed_image, SSE2, df = get_classed_image(
                    wvFilt, cc, inputs.dcom_level)
                print(f"SSE: {SSE2}")
                sse_list.append(SSE2)
                if not SSE:
                    SSE = SSE2
                if isSignificant(SSE, SSE2, df):
                    SSE = SSE2
                    classn = cc
                    print('significant improvement')
                print('\n')

    print(f'Using a total of {classn} classes.')
    classed_image, SSE, df = get_classed_image(
        wvFilt, classn, inputs.dcom_level)

    ratio_image_unpadded = remove_padding(ratioImg, ShapeOrig, ypad, xpad)
    statistics = write_histogram_files(classed_image, ratio_image_unpadded,
                                       classn, inputs.result_path)

    # Identify no-change class
    noChange = getNoChangeClass(statistics, ratio_image_unpadded)

    # Reconstruct 3 binary classes
    if inputs.symbology == 'binary':
        classed_image, noChange = reclass(classed_image, statistics, noChange)
        statistics = write_histogram_files(classed_image, ratio_image_unpadded,
                                           3, inputs.result_path)

    write_geotiff(classed_image, georef, os.path.join(
        inputs.result_path, 'ChangeDetectionGeo.tif'), noData=int(noChange), dtype=gdal.GDT_Byte, colors=inputs.symbology, statistics=statistics)


if __name__ == '__main__':
    main()
