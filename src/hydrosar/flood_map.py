"""Generate flood depth map from surface water extent map.

Create a flood depth map from a surface water extent map and
a HAND image. The HAND image must be pixel-aligned (same extent and size) to
the water extent map, and the surface water extent map should be a byte GeoTIFF
indicating water (true), not water (false). Flood depth maps are estimated
using either a numerical, normalized median absolute deviation, logarithmic
or iterative approach.
"""

import argparse
import logging
import sys
import tempfile
import warnings
from pathlib import Path
from shutil import make_archive
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from osgeo import gdal
from scipy import ndimage, optimize, stats
from tqdm import tqdm

from asf_tools.aws import get_path_to_s3_file, upload_file_to_s3
from asf_tools.raster import read_as_masked_array, write_cog
from asf_tools.util import get_coordinates, get_epsg_code

log = logging.getLogger(__name__)


def get_pw_threshold(water_array: np.array) -> float:
    hist, bin_edges = np.histogram(water_array, density=True, bins=100)
    reverse_cdf = np.cumsum(np.flipud(hist)) * (bin_edges[1] - bin_edges[0])
    ths_orig = np.flipud(bin_edges)[np.searchsorted(np.array(reverse_cdf), 0.95)]
    return round(ths_orig) + 1


def get_waterbody(input_info: dict, threshold: Optional[float] = None) -> np.array:
    epsg = get_epsg_code(input_info)

    west, south, east, north = get_coordinates(input_info)
    width, height = input_info['size']

    data_dir = Path(__file__).parent / 'data'
    water_extent_vrt = data_dir / 'water_extent.vrt'  # All Perennial Flood Data

    with tempfile.NamedTemporaryFile() as water_extent_file:
        gdal.Warp(water_extent_file.name, str(water_extent_vrt), dstSRS=f'EPSG:{epsg}',
                  outputBounds=[west, south, east, north],
                  width=width, height=height, resampleAlg='nearest', format='GTiff')
        water_array = gdal.Open(water_extent_file.name, gdal.GA_ReadOnly).ReadAsArray()

    if threshold is None:
        threshold = get_pw_threshold(water_array)

    return water_array > threshold


def iterative(hand: np.array, extent: np.array, water_levels: np.array = np.arange(15),
              minimization_metric: str = 'ts'):
    def get_confusion_matrix(w):
        iterative_flood_extent = hand < w  # w=water level
        tp = np.nansum(np.logical_and(iterative_flood_extent == 1, extent == 1))  # true positive
        tn = np.nansum(np.logical_and(iterative_flood_extent == 0, extent == 0))  # true negative
        fp = np.nansum(np.logical_and(iterative_flood_extent == 1, extent == 0))  # False positive
        fn = np.nansum(np.logical_and(iterative_flood_extent == 0, extent == 1))  # False negative
        return tp, tn, fp, fn

    def _goal_ts(w):
        tp, _, fp, fn = get_confusion_matrix(w)
        return 1 - tp / (tp + fp + fn)  # threat score -- we will minimize goal func, hence `1 - threat_score`.

    def _goal_fmi(w):
        tp, _, fp, fn = get_confusion_matrix(w)
        return 1 - np.sqrt((tp/(tp+fp))*(tp/(tp+fn)))

    class MyBounds(object):
        def __init__(self, xmax=max(water_levels), xmin=min(water_levels)):
            self.xmax = np.array(xmax)
            self.xmin = np.array(xmin)

        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            return tmax and tmin

    bounds = MyBounds()
    MINIMIZATION_FUNCTION = {'fmi': _goal_fmi, 'ts': _goal_ts}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        opt_res = optimize.basinhopping(MINIMIZATION_FUNCTION[minimization_metric], np.mean(water_levels),
                                        niter=10000, niter_success=100, accept_test=bounds, stepsize=3)

    if opt_res.message[0] == 'success condition satisfied' \
            or opt_res.message[0] == 'requested number of basinhopping iterations completed successfully':
        return opt_res.x[0]
    else:
        return np.inf  # set as inf to mark unstable solution


def logstat(data: np.ndarray, func: Callable = np.nanstd) -> Union[np.ndarray, float]:
    """ Calculate a function in logarithmic scale and return in linear scale.
        INF values inside the data array are set to nan.

        Args:
            data: array of data
            func: statistical function to calculate in logarithmic scale
        Returns:
            statistic: statistic of data in linear scale
    """
    ld = np.log(data)
    ld[np.isinf(ld)] = np.nan
    st = func(ld)
    return np.exp(st)


def estimate_flood_depth(label: int, hand: np.ndarray, flood_labels: np.ndarray, estimator: str = 'iterative',
                         water_level_sigma: float = 3., iterative_bounds: Tuple[int, int] = (0, 15),
                         iterative_min_size: int = 0, minimization_metric: str = 'ts') -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')

        if estimator.lower() == "iterative":
            if (flood_labels == label).sum() < iterative_min_size:
                return np.nan

            water_levels = np.arange(*iterative_bounds)
            return iterative(hand, flood_labels == label,
                             water_levels=water_levels, minimization_metric=minimization_metric)

        if estimator.lower() == "nmad":
            hand_mean = np.nanmean(hand[flood_labels == label])
            hand_std = stats.median_abs_deviation(hand[flood_labels == label], scale='normal', nan_policy='omit')

        if estimator.lower() == "numpy":
            hand_mean = np.nanmean(hand[flood_labels == label])
            hand_std = np.nanstd(hand[flood_labels == label])

        elif estimator.lower() == "logstat":
            hand_mean = logstat(hand[flood_labels == label], func=np.nanmean)
            hand_std = logstat(hand[flood_labels == label])

        else:
            raise ValueError(f'Unknown flood depth estimator {estimator}')

    return hand_mean + water_level_sigma * hand_std


def make_flood_map(out_raster: Union[str, Path], vv_raster: Union[str, Path],
                   water_raster: Union[str, Path], hand_raster: Union[str, Path],
                   estimator: str = 'iterative',
                   water_level_sigma: float = 3.,
                   known_water_threshold: Optional[float] = None,
                   iterative_bounds: Tuple[int, int] = (0, 15),
                   iterative_min_size: int = 0,
                   minimization_metric: str = 'ts',
                   ):
    """Create a flood depth map from a surface water extent map.

    WARNING: This functionality is still under active development and the products
    created using this function are likely to change in the future.

    Create a flood depth map from a single surface water extent map and
    a HAND image. The HAND image must be pixel-aligned to the surface water extent map.
    The surface water extent map should be a byte GeoTIFF indicating water (true) and
    not water (false)

    Known perennial Global Surface-water data are produced under the Copernicus Programme (Pekel et al., 2016),
    and are included with surface-water detection maps when generating the flood depth product.

    Flood depth maps are estimated using one of the approaches:
    *Iterative: (Default) Basin hopping optimization method matches flooded areas to flood depth
    estimates given by the HAND layer. This is the most accurate method but also the
    most time-intensive.
    *Normalized Median Absolute Deviation (nmad): Uses a median operator to estimate
    the variation to increase robustness in the presence of outliers.
    *Logstat: Calculates the mean and standard deviation of HAND heights in the logarithmic
    domain to improve robustness for very non-Gaussian data distributions.
    *Numpy: Calculates statistics on a linear scale. Least robust to outliers and non-Gaussian
    distributions.

    Args:
        out_raster: Flood depth GeoTIFF to create
        vv_raster: Sentinel-1 RTC GeoTIFF, in power scale, with VV polarization
        water_raster: Surface water extent GeoTIFF
        hand_raster: Height Above Nearest Drainage (HAND) GeoTIFF aligned to the surface water extent raster
        estimator: Estimation approach for determining flood depth
        water_level_sigma: Max water height used in logstat, nmad, and numpy estimations
        known_water_threshold: Threshold for extracting the known water area in percent.
            If `None`, the threshold is calculated.
        iterative_bounds: Minimum and maximum bound on the flood depths calculated by the basin-hopping algorithm
            used in the iterative estimator
        iterative_min_size: Minimum size of a connected waterbody in pixels for calculating flood depths with the
            iterative estimator. Waterbodies smaller than this wll be skipped.
        minimization_metric: Evaluation method to minimize when using the iterative estimator.
            Options include a Fowlkes-Mallows index (fmi) or a threat score (ts).

    References:
        Jean-Francios Pekel, Andrew Cottam, Noel Gorelik, Alan S. Belward. 2016. <https://doi:10.1038/nature20584>
    """

    info = gdal.Info(str(water_raster), format='json')
    epsg = get_epsg_code(info)
    geotransform = info['geoTransform']
    hand_array = gdal.Open(str(hand_raster), gdal.GA_ReadOnly).ReadAsArray()

    log.info('Fetching perennial flood data.')
    known_water_mask = get_waterbody(info, threshold=known_water_threshold)
    write_cog(str(out_raster).replace('.tif', f'_{estimator}_PW.tif'), known_water_mask, transform=geotransform,
              epsg_code=epsg, dtype=gdal.GDT_Byte, nodata_value=False)

    water_map = gdal.Open(str(water_raster)).ReadAsArray()
    flood_mask = np.logical_or(water_map, known_water_mask)
    del water_map

    vv_array = read_as_masked_array(vv_raster)
    flood_mask[vv_array.mask] = False
    padding_mask = vv_array.mask
    del vv_array

    labeled_flood_mask, num_labels = ndimage.label(flood_mask)
    object_slices = ndimage.find_objects(labeled_flood_mask)
    log.info(f'Detected {num_labels} waterbodies...')
    if estimator.lower() == 'iterative':
        log.info(f'Skipping waterbodies less than {iterative_min_size} pixels.')

    flood_depth = np.zeros(flood_mask.shape)

    for ll in tqdm(range(1, num_labels)):  # Skip first, largest label.
        slices = object_slices[ll - 1]
        min0, max0 = slices[0].start, slices[0].stop
        min1, max1 = slices[1].start, slices[1].stop

        flood_window = labeled_flood_mask[min0:max0, min1:max1]
        hand_window = hand_array[min0:max0, min1:max1]

        water_height = estimate_flood_depth(
            ll, hand_window, flood_window, estimator=estimator, water_level_sigma=water_level_sigma,
            iterative_bounds=iterative_bounds, minimization_metric=minimization_metric,
            iterative_min_size=iterative_min_size,
        )

        flood_depth_window = flood_depth[min0:max0, min1:max1]
        flood_depth_window[flood_window == ll] = water_height - hand_window[flood_window == ll]

    flood_depth[flood_depth < 0] = 0

    nodata = -1
    flood_depth[padding_mask] = nodata

    floodmask_nodata = np.iinfo(np.uint8).max
    flood_mask_byte = flood_mask.astype(np.uint8)
    flood_mask_byte[padding_mask] = floodmask_nodata

    write_cog(str(out_raster).replace('.tif', f'_{estimator}_WaterDepth.tif'), flood_depth, transform=geotransform,
              epsg_code=epsg, dtype=gdal.GDT_Float64, nodata_value=nodata)
    write_cog(str(out_raster).replace('.tif', f'_{estimator}_FloodMask.tif'), flood_mask_byte, transform=geotransform,
              epsg_code=epsg, dtype=gdal.GDT_Byte, nodata_value=floodmask_nodata)

    flood_mask[known_water_mask] = False
    flood_depth[np.logical_not(flood_mask)] = 0
    flood_depth[padding_mask] = nodata
    write_cog(str(out_raster).replace('.tif', f'_{estimator}_FloodDepth.tif'), flood_depth, transform=geotransform,
              epsg_code=epsg, dtype=gdal.GDT_Float64, nodata_value=nodata)


def optional_str(value: str) -> Optional[str]:
    if value.lower() == 'none':
        return None
    return value


def optional_float(value: str) -> Optional[float]:
    if value.lower() == 'none':
        return None
    return float(value)


def _get_cli(interface: Literal['hyp3', 'main']) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    available_estimators = ['iterative', 'logstat', 'nmad', 'numpy']
    estimator_help = 'Flood depth estimation approach.'
    if interface == 'hyp3':
        parser.add_argument('--bucket')
        parser.add_argument('--bucket-prefix', default='')
        parser.add_argument('--wm-raster',
                            help='Water map GeoTIFF raster, with suffix `_WM.tif`.')
        available_estimators.append(None)
        estimator_help += ' If `None`, flood depth will not be calculated.'
    elif interface == 'main':
        parser.add_argument('out_raster',
                            help='File to which flood depth map will be saved.')
        parser.add_argument('vv_raster',
                            help='Sentinel-1 RTC GeoTIFF raster, in power scale, with VV polarization')
        parser.add_argument('water_extent_map',
                            help='HyP3-generated water extent raster file.')
        parser.add_argument('hand_raster',
                            help='Height Above Nearest Drainage (HAND) GeoTIFF aligned to the RTC rasters. '
                                 'If not specified, HAND data will be extracted from the GLO-30 HAND.')
    else:
        raise NotImplementedError(f'Unknown interface: {interface}')

    parser.add_argument('--estimator', type=optional_str, default='iterative', choices=available_estimators,
                        help=estimator_help)
    parser.add_argument('--water-level-sigma', type=float, default=3.,
                        help='Estimate max water height for each object.')
    parser.add_argument('--known-water-threshold', type=optional_float, default=None,
                        help='Threshold for extracting known water area in percent.'
                             ' If `None`, threshold will be calculated.')
    parser.add_argument('--minimization-metric', type=str, default='ts', choices=['fmi', 'ts'],
                        help='Evaluation method to minimize when using the iterative estimator. '
                             'Options include a Fowlkes-Mallows index (fmi) or a threat score (ts).')
    parser.add_argument('--iterative-min-size', type=int, default=0,
                        help='Minimum size of a connected waterbody in pixels for calculating flood depths with the '
                             'iterative estimator. Waterbodies smaller than this wll be skipped.')

    if interface == 'hyp3':
        parser.add_argument('--iterative-min', type=int, default=0,
                            help='Minimum bound on the flood depths calculated using the iterative estimator.')
        parser.add_argument('--iterative-max', type=int, default=15,
                            help='Maximum bound on the flood depths calculated using the iterative estimator.')
    elif interface == 'main':
        parser.add_argument('--iterative-bounds', type=int, nargs=2, default=[0, 15],
                            help='Minimum and maximum bound on the flood depths calculated using the iterative '
                                 'estimator.')
    else:
        raise NotImplementedError(f'Unknown interface: {interface}')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')

    return parser


def hyp3():
    parser = _get_cli(interface='hyp3')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    log.debug(' '.join(sys.argv))

    if args.estimator is None:
        # NOTE: HyP3's current step function implementation does not have a good way of conditionally
        #       running processing steps. This allows HyP3 to always run this step but exit immediately
        #       and do nothing if flood depth maps are not requested.
        log.info(f'{args.estimator} estimator provided; nothing to do!')
        sys.exit()

    if args.wm_raster:
        water_map_raster = args.wm_raster
    elif args.bucket:
        water_map_raster = get_path_to_s3_file(args.bucket, args.bucket_prefix, '_WM.tif')
        log.info(f'Found WM raster: {water_map_raster}')
    else:
        raise ValueError('Arguments --wm-raster or --bucket must be provided.')

    vv_raster = water_map_raster.replace('_WM.tif', '_VV.tif')
    hand_raster = water_map_raster.replace('_WM.tif', '_WM_HAND.tif')

    product_name = Path(water_map_raster).name.replace('_WM.tif', '_FM')
    product_dir = Path.cwd() / product_name
    product_dir.mkdir(exist_ok=True)

    flood_map_raster = product_dir / f'{product_name}.tif'

    make_flood_map(
        out_raster=flood_map_raster, vv_raster=vv_raster, water_raster=water_map_raster, hand_raster=hand_raster,
        estimator=args.estimator, water_level_sigma=args.water_level_sigma,
        known_water_threshold=args.known_water_threshold, iterative_bounds=(args.iterative_min, args.iterative_max),
        iterative_min_size=args.iterative_min_size, minimization_metric=args.minimization_metric,
    )

    log.info(f'Flood depth map created successfully: {flood_map_raster}')

    if args.bucket:
        output_zip = make_archive(base_name=product_name, format='zip', base_dir=product_name)
        upload_file_to_s3(Path(output_zip), args.bucket, args.bucket_prefix)
        for product_file in product_dir.iterdir():
            upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)


def main():
    parser = _get_cli(interface='main')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    log.debug(' '.join(sys.argv))

    make_flood_map(
        out_raster=args.out_raster, vv_raster=args.vv_raster, water_raster=args.water_extent_map,
        hand_raster=args.hand_raster, estimator=args.estimator, water_level_sigma=args.water_level_sigma,
        known_water_threshold=args.known_water_threshold, iterative_bounds=tuple(args.iterative_bounds),
        iterative_min_size=args.iterative_min_size, minimization_metric=args.minimization_metric,
    )

    log.info(f"Flood depth map created successfully: {args.out_raster}")
