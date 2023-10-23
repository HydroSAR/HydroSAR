"""Generate surface water maps from Sentinel-1 RTC products

Create a surface water extent map from a dual-pol Sentinel-1 RTC product and
a HAND image. The HAND image must be pixel-aligned (same extent and size) to
the RTC images. The water extent maps are created using an adaptive Expectation
Maximization thresholding approach and refined using Fuzzy Logic.
"""

import argparse
import logging
import sys
from pathlib import Path
from shutil import make_archive
from typing import Literal, Optional, Tuple, Union

import numpy as np
import skfuzzy as fuzz
from osgeo import gdal
from skimage import measure

from asf_tools.aws import get_path_to_s3_file, upload_file_to_s3
from asf_tools.hydrosar.hand.prepare import prepare_hand_for_raster
from asf_tools.hydrosar.threshold import expectation_maximization_threshold as em_threshold
from asf_tools.raster import read_as_masked_array, write_cog
from asf_tools.tile import tile_array, untile_array
from asf_tools.util import get_epsg_code

log = logging.getLogger(__name__)


def mean_of_subtiles(tiles: np.ndarray) -> np.ndarray:
    sub_tile_shape = (tiles.shape[1] // 2, tiles.shape[2] // 2)
    sub_tiles_mean = np.zeros((tiles.shape[0], 4))
    for ii, tile in enumerate(tiles):
        sub_tiles = tile_array(tile.filled(0), tile_shape=sub_tile_shape)
        sub_tiles_mean[ii, :] = sub_tiles.mean(axis=(1, 2))
    return sub_tiles_mean


def select_hand_tiles(tiles: Union[np.ndarray, np.ma.MaskedArray],
                      hand_threshold: float, hand_fraction: float) -> np.ndarray:
    if np.allclose(tiles, 0.0):
        raise ValueError(f'All pixels in scene have a HAND value of {0.0} (all water); '
                         f'scene is not a good candidate for water mapping.')

    tile_indexes = np.arange(tiles.shape[0])

    tiles = np.ma.masked_greater_equal(tiles, hand_threshold)
    percent_valid_pixels = np.sum(~tiles.mask, axis=(1, 2)) / (tiles.shape[1] * tiles.shape[2])

    return tile_indexes[percent_valid_pixels > hand_fraction]


def select_backscatter_tiles(backscatter_tiles: np.ndarray, hand_candidates: np.ndarray) -> np.ndarray:
    tile_indexes = np.arange(backscatter_tiles.shape[0])

    sub_tile_means = mean_of_subtiles(backscatter_tiles)
    sub_tile_means_std = sub_tile_means.std(axis=1)
    tile_medians = np.ma.median(backscatter_tiles, axis=(1, 2))
    tile_variance = sub_tile_means_std / tile_medians

    low_mean_threshold = np.ma.median(tile_medians[hand_candidates])
    low_mean_candidates = tile_indexes[tile_medians < low_mean_threshold]

    potential_candidates = np.intersect1d(hand_candidates, low_mean_candidates)

    for variance_threshold in np.nanpercentile(tile_variance.filled(np.nan), np.arange(5, 96)[::-1]):
        variance_candidates = tile_indexes[tile_variance > variance_threshold]
        selected = np.intersect1d(variance_candidates, potential_candidates)
        sort_index = np.argsort(sub_tile_means_std[selected])[::-1]
        if len(selected) >= 5:
            return selected[sort_index][:5]
    return np.array([])


def determine_em_threshold(tiles: np.ndarray, scaling: float) -> float:
    thresholds = []
    for ii in range(tiles.shape[0]):
        test_tile = (np.around(tiles[ii, :, :] * scaling)).astype(int)
        thresholds.append(em_threshold(test_tile) / scaling)

    return np.median(np.sort(thresholds)[:4])


def calculate_slope_magnitude(array: np.ndarray, pixel_size) -> np.ndarray:
    dx, dy = np.gradient(array)
    magnitude = np.sqrt(dx ** 2, dy ** 2) / pixel_size
    slope = np.arctan(magnitude) / np.pi * 180.
    return slope


def determine_membership_limits(
        array: np.ndarray, mask_percentile: float = 90., std_range: float = 3.0) -> Tuple[float, float]:
    array = np.ma.masked_values(array, 0.)
    array = np.ma.masked_greater(array, np.nanpercentile(array.filled(np.nan), mask_percentile))
    lower_limit = np.ma.median(array)
    upper_limit = lower_limit + std_range * array.std() + 5.0
    return lower_limit, upper_limit


def min_max_membership(array: np.ndarray, lower_limit: float, upper_limit: float, resolution: float) -> np.ndarray:
    possible_values = np.arange(array.min(), array.max(), resolution)
    activation = fuzz.zmf(possible_values, lower_limit, upper_limit)
    membership = fuzz.interp_membership(possible_values, activation, array)
    return membership


def segment_area_membership(segments: np.ndarray, min_area: int = 3, max_area: int = 10) -> np.ndarray:
    segment_areas = np.bincount(segments.ravel())

    possible_areas = np.arange(min_area, max_area + 1)
    activation = 1 - fuzz.zmf(possible_areas, min_area, max_area)

    segment_membership = np.zeros_like(segments)

    segments_above_threshold = np.squeeze((segment_areas > max_area).nonzero())
    segments_above_threshold = np.delete(segments_above_threshold, (segments_above_threshold == 0).nonzero())
    np.putmask(segment_membership, np.isin(segments, segments_above_threshold), 1)

    for area in possible_areas:
        mask = np.isin(segments, (segment_areas == area).nonzero())
        np.putmask(segment_membership, mask, fuzz.interp_membership(possible_areas, activation, area))
    return segment_membership


def remove_small_segments(segments: np.ndarray, min_area: int = 3) -> np.ndarray:
    valid_segments = segments != 0

    segment_areas = np.bincount(segments.ravel())
    segments_below_threshold = (segment_areas < min_area).nonzero()
    np.putmask(valid_segments, np.isin(segments, segments_below_threshold), False)

    return valid_segments


def format_raster_data(raster, padding_mask=None, nodata=np.iinfo(np.uint8).max):
    """
    Ensure raster data is uint8 and set the area outside the valid data to nodata
    """
    if padding_mask is None:
        array = read_as_masked_array(raster)
        padding_mask = array.mask
    raster = raster.astype(np.uint8)
    raster[padding_mask] = nodata

    return raster


def fuzzy_refinement(initial_map: np.ndarray, gaussian_array: np.ndarray, hand_array: np.ndarray, pixel_size: float,
                     gaussian_thresholds: Tuple[float, float], membership_threshold: float = 0.45) -> np.ndarray:
    water_map = np.ones_like(initial_map)

    water_segments = measure.label(initial_map, connectivity=2)
    water_segment_membership = segment_area_membership(water_segments)
    water_map &= ~np.isclose(water_segment_membership, 0.)

    gaussian_membership = min_max_membership(gaussian_array, gaussian_thresholds[0], gaussian_thresholds[1], 0.005)
    water_map &= ~np.isclose(gaussian_membership, 0.)

    hand_lower_limit, hand_upper_limit = determine_membership_limits(hand_array)
    hand_membership = min_max_membership(hand_array, hand_lower_limit, hand_upper_limit, 0.1)
    water_map &= ~np.isclose(hand_membership, 0.)

    hand_slopes = calculate_slope_magnitude(hand_array, pixel_size)
    slope_membership = min_max_membership(hand_slopes, 0., 15., 0.1)
    water_map &= ~np.isclose(slope_membership, 0.)

    water_map_weights = (gaussian_membership + hand_membership + slope_membership + water_segment_membership) / 4.
    water_map &= water_map_weights >= membership_threshold

    return water_map


def make_water_map(out_raster: Union[str, Path], vv_raster: Union[str, Path], vh_raster: Union[str, Path],
                   hand_raster: Optional[Union[str, Path]] = None, tile_shape: Tuple[int, int] = (100, 100),
                   max_vv_threshold: float = -15.5, max_vh_threshold: float = -23.0,
                   hand_threshold: float = 15., hand_fraction: float = 0.8, membership_threshold: float = 0.45):
    """Creates a surface water extent map from a Sentinel-1 RTC product

    Create a surface water extent map from a dual-pol Sentinel-1 RTC product and
    a HAND image. The HAND image must be pixel-aligned (same extent and size) to
    the RTC images. The water extent maps are created using an adaptive Expectation
    Maximization thresholding approach and refined with Fuzzy Logic.

    The input images are broken into a set of corresponding tiles with a shape of
    `tile_shape`, and a set of tiles are selected from the VH RTC
    image that contain water boundaries to determine an appropriate water threshold.
     Candidate tiles must meet these criteria:
    * `hand_fraction` of pixels within a tile must have HAND pixel values lower
      than `hand_threshold`
    * The median backscatter value for the tile must be lower than an average tiles'
      backscatter values
    * The tile must have a high variance -- high variance is considered initially to
      be a variance in the 95th percentile of the tile variances, but progressively
      relaxed to the 5th percentile if there not at least 5 candidate tiles.

    The 5 VH tiles with the highest variance are selected for thresholding and a
    water threshold value is determined using an Expectation Maximization approach.
    If there were not enough candidate tiles or the threshold is too high,
    `max_vh_threshold` and/or `max_vv_threshold` will be used instead.

    From the initial threshold-based water extent maps, Fuzzy Logic is used to remove
    spurious false detections and improve the water extent map quality. The fuzzy logic
    uses these indicators for the presence of water:
    * radar cross section in a pixel relative to the determined detection threshold
    * the height above nearest drainage (HAND)
    * the surface slope, which is derived from the HAND data
    * the size of the detected water feature

    For each indicator, a Z-shaped activation function is used to determine pixel membership.
    The membership maps are combined to form the final water extent map. Pixels classified
    as water pixels will:
    * have non-zero membership in all of the indicators, and
    * have an average membership above the `membership_threshold` value.

    Finally, the VV and VH water masks will be combined to include all water pixels
    from both masks, and the combined water map will be written to `out_raster`.

    Args:
        out_raster: Water map GeoTIFF to create
        vv_raster: Sentinel-1 RTC GeoTIFF, in power scale, with VV polarization
        vh_raster: Sentinel-1 RTC GeoTIFF, in power scale, with VH polarization
        hand_raster: Height Above Nearest Drainage (HAND) GeoTIFF aligned to the RTC rasters
        tile_shape: shape (height, width) in pixels to tile the image to
        max_vv_threshold: Maximum threshold value to use for `vv_raster` in decibels (db)
        max_vh_threshold:  Maximum threshold value to use for `vh_raster` in decibels (db)
        hand_threshold: The maximum height above nearest drainage in meters to consider
            a pixel valid
        hand_fraction: The minimum fraction of valid HAND pixels required in a tile for
            thresholding
        membership_threshold: The average membership to the fuzzy indicators required for a water pixel
    """
    if tile_shape[0] % 2 or tile_shape[1] % 2:
        raise ValueError(f'tile_shape {tile_shape} requires even values.')

    info = gdal.Info(str(vh_raster), format='json')

    out_transform = info['geoTransform']
    out_epsg = get_epsg_code(info)

    if hand_raster is None:
        hand_raster = str(out_raster).replace('.tif', '_HAND.tif')
        log.info(f'Extracting HAND data to: {hand_raster}')
        prepare_hand_for_raster(hand_raster, vh_raster)

    log.info(f'Determining HAND memberships from {hand_raster}')
    hand_array = read_as_masked_array(hand_raster)
    hand_tiles = tile_array(hand_array, tile_shape=tile_shape, pad_value=np.nan)

    hand_candidates = select_hand_tiles(hand_tiles, hand_threshold, hand_fraction)
    log.debug(f'Selected HAND tile candidates {hand_candidates}')

    selected_tiles = None
    nodata = np.iinfo(np.uint8).max
    water_extent_maps = []
    for max_db_threshold, raster, pol in ((max_vh_threshold, vh_raster, 'VH'), (max_vv_threshold, vv_raster, 'VV')):
        log.info(f'Creating initial {pol} water extent map from {raster}')
        array = read_as_masked_array(raster)
        padding_mask = array.mask
        tiles = tile_array(array, tile_shape=tile_shape, pad_value=0.)
        # Masking less than zero only necessary for old HyP3/GAMMA products which sometimes returned negative powers
        tiles = np.ma.masked_less_equal(tiles, 0.)
        if selected_tiles is None:
            selected_tiles = select_backscatter_tiles(tiles, hand_candidates)
            log.info(f'Selected tiles {selected_tiles} from {raster}')

        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)  # invalid value and divide by zero encountered in log10
            tiles = np.log10(tiles) + 30.  # linear power scale -> Gaussian scale optimized for thresholding
        max_gaussian_threshold = max_db_threshold / 10. + 30.  # db -> Gaussian scale optimized for thresholding
        if selected_tiles.size:
            scaling = 256 / (np.mean(tiles) + 3 * np.std(tiles))
            gaussian_threshold = determine_em_threshold(tiles[selected_tiles, :, :], scaling)
            threshold_db = 10. * (gaussian_threshold - 30.)
            log.info(f'Threshold determined to be {threshold_db} db')
            if gaussian_threshold > max_gaussian_threshold:
                log.warning(f'Threshold too high! Using maximum threshold {max_db_threshold} db')
                gaussian_threshold = max_gaussian_threshold
        else:
            log.warning(f'Tile selection did not converge! using default threshold {max_db_threshold} db')
            gaussian_threshold = max_gaussian_threshold

        gaussian_array = untile_array(tiles, array.shape)
        water_map = np.ma.masked_less_equal(gaussian_array, gaussian_threshold).mask
        water_map &= ~array.mask

        write_cog(str(out_raster).replace('.tif', f'_{pol}_initial.tif'),
                  format_raster_data(water_map, padding_mask, nodata),
                  transform=out_transform, epsg_code=out_epsg, dtype=gdal.GDT_Byte, nodata_value=nodata)

        log.info(f'Refining initial {pol} water extent map using Fuzzy Logic')
        array = np.ma.masked_where(~water_map, array)
        gaussian_lower_limit = np.log10(np.ma.median(array)) + 30.

        water_map = fuzzy_refinement(
            water_map, gaussian_array, hand_array, pixel_size=out_transform[1],
            gaussian_thresholds=(gaussian_lower_limit, gaussian_threshold), membership_threshold=membership_threshold
        )
        water_map &= ~array.mask

        write_cog(str(out_raster).replace('.tif', f'_{pol}_fuzzy.tif'),
                  format_raster_data(water_map, padding_mask, nodata),
                  transform=out_transform, epsg_code=out_epsg, dtype=gdal.GDT_Byte, nodata_value=nodata)

        water_extent_maps.append(water_map)

    log.info('Combining Fuzzy VH and VV extent map')
    combined_water_map = np.logical_or(*water_extent_maps)

    combined_segments = measure.label(combined_water_map, connectivity=2)
    combined_water_map = remove_small_segments(combined_segments)

    write_cog(out_raster, format_raster_data(combined_water_map, padding_mask, nodata), transform=out_transform,
              epsg_code=out_epsg, dtype=gdal.GDT_Byte, nodata_value=nodata)


def _get_cli(interface: Literal['hyp3', 'main']) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    if interface == 'hyp3':
        parser.add_argument('--bucket')
        parser.add_argument('--bucket-prefix', default='')
        parser.add_argument('--vv-raster',
                            help='Sentinel-1 RTC GeoTIFF raster, in power scale, with VV polarization.')
    elif interface == 'main':
        parser.add_argument('out_raster', help='Water map GeoTIFF to create')
        # FIXME: Decibel RTCs would be real nice.
        parser.add_argument('vv_raster',
                            help='Sentinel-1 RTC GeoTIFF raster, in power scale, with VV polarization')
        parser.add_argument('vh_raster',
                            help='Sentinel-1 RTC GeoTIFF raster, in power scale, with VH polarization')

        parser.add_argument('--hand-raster',
                            help='Height Above Nearest Drainage (HAND) GeoTIFF aligned to the RTC rasters. '
                                 'If not specified, HAND data will be extracted from the GLO-30 HAND.')
        parser.add_argument('--tile-shape', type=int, nargs=2, default=(100, 100),
                            help='image tiles will have this shape (height, width) in pixels')
    else:
        raise NotImplementedError(f'Unknown interface: {interface}')

    parser.add_argument('--max-vv-threshold', type=float, default=-15.5,
                        help='Maximum threshold value to use for `vv_raster` in decibels (db)')
    parser.add_argument('--max-vh-threshold', type=float, default=-23.0,
                        help='Maximum threshold value to use for `vh_raster` in decibels (db)')
    parser.add_argument('--hand-threshold', type=float, default=15.,
                        help='The maximum height above nearest drainage in meters to consider a pixel valid')
    parser.add_argument('--hand-fraction', type=float, default=0.8,
                        help='The minimum fraction of valid HAND pixels required in a tile for thresholding')
    parser.add_argument('--membership-threshold', type=float, default=0.45,
                        help='The average membership to the fuzzy indicators required for a water pixel')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')

    return parser


def hyp3():
    parser = _get_cli(interface='hyp3')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    log.debug(' '.join(sys.argv))

    if args.vv_raster:
        vv_raster = args.vv_raster
    elif args.bucket:
        vv_raster = get_path_to_s3_file(args.bucket, args.bucket_prefix, '_VV.tif')
        log.info(f'Found VV raster: {vv_raster}')
    else:
        raise ValueError('Arguments --vv-raster or --bucket must be provided.')

    vh_raster = vv_raster.replace('_VV.tif', '_VH.tif')

    product_name = Path(vv_raster).name.replace('_VV.tif', '_WM')
    product_dir = Path.cwd() / product_name
    product_dir.mkdir(exist_ok=True)

    water_map_raster = product_dir / f'{product_name}.tif'

    make_water_map(
        out_raster=water_map_raster, vv_raster=vv_raster, vh_raster=vh_raster,
        max_vv_threshold=args.max_vv_threshold, max_vh_threshold=args.max_vh_threshold,
        hand_threshold=args.hand_threshold, hand_fraction=args.hand_fraction,
        membership_threshold=args.membership_threshold
    )

    log.info(f'Water map created successfully: {water_map_raster}')

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

    make_water_map(args.out_raster, args.vv_raster, args.vh_raster, args.hand_raster, args.tile_shape,
                   args.max_vv_threshold, args.max_vh_threshold, args.hand_threshold, args.hand_fraction,
                   args.membership_threshold)

    log.info(f'Water map created successfully: {args.out_raster}')
