"""Prepare a Height Above Nearest Drainage (HAND) virtual raster (VRT) covering a given geometry"""
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Union

from osgeo import gdal, ogr
from rasterio.enums import Resampling
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from asf_tools import vector
from asf_tools.util import GDALConfigManager, get_epsg_code

HAND_GEOJSON = '/vsicurl/https://glo-30-hand.s3.amazonaws.com/v1/2021/glo-30-hand.geojson'

gdal.UseExceptions()
ogr.UseExceptions()


def prepare_hand_vrt(vrt: Union[str, Path], geometry: Union[ogr.Geometry, BaseGeometry]):
    """Prepare a HAND mosaic VRT covering a given geometry

    Prepare a Height Above Nearest Drainage (HAND) virtual raster (VRT) covering a given geometry.
    The Height Above Nearest Drainage (HAND) mosaic is assembled from the HAND tiles that intersect
    the geometry, using a HAND derived from the Copernicus GLO-30 DEM.

    Note: `asf_tools` does not currently support geometries that cross the antimeridian.

    Args:
        vrt: Path for the output VRT file
        geometry: Geometry in EPSG:4326 (lon/lat) projection for which to prepare a HAND mosaic

    """
    with GDALConfigManager(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
        if isinstance(geometry, BaseGeometry):
            geometry = ogr.CreateGeometryFromWkb(geometry.wkb)

        min_lon, max_lon, _, _ = geometry.GetEnvelope()
        if min_lon < -160. and max_lon > 160.:
            raise ValueError(f'asf_tools does not currently support geometries that cross the antimeridian: {geometry}')

        tile_features = vector.get_features(HAND_GEOJSON)
        if not vector.get_property_values_for_intersecting_features(geometry, tile_features):
            raise ValueError(f'Copernicus GLO-30 HAND does not intersect this geometry: {geometry}')

        hand_file_paths = vector.intersecting_feature_properties(geometry, tile_features, 'file_path')

        gdal.BuildVRT(str(vrt), hand_file_paths)


def prepare_hand_for_raster(hand_raster: Union[str, Path], source_raster: Union[str, Path],
                            resampling_method: str = 'lanczos'):
    """Create a HAND raster pixel-aligned to a source raster

    Args:
        hand_raster: Path for the output HAND raster
        source_raster: Path for the source raster
        resampling_method: Name of the resampling method to use. For available methods, see:
            https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r
    """
    info = gdal.Info(str(source_raster), format='json')

    hand_geometry = shape(info['wgs84Extent'])
    hand_bounds = [info['cornerCoordinates']['upperLeft'][0],
                   info['cornerCoordinates']['lowerRight'][1],
                   info['cornerCoordinates']['lowerRight'][0],
                   info['cornerCoordinates']['upperLeft'][1]]

    with NamedTemporaryFile(suffix='.vrt', delete=False) as hand_vrt:
        prepare_hand_vrt(hand_vrt.name, hand_geometry)
        gdal.Warp(str(hand_raster), hand_vrt.name, dstSRS=f'EPSG:{get_epsg_code(info)}',
                  outputBounds=hand_bounds, width=info['size'][0], height=info['size'][1],
                  resampleAlg=Resampling[resampling_method].value)
