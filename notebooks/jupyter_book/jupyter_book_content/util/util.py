from datetime import datetime
import os
from pathlib import Path
import re
from typing import List, Union, Tuple

import ciso8601
import geopandas as gpd
import opensarlab_lib as osl
from osgeo import gdal, ogr, osr
gdal.UseExceptions()
from pyproj import Transformer
import rasterio
from shapely.geometry import Polygon
import shapely.wkt


def datetime_from_product_name(product_name: Union[str, os.PathLike]) -> Union[str, None]:
    """
    Takes: a string or posix path containing an ISO 8601 date time string

    Returns: a ISO 8601 date time string or None
    """
    regex = r"[0-9]{8}T[0-9]{6}"
    results = re.search(regex, str(product_name))
    if results:
        return results.group(0)
    else:
        return None


def get_datetimes(product_paths: List[Union[str, os.PathLike]]) -> List[datetime]:
    """
    Takes: a list of string paths or posix paths containing an ISO 8601 date time strings

    Returns: a list of datetime objects parsed from the input paths
    """
    product_paths = [Path(p) for p in product_paths]
    dates = []
    for pth in product_paths:
        date_str = datetime_from_product_name(pth.stem)
        dates.append(ciso8601.parse_datetime(date_str))
    return dates
    

def date_from_product_name(product_name: Union[str, os.PathLike]) -> Union[str, None]:
    """
    Takes: a string or posix path containing a date string in format %Y%m%d

    Returns: a a date string in format %Y%m%d or None
    """
    regex = r"[0-9]{8}"
    results = re.search(regex, str(product_name))
    if results:
        return results.group(0)
    else:
        return None


def get_dates(product_paths: List[Union[str, os.PathLike]]) -> List[datetime.date]:
    """
    Takes: a list of string paths or posix paths containing a date string in format %Y%m%d

    Returns: a list of datetime objects parsed from the input paths
    """
    product_paths = [Path(p) for p in product_paths]
    dates = []
    for pth in product_paths:
        date_str = date_from_product_name(pth.stem)
        dates.append(datetime.strptime(date_str, '%Y%m%d'))                            
    return dates


def get_epsg(geotiff_path: Union[str, os.PathLike]) -> str:
    """
    Takes: A string path or posix path to a GeoTiff

    Returns: The string EPSG of the Geotiff
    """
    ds = gdal.Open(str(geotiff_path))
    proj = ds.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    srs.AutoIdentifyEPSG()
    return srs.GetAuthorityCode(None)
    

def get_geotiff_bbox(geotiff_path: Union[str, os.PathLike], dst_epsg: str=None) -> Polygon:
    with rasterio.open(geotiff_path) as dataset:
        bounds = dataset.bounds
        min_x, min_y = (bounds.left, bounds.bottom)
        max_x, max_y = (bounds.right, bounds.top)
        
    if dst_epsg:
        srs_crs = dataset.crs
        transformer = Transformer.from_crs(srs_crs, f'EPSG:{str(dst_epsg)}', always_xy=True)
        min_x, min_y = transformer.transform(bounds.left, bounds.bottom)
        max_x, max_y = transformer.transform(bounds.right, bounds.top)

    return Polygon([
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y)
    ])


def get_valid_wkt() -> Tuple[str, Polygon]:
    """
    Prompts user for WKT

    Returns: WKT string, Shapely Polygon from WKT
    """
    while True:
        try:
            wkt = input("Please enter your WKT: ")
            shapely_geom = shapely.wkt.loads(wkt)
            
            if not gpd.GeoSeries([shapely_geom]).is_valid[0]:
                print('Invalid geometry detected. Please enter a valid WKT.')
                continue
            
            return wkt, shapely_geom
        except Exception as e:
            print(f'Error: {e}. Please enter a valid WKT.')

def check_within_bounds(wkt_shapely_geom: Polygon, gdf: gpd.GeoDataFrame) -> bool:
    """
    wkt_shapely_geom: A shapely Polygon describing a subset AOI
    gdf: a geopandas.GeoDataFrame containing geometries for each dataset to subset to wkt_shapely_geom

    returns: True if wkt_shapely_geom is contained within all geometries in the GeoDataFrame, else False
    """
    return gdf['geometry'].apply(lambda geom: wkt_shapely_geom.within(geom)).all()

def save_shapefile(
    ogr_geom: ogr.Geometry, 
    epsg: Union[str, int], 
    dst_path: Union[str, os.PathLike]=Path.cwd()/f'shape_{datetime.strftime(datetime.now(), "%Y%m%dT%H%M%S")}.shp'
):
    """
    Writes a shapefile from an ogr geometry in a given projection
    
    ogr_geom: An ogr geometry
    epsg: the EPSG projection to apply to the shapefile
    dst_path: (optional) shapefile destination path
    """
    epsg = int(epsg)
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(str(dst_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    layer = ds.CreateLayer('', srs, ogr.wkbPolygon)
    defn = layer.GetLayerDefn()

    feat = ogr.Feature(defn)
    feat.SetGeometry(ogr_geom)
    
    layer.CreateFeature(feat)
    feat = geom = None

    ds = layer = feat = geom = None
