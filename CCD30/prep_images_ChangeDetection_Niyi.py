#!/usr/bin/env python
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
###############################################################################
# prep_images_ChangeDetection_Niyi.py
#
# Project:  APD Niyi Change Detection
# Purpose:  Prepare Images as input for Niyi's Change Detection Algorithm
#            - crop to the region of overlap between images
#            - if needed, crop to the area of interest (AOI)
#            - if needed, decimate/resample output images
#
# Author:   Kenneth Arnoult
#           (adapted from run_sacd.py written by Tom Logan and Scott Arko)
###############################################################################
# Copyright (c) 2017, Alaska Satellite Facility
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
###############################################################################

import numpy as np
import sys
import os
import shutil
import scipy.ndimage as filt
import argparse
from osgeo import gdal, ogr, osr
# sys.path.append('/usr/bin')
# import saa_func_lib as saa
# import glob
# import gdal_merge as gm
# import re


# def usage():
#     """"
#     ***************************
#     Usage:
#        prep_images_ChangeDetection_Niyi.py <PreImage> <PostImage> [-aoi AOI_Shapefile] [-bb BoundingBox] [-r rows] [-c cols]
#
#     Inputs:
#        PreImage   Pre-event Image (in GeoTiff format)
#
#        PostImage  Post-event Image (in GeoTiff format)
#
#     Options:
#        -aoi AOI_Shapefile  Area Of Interest (in Shapefile format)
#
#        -bb BoundingBox     Crop to the bounding box
#                            type = bool, default = False
#
#        -r rows       Upper limit on allowable number of rows in output images
#                      type = int, default = 4096, value not to exceed 4096
#
#        -c cols       Upper limit on allowable number of columns in output images
#                      type = int, default = 4096, value not to exceed 4096
#     Outputs:
#        PreImage_Crop_geo.tif   Cropped Pre-event Image
#
#        PostImage_Crop_geo.tif  Cropped Post-event Image
#     ***************************
#     """
#     sys.exit(0)

# # old input options
# print('   -va vertAlign Vertical alignment of window within area of overlap')
# print('                 Number of rows from top')
# print('                 type = int, default will yield a centered window')
# print('   ')
# print('   -ha horzAlign Horizontal alignment of window within area of overlap')
# print('                 Number of columns from left')
# print('                 type = int, default will yield a centered window')
# print('   ')


def readInputs():

    parser = argparse.ArgumentParser()

    parser.add_argument('PreImage', help='Pre-event image (in GeoTiff format)')
    parser.add_argument(
        'PostImage', help='Post-event image (in GeoTiff format)')
    parser.add_argument('-aoi', '-AOI_Shapefile',
                        help='Area Of Interest (in Shapefile format)', default=None)
    parser.add_argument('-bb', '-BoundingBox',
                        action='store_true', help='Crop to the bounding box')
    parser.add_argument('-r', '-rows', type=int, default=4096,
                        help='Upper limit on allowable number of rows in output images')
    parser.add_argument('-c', '-cols', type=int, default=4096,
                        help='Upper limit on allowable number of columns in output images')
    # parser.add_argument('-h', '-help')
    # parser.add_argument('-va','-vertAlign',type=int,help='Vertical Alignment of window within area of overlap')
    # parser.add_argument('-ha','-horzAlign',type=int,help='Horizontal Alignment of window within area of overlap')

    args = parser.parse_args()

    return args


# def getOverlap(prefile,postfile):
#     (x1,y1,t1,p1) = saa.read_gdal_file_geo(saa.open_gdal_file(prefile))
#     (x2,y2,t2,p2) = saa.read_gdal_file_geo(saa.open_gdal_file(postfile))
#     ullon1 = t1[0]
#     ullat1 = t1[3]
#     lrlon1 = t1[0] + x1*t1[1]
#     lrlat1 = t1[3] + y1*t1[5]
#
#     ullon2 = t2[0]
#     ullat2 = t2[3]
#     lrlon2 = t2[0] + x1*t1[1]
#     lrlat2 = t2[3] + y1*t1[5]
#
#     ullat = min(ullat1,ullat2)
#     ullon = max(ullon1,ullon2)
#     lrlat = max(lrlat1,lrlat2)
#     lrlon = min(lrlon1,lrlon2)
#
#     return (ullon,ullat,lrlon,lrlat)


# Extract boundary of GeoTIFF file into geometry with geographic coordinates
def geotiff2boundary(inGeotiff):
    # Generating a mask for the GeoTIFF
    inRaster = gdal.Open(inGeotiff)
    geoTrans = inRaster.GetGeoTransform()
    proj = osr.SpatialReference()
    proj.ImportFromWkt(inRaster.GetProjectionRef())
    inBand = inRaster.GetRasterBand(1)
    data = inBand.ReadAsArray()
    [cols, rows] = data.shape
    data[data > 0] = 1
    data = filt.binary_closing(
        data, iterations=10, structure=np.ones((3, 3))).astype(data.dtype)
    gdalDriver = gdal.GetDriverByName('Mem')
    outRaster = gdalDriver.Create('out', rows, cols, 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform(geoTrans)
    outRaster.SetProjection(proj.ExportToWkt())
    outBand = outRaster.GetRasterBand(1)
    outBand.WriteArray(data)
    inBand = None
    inRaster = None
    data = None

    # Polygonize the raster image
    inBand = outRaster.GetRasterBand(1)
    ogrDriver = ogr.GetDriverByName('Memory')
    outVector = ogrDriver.CreateDataSource('out')
    outLayer = outVector.CreateLayer('boundary', srs=proj)
    fieldDefinition = ogr.FieldDefn('ID', ogr.OFTInteger)
    outLayer.CreateField(fieldDefinition)
    gdal.Polygonize(inBand, inBand, outLayer, 0, [], None)
    outRaster = None

    # Extract geometry from layer
    inSpatialRef = outLayer.GetSpatialRef()
    multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
    for outFeature in outLayer:
        geometry = outFeature.GetGeometryRef()
        multipolygon.AddGeometry(geometry)
        outFeature = None
    outLayer = None

    # Convert geometry from projected to geographic coordinates
    (multipolygon, outSpatialRef) = geometry_proj2geo(multipolygon, inSpatialRef)

    return multipolygon, outSpatialRef


# Converted geometry from projected to geographic
def geometry_proj2geo(inMultipolygon, inSpatialRef):

    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(4326)
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    outMultipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
    for polygon in inMultipolygon:
        if inSpatialRef != outSpatialRef:
            polygon.Transform(coordTrans)
        outMultipolygon.AddGeometry(polygon)

    return outMultipolygon, outSpatialRef


# Save geometry with fields to shapefile
def geometry2shape(fields, values, spatialRef, merge, shapeFile):

    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(shapeFile):
        driver.DeleteDataSource(shapeFile)
    outShape = driver.CreateDataSource(shapeFile)
    outLayer = outShape.CreateLayer('layer', srs=spatialRef)
    for field in fields:
        fieldDefinition = ogr.FieldDefn(field['name'], field['type'])
        if field['type'] == ogr.OFTString:
            fieldDefinition.SetWidth(field['width'])
        outLayer.CreateField(fieldDefinition)
    featureDefinition = outLayer.GetLayerDefn()
    if merge == True:
        combine = ogr.Geometry(ogr.wkbMultiPolygon)
        for value in values:
            combine = combine.Union(value['geometry'])
        outFeature = ogr.Feature(featureDefinition)
        for field in fields:
            name = field['name']
            outFeature.SetField(name, 'multipolygon')
        outFeature.SetGeometry(combine)
        outLayer.CreateFeature(outFeature)
        outFeature.Destroy()
    else:
        for value in values:
            outFeature = ogr.Feature(featureDefinition)
            for field in fields:
                name = field['name']
                outFeature.SetField(name, value[name])
            outFeature.SetGeometry(value['geometry'])
            outLayer.CreateFeature(outFeature)
            outFeature.Destroy()
    outShape.Destroy()


def shapefile_generation(spatialRef, polygon, shapeFile):
    fields = []
    field = {}
    field['name'] = 'Poly'
    field['type'] = ogr.OFTString
    field['width'] = 15
    fields.append(field)
    values = []
    value = {}
    value['Poly'] = 'boundary'
    value['geometry'] = polygon
    values.append(value)
    geometry2shape(fields, values, spatialRef, False, shapeFile)


def overlap_polygons(firstPolygon, secondPolygon):

    multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
    overlap = firstPolygon.Intersection(secondPolygon)
    geometryName = overlap.GetGeometryName()
    if geometryName == 'GEOMETRYCOLLECTION':
        for geometry in overlap:
            if geometry.GetGeometryName() == 'POLYGON':
                multipolygon.AddGeometry(geometry)
    elif geometryName == 'POLYGON':
        multipolygon.AddGeometry(overlap)
    elif geometryName == 'MULTIPOLYGON':
        multipolygon = overlap

    return multipolygon


def get_projected_vector_geometry(shapeFile, rasterSpatialRef):

    driver = ogr.GetDriverByName('ESRI Shapefile')
    shape = driver.Open(shapeFile, 0)
    vectorMultipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
    layer = shape.GetLayer()
    vectorSpatialRef = layer.GetSpatialRef()
    if vectorSpatialRef != rasterSpatialRef:
        coordTrans = osr.CoordinateTransformation(
            vectorSpatialRef, rasterSpatialRef)
    for feature in layer:
        geometry = feature.GetGeometryRef()
        count = geometry.GetGeometryCount()
        if geometry.GetGeometryName() == 'MULTIPOLYGON':
            for i in range(count):
                polygon = geometry.GetGeometryRef(i)
                if vectorSpatialRef != rasterSpatialRef:
                    polygon.Transform(coordTrans)
                vectorMultipolygon.AddGeometry(polygon)
        else:
            if vectorSpatialRef != rasterSpatialRef:
                geometry.Transform(coordTrans)
            vectorMultipolygon.AddGeometry(geometry)
    shape.Destroy()

    return vectorMultipolygon


# Extract geometry from shapefile
def shape2geometry_only(shapeFile):

    driver = ogr.GetDriverByName('ESRI Shapefile')
    shape = driver.Open(shapeFile, 0)
    multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
    layer = shape.GetLayer()
    spatialRef = layer.GetSpatialRef()
    for feature in layer:
        geometry = feature.GetGeometryRef()
        count = geometry.GetGeometryCount()
        if geometry.GetGeometryName() == 'MULTIPOLYGON':
            for i in range(0, count):
                polygon = geometry.GetGeometryRef(i)
                multipolygon.AddGeometry(polygon)
        else:
            multipolygon.AddGeometry(geometry)
    shape.Destroy()

    return multipolygon, spatialRef


# def main(prefile,postfile,preH=None,postH=None,outfile=None):
def main():
    # pixel sizes between images are considered equal if diff is less than pixel_eps (UTM)
    pixel_eps = 0.01

    i = readInputs()
    print('PreImage: ' + i.PreImage)
    print('PostImage: ' + i.PostImage)
    prefile = i.PreImage
    postfile = i.PostImage

    # geographic coords.
    pre_multipolygon, gcsSpatialRef = geotiff2boundary(prefile)
    pre_raster = gdal.Open(prefile)
    pre_outSpatialRef = osr.SpatialReference()
    pre_outSpatialRef.ImportFromWkt(pre_raster.GetProjectionRef())
    coordTransPre = osr.CoordinateTransformation(
        gcsSpatialRef, pre_outSpatialRef)
    pre_gt = pre_raster.GetGeoTransform()

    post_multipolygon, gcsSpatialRef = geotiff2boundary(postfile)
    post_raster = gdal.Open(postfile)
    post_outSpatialRef = osr.SpatialReference()
    post_outSpatialRef.ImportFromWkt(post_raster.GetProjectionRef())
    coordTransPost = osr.CoordinateTransformation(
        gcsSpatialRef, post_outSpatialRef)
    post_gt = post_raster.GetGeoTransform()

    # determine pixel width
    if abs(pre_gt[1]-post_gt[1]) < pixel_eps:
        pixelWidth = abs(pre_gt[1])
    else:
        print('Error: Pixel widths differ between images ' +
              prefile+' and '+postfile)
        sys.exit(1)

    # determine pixel height
    if abs(pre_gt[5]-post_gt[5]) < pixel_eps:
        pixelHeight = abs(pre_gt[5])
    else:
        print('Error: Pixel heights differ between images ' +
              prefile+' and '+postfile)
        sys.exit(1)

    # test geotiff2boundary(inGeotiff)

    # re-project post_multipolygon if needed
    if pre_outSpatialRef == post_outSpatialRef:
        pre_multipolygon.Transform(coordTransPre)
        post_multipolygon.Transform(coordTransPost)
    else:
        print('Needed to re-project post_multipolygon.')
        pre_multipolygon.Transform(coordTransPre)
        post_multipolygon.Transform(coordTransPre)

    # shapefile_generation(pre_outSpatialRef,pre_multipolygon,'pre_polygon.shp')
    # coordTrans = osr.CoordinateTransformation(vectorSpatialRef, rasterSpatialRef)
    # polygon.Transform(coordTrans)

    # shapefile_generation(post_outSpatialRef,post_multipolygon,'post_polygon.shp')

    # print 'pre_multipolygon:'
    # print pre_multipolygon
    # print ' '
    # print 'pre_outSpatialRef:'
    # print pre_outSpatialRef
    # print ' '
    # print 'post_multipolygon:'
    # print post_multipolygon
    # print ' '
    # print 'post_outSpatialRef:'
    # print post_outSpatialRef
    # print ' '

    if i.aoi is None:
        # if an area-of-interest is NOT defined
        overlap_poly = overlap_polygons(pre_multipolygon, post_multipolygon)
        overlap_SpatialRef = pre_outSpatialRef
    else:
        # if an area-of-interest is defined
        overlap_poly_temp = overlap_polygons(
            pre_multipolygon, post_multipolygon)
        overlap_SpatialRef = pre_outSpatialRef
        AOI_poly, AOI_SpatialRef = shape2geometry_only(i.aoi)
        AOI_poly.Transform(coordTransPre)
        if ogr.Geometry.Contains(overlap_poly_temp, AOI_poly):
            # if area-of-interest is fully contained by the intersecton of granule scenes
            overlap_poly = AOI_poly
        elif ogr.Geometry.Overlaps(overlap_poly_temp, AOI_poly):
            # if only part of the area-of-interest lies within the intersection of granule scenes
            overlap_poly = overlap_polygons(overlap_poly_temp, AOI_poly)
        else:
            # if the area-of-interest is completely disjointed from the intersection of the granule scenes
            print(
                'Error: Area of interest is disjointed from the intersection of the granule scenes.')
            print(
                '       The intersection of scenes will be used for analysis instead of the area of interest.')
            overlap_poly = overlap_poly_temp

    # overlap_poly = pre_multipolygon.Intersection(post_multipolygon)
    # print overlap_poly
    # print overlap_poly.GetEnvelope()

    # print dir(overlap_poly)
    # print overlap_poly.GetGeometryName()
    # if overlap_poly.GetGeometryName()=='MULTIPOLYGON':
    #     for polygon in overlap_poly:
    #         print polygon.GetEnvelope()
    #         print dir(polygon)
    # print overlap_poly
    # print overlap_poly.GetPoints()
    # print overlap_poly.GetExtent()

    env = overlap_poly.GetEnvelope()
    # print 'xmin='+str(env[0])
    # print 'xmax='+str(env[1])
    # print 'ymin='+str(env[2])
    # print 'ymax='+str(env[3])
    # print 'xRange='+str(env[1]-env[0])
    # print 'yRange='+str(env[3]-env[2])

    xMin = env[0]
    xMax = env[1]
    yMin = env[2]
    yMax = env[3]
    xRange = xMax-xMin
    yRange = yMax-yMin
    colRange = np.ceil((xRange+1)/pixelWidth)
    rowRange = np.ceil((yRange+1)/pixelHeight)

    # print 'pre width and height = '+str(pre_gt[1])+' '+str(pre_gt[5])
    # print 'post width and height = '+str(post_gt[1])+' '+str(post_gt[5])
    # print 'number of columns = '+str(colRange)
    # print 'number of rows = '+str(rowRange)

    # generate overlap.shp
    shapefile_generation(overlap_SpatialRef, overlap_poly, 'overlap.shp')

    if i.bb:
        # generate overlap_BB.shp (shapefile of bounding box of overlap.shp)
        ring_BB = ogr.Geometry(ogr.wkbLinearRing)
        ring_BB.AddPoint_2D(xMin, yMin)
        ring_BB.AddPoint_2D(xMin, yMax)
        ring_BB.AddPoint_2D(xMax, yMax)
        ring_BB.AddPoint_2D(xMax, yMin)
        ring_BB.AddPoint_2D(xMin, yMin)
        overlap_BB_poly = ogr.Geometry(ogr.wkbPolygon)
        overlap_BB_poly.AddGeometry(ring_BB)
        shapefile_generation(overlap_SpatialRef,
                             overlap_BB_poly, 'overlap_BB.shp')
        CropShapeFile = 'overlap_BB.shp'
    else:
        CropShapeFile = 'overlap.shp'

    # cmd1=' '.join(['gdalwarp','-cutline overlap.shp','-crop_to_cutline', \
    #               '-overwrite','-dstnodata 0',prefile,'PreImage_Crop_geo.tif'])
    # cmd2=' '.join(['gdalwarp','-cutline overlap.shp','-crop_to_cutline', \
    #               '-overwrite','-dstnodata 0',postfile,'PostImage_Crop_geo.tif'])
    # os.system(cmd1)
    # os.system(cmd2)

    if colRange <= i.c and rowRange <= i.r:
        # if the bounding box of the overlap is small enough, then crop it and use it
        print('Cropping images...')
        gdal.Warp('PreImage_Crop_geo.tif', prefile,
                  cutlineDSName=CropShapeFile, cropToCutline=True, dstNodata=0)
        gdal.Warp('PostImage_Crop_geo.tif', postfile,
                  cutlineDSName=CropShapeFile, cropToCutline=True, dstNodata=0)
    else:
        # if the overlap is too large, decimate/resample the image
        xFactor = colRange/i.c
        yFactor = rowRange/i.r
        Factor = max(np.ceil(xFactor), np.ceil(yFactor))

        print('Cropping images...')
        gdal.Warp('pre_temp_geo.tif', prefile,
                  cutlineDSName=CropShapeFile, cropToCutline=True, dstNodata=0)
        gdal.Warp('post_temp_geo.tif', postfile,
                  cutlineDSName=CropShapeFile, cropToCutline=True, dstNodata=0)

        # down-sample and save final geotiffs
        gdal.Translate('PreImage_Crop_geo.tif', 'pre_temp_geo.tif', xRes=pixelWidth *
                       Factor, yRes=pixelHeight*Factor, resampleAlg='average', noData=0)
        gdal.Translate('PostImage_Crop_geo.tif', 'post_temp_geo.tif', xRes=pixelWidth *
                       Factor, yRes=pixelHeight*Factor, resampleAlg='average', noData=0)
        print('Needed to decimate/resample images by a factor of '+str(Factor)+'...')

        # delete temporary geotiffs
        os.remove('pre_temp_geo.tif')
        os.remove('post_temp_geo.tif')

        # Lines of unused code to follow:

        # # calculate intersection between AOI and overlap_poly
        # AOI_poly,AOI_SpatialRef=shape2geometry_only(AOI_shapeFile)
        # # ???? AOI_poly.Transform(coordTransPre) ?????
        # overlap_AOI_poly = overlap_polygons(AOI_poly, overlap_poly)
        # env2=overlap_AOI_poly.GetEnvelope()
        # xMin2=env2[0]
        # xMax2=env2[1]
        # yMin2=env2[2]
        # yMax2=env2[3]
        # xRange2=xMax2-xMin2
        # yRange2=yMax2-yMin2
        # colRange2=np.ceil((xRange2+1)/abs(pixelWidth))
        # rowRange2=np.ceil((yRange2+1)/abs(pixelHeight))
        # if colRange2 <= i.c and rowRange2 <= i.r:
        # # if the intersection between the scene overlap and the AOI is small enough, use it
        # shapefile_generation(overlap_SpatialRef,overlap_AOI_poly,'overlap.shp')
        # # ???? Do I use overlap_SpatialRef or AOI_SpatialRef above ?????
        # os.system('gdalwarp -cutline overlap.shp -crop_to_cutline '+prefile+' PreImage_Crop_geo.tif')
        # os.system('gdalwarp -cutline overlap.shp -crop_to_cutline '+postfile+' PostImage_Crop_geo.tif')

        # if AOI is defined and not disjointed, then....
        # if i.AOI_Shapefile is not None:
        # I can assume that SpacialRef for AOI is the same as preImage??...make option??
        # # AOI_poly=get_projected_vector_geometry(i.AOI_Shapefile, pre_outSpatialRef)
        # AOI_poly,AOI_SpatialRef=shape2geometry_only(shapeFile)
        # if ogr_g_overlaps(overlap_poly, AOI_poly):

        # preH=None
        # postH=None
        # outfile=None
        #
        # # procTwo = True
        # # if preH is not None:
        # #     procTwo = False
        #
        # # Open prefile, get projection and pixsize
        # dst1 = gdal.Open(prefile)
        # t1 = dst1.GetGeoTransform()
        # pixsize = t1[1]
        # p1 = dst1.GetProjection()
        # # print p1
        #
        # # Open up postfile, get projection
        # dst2 = gdal.Open(postfile)
        # p2 = dst2.GetProjection()
        #
        # # Cut the UTM zone out of projection1
        # ptr = p1.find("UTM zone ")
        # (zone1,hemi) = [t(s) for t,s in zip((int,str), re.search("(\d+)(.)",p1[ptr:]).groups())]
        # print "zone 1 is %s" % zone1
        # print "hemisphere is %s" % hemi
        #
        # # Cut the UTM zone out of projection2
        # ptr = p2.find("UTM zone ")
        # zone2 = re.search("(\d+)",p2[ptr:]).groups()
        # zone2 = int(zone2[0])
        # print "zone 2 is %s" % zone2
        #
        # if zone1 != zone2:
        #     print "Projections don't match..."
        #     print "hemisphere is %s" % hemi
        #     print "zone is %s" % zone1
        #     if hemi == "N":
        #         proj = ('EPSG:326%02d' % int(zone1))
        #     else:
        #         proj = ('EPSG:327%02d' % int(zone1))
        #
        #     print "    reprojecting post image"
        #     print "    proj is %s" % proj
        #     gdal.Warp("tempPostFile.tif",postfile,dstSRS=proj,xRes=pixsize,yRes=pixsize)
        #     postfile = "tempPostFile.tif"
        #     # if procTwo == False:
        #     #     print "    reprojecting post crosspol image"
        #     #     gdal.Warp("tempPostHFile.tif",postH,dstSRS=proj,xRes=pixsize,yRes=pixsize)
        # 	#     postH = "tempPostHFile.tif"
        #
        # coords = getOverlap(prefile,postfile)
        # # print coords
        #
        # dst_d1 = gdal.Translate('',prefile,format = 'MEM',projWin=coords)
        #
        # # determine srcWinW (width) and srcWinL (left postion) from inputs
        # if 'ha' in i and not i.ha is None:
        #     # print 'i.ha = '+str(i.ha)
        #     srcWinL = min(max(0,int(i.ha)), dst_d1.RasterXSize-1)
        #     if 'c' in i:
        #         # print 'i.c = '+str(i.c)
        #         srcWinW = np.amin([max(0,int(i.c)),dst_d1.RasterXSize-srcWinL, 4096])
        #     else:
        #         srcWinW = min(dst_d1.RasterXSize-srcWinL, 4096)
        # elif 'c' in i and not i.c is None:
        #     # print 'i.c = '+str(i.c)
        #     srcWinW = np.amin([max(0,int(i.c)), dst_d1.RasterXSize, 4096])
        #     srcWinL = int((dst_d1.RasterXSize-srcWinW)/2)
        # else:
        #     srcWinW = min(dst_d1.RasterXSize, 4096)
        #     srcWinL = int((dst_d1.RasterXSize-srcWinW)/2)
        #
        # # determine srcWinH (height) and srcWinT (top postion) from inputs
        # if 'va' in i and not i.va is None:
        #     # print 'i.va = '+str(i.va)
        #     srcWinT = min(max(0,int(i.va)), dst_d1.RasterYSize-1)
        #     if 'r' in i:
        #         # print 'i.r = '+str(i.r)
        #         srcWinH = np.amin([max(0,int(i.r)),dst_d1.RasterYSize-srcWinT, 4096])
        #     else:
        #         srcWinH = min(dst_d1.RasterYSize-srcWinT, 4096)
        # elif 'r' in i and not i.r is None:
        #     # print 'i.r = '+str(i.r)
        #     srcWinH = np.amin([max(0,int(i.r)), dst_d1.RasterYSize, 4096])
        #     srcWinT = int((dst_d1.RasterYSize-srcWinH)/2)
        # else:
        #     srcWinH = min(dst_d1.RasterYSize, 4096)
        #     srcWinT = int((dst_d1.RasterYSize-srcWinH)/2)
        #
        # # print [srcWinL, srcWinT, srcWinW, srcWinH]
        #
        # # crop and save pre-image
        # # dst_d1 = gdal.Translate('',prefile,format = 'MEM',projWin=coords)
        # # print dst_d1.RasterXSize
        # # print dst_d1.RasterYSize
        # gdal.Translate('PreImage_Crop_geo.tif', dst_d1, srcWin=[srcWinL, srcWinT, srcWinW, srcWinH])
        # del dst_d1
        #
        # # crop and save post-image
        # dst_d2 = gdal.Translate('',postfile,format = 'MEM',projWin=coords)
        # # print dst_d2.RasterXSize
        # # print dst_d2.RasterYSize
        # gdal.Translate('PostImage_Crop_geo.tif', dst_d2, srcWin=[srcWinL, srcWinT, srcWinW, srcWinH])
        # del dst_d2


if __name__ == '__main__':
    main()
