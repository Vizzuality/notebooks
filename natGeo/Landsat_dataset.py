from urllib.request import urlopen
import zipfile
import rasterio
import os, urllib
import sys
import shutil
import numpy as np
import math
import ee

def cloudMaskL457(image):
    qa = image.select('pixel_qa')
    #If the cloud bit (5) is set and the cloud confidence (7) is high
    #or the cloud shadow bit is set (3), then it's a bad pixel.
    cloud = qa.bitwiseAnd(1 << 5).And(qa.bitwiseAnd(1 << 7)).Or(qa.bitwiseAnd(1 << 3));
    #Remove edge pixels that don't occur in all bands
    mask2 = image.mask().reduce(ee.Reducer.min());
    return image.updateMask(cloud.Not()).updateMask(mask2);

def Cloud_Free_Composite_L7_5(Collection_id, startDate, stopDate, geom, scale, bandNames = None):
    ## Define your collection
    collection = ee.ImageCollection(Collection_id)

    ## Filter 
    collection = collection.filterBounds(geom).filterDate(startDate,stopDate)\
            .map(cloudMaskL457)

    ## Composite
    composite = collection.median()

    ## Choose the scale
    composite =  composite.reproject(crs='EPSG:4326', scale=scale)

    ## Select the bands
    if bandNames:
        composite = composite.select(bandNames)
    
    return composite

def download_image_tif(image, download_zip, mn, mx, scale, bandNames = None, region = None):
    
    if bandNames:
        image = image.select(bandNames)
        
    Vizparam = {'min': mn, 'max': mx, 'scale': scale, 'crs': 'EPSG:4326'}
    if region:
        Vizparam['region'] = region
    
   
    url = image.getDownloadUrl(Vizparam)     

    print('Downloading image...')
    print("url: ", url)
    data = urlopen(url)
    with open(download_zip, 'wb') as fp:
        while True:
            chunk = data.read(16 * 1024)
            if not chunk: break
            fp.write(chunk)
            
    # extract the zip file transformation data
    z = zipfile.ZipFile(download_zip, 'r')
    target_folder_name = download_zip.split('.zip')[0]
    z.extractall(target_folder_name)
    print('Download complete!')
        
def load_tif_bands(path, files):
    data = np.array([]) 
    for n, file in enumerate(files):
        image_path = path+file
        image = rasterio.open(image_path)
        data = np.append(data, image.read(1))
    data = data.reshape((n+1, image.read(1).shape[0], image.read(1).shape[1]))
    data = np.moveaxis(data, 0, 2)
    
    return data


class landsat_dataset:
    
    def __init__(self, point, buffer, startDate, stopDate, scale):
        """
        Class used to get the datasets for Sentinel 2 Cloud Free Composite and 
        USDA NASS Cropland Data Layers
        Parameters
        ----------
        point : list
            A list of two [x,y] coordinates with the center of the area of interest.
        buffer : number
            Buffer in meters
        startDate : string
        stopDate : string
        scale: number
            Pixel size in meters.

        """
        
        self.point = point
        self.buffer = buffer
        self.startDate = startDate
        self.stopDate = stopDate       
        self.scale = scale 
        
        # Area of Interest
        self.geom = ee.Geometry.Point(self.point).buffer(self.buffer)
        self.region = self.geom.bounds().getInfo()['coordinates']
        
        # Image Collections
        self.input_collection = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')
        
    def read_datasets(self):
        
        ## Composite
        input_image = Cloud_Free_Composite_L7_5(self.input_collection, self.startDate, self.stopDate, self.geom, self.scale)
        
        ## Calculate NDVI
        image_ndvi = input_image.normalizedDifference(['B4','B3'])
        
        ## Concatenate images into one multi-band image
        input_image = ee.Image.cat([input_image.select(['B3','B2','B1','B4']), image_ndvi])
             
        # Choose the scale
        input_image =  input_image.reproject(crs='EPSG:4326', scale=self.scale)
        
        
        # Download images as tif
        download_image_tif(input_image, 'data.zip', mn=0, mx=3000, scale = self.scale, region = self.region)
        
        # Load data
        directory_x = "./data/"
        files_x = sorted(f for f in os.listdir(directory_x) if f.endswith('.' + 'tif'))
        
        data_x = load_tif_bands(directory_x, files_x)
        
        # Remove data folders and files
        files=["data.zip"]
        for file in files:
            ## If file exists, delete it ##
            if os.path.isfile(file):
                os.remove(file)
            else:    ## Show an error ##
                print("Error: %s file not found" % file)
        ## Try to remove tree; if failed show an error using try...except on screen
        for folder in ["./data", "./cdl_data"]:
            try:
                shutil.rmtree(folder)
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))
   
            
        return data_x