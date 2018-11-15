import os
import requests
import urllib
import numpy as np
import math
import ee_collection_specifics
from IPython.display import display, Image
import ee
    
class ee_composite:
    
    def __init__(self, point, buffer, startDate, stopDate, scale, path, collection):
        """
        Class used to get the datasets from Earth Engine
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
        collection: string
            Name of each collection.

        """
        
        self.point = point
        self.buffer = buffer
        self.startDate = startDate
        self.stopDate = stopDate       
        self.scale = scale 
        self.path = path 
        self.collection = collection
        
        # Area of Interest
        self.geom = ee.Geometry.Point(self.point).buffer(self.buffer)
        self.region = self.geom.bounds().getInfo()['coordinates']
        
        # Image Collection
        self.image_collection = ee_collection_specifics.ee_collections(self.collection)
        
        # Image Visualization parameters
        self.vis = ee_collection_specifics.ee_vis(self.collection)
        
        # Saving parameters
        self.visSave = {'dimensions': 1024, 'format': 'png', 'crs': 'EPSG:4326'}
        
        
    def read_composite(self):
        
        ## Composite
        self.image = ee_collection_specifics.Composite(self.collection)(self.image_collection, self.startDate, self.stopDate, self.geom, self.scale)
        
        self.image = self.image.visualize(**self.vis)
    
    def save_composite_png(self):
        """
        Save composite as png
        """ 
        self.visSave['region'] = self.region
    
        path = self.image.getThumbURL(self.visSave)
    
        if os.path.exists(self.path):
            os.remove(self.path)
    
        urllib.request.urlretrieve(path, self.path)
        
    def display_composite(self):
        """
        Displays composite in notebook
        """ 
        visual = Image(url=self.image.getThumbUrl({
                    'region':self.region
                    }))
    
        display(visual)
        