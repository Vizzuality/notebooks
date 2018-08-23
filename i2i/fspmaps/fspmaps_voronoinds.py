

import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
from iso3166 import countries
import matplotlib.pyplot as plt
from shapely.ops import cascaded_union
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree
import rasterio
from rasterstats import zonal_stats
import matplotlib as mpl
import matplotlib.cm as cm
from shapely.ops import cascaded_union


# ---------------------------------  Read FSP maps data -----------------------------------------
fsp = pd.read_csv('/Users/ikersanchez/Vizzuality/PROIEKTUAK/i2i/Data/FSP_Maps/FSP_maps.csv', index_col=0)

# ---------------------------------  Read country maps -------------------------------------------
# ** India **
indMap = gpd.read_file('/Users/ikersanchez/Vizzuality/PROIEKTUAK/i2i/Data/gadm36_shp/gadm36_IND.shp')

# Get Uttar Pradesh and Bihar states
indMapStates = indMap[(indMap['NAME_1'] == 'Uttar Pradesh') | (indMap['NAME_1'] == 'Bihar')]

# ** Kenya **
kenMap = gpd.read_file('/Users/ikersanchez/Vizzuality/PROIEKTUAK/i2i/Data/gadm36_shp/gadm36_KEN.shp')

# ** Uganda **
ugaMap = gpd.read_file('/Users/ikersanchez/Vizzuality/PROIEKTUAK/i2i/Data/gadm36_shp/gadm36_UGA.shp')

# ** Bangladesh **
bgdMap = gpd.read_file('/Users/ikersanchez/Vizzuality/PROIEKTUAK/i2i/Data/gadm36_shp/gadm36_BGD.shp')

# ** Nigeria **
ngaMap = gpd.read_file('/Users/ikersanchez/Vizzuality/PROIEKTUAK/i2i/Data/gadm36_shp/gadm36_NGA.shp')

# ** Tanzania **
tzaMap = gpd.read_file('/Users/ikersanchez/Vizzuality/PROIEKTUAK/i2i/Data/gadm36_shp/gadm36_TZA.shp')

# Get the boundary of each country
indBoundary = gpd.GeoSeries(cascaded_union(indMapStates['geometry']))
indBoundary = gpd.GeoDataFrame(indBoundary).rename(columns={0: 'geometry'})
indBoundary['country'] = 'India'

kenBoundary = gpd.GeoSeries(cascaded_union(kenMap['geometry']))
kenBoundary = gpd.GeoDataFrame(kenBoundary).rename(columns={0: 'geometry'})
kenBoundary['country'] = 'Kenya'

ugaBoundary = gpd.GeoSeries(cascaded_union(ugaMap['geometry']))
ugaBoundary = gpd.GeoDataFrame(ugaBoundary).rename(columns={0: 'geometry'})
ugaBoundary['country'] = 'Uganda'

bgdBoundary = gpd.GeoSeries(cascaded_union(bgdMap['geometry']))
bgdBoundary = gpd.GeoDataFrame(bgdBoundary).rename(columns={0: 'geometry'})
bgdBoundary['country'] = 'Bangladesh'

ngaBoundary = gpd.GeoSeries(cascaded_union(ngaMap['geometry']))
ngaBoundary = gpd.GeoDataFrame(ngaBoundary).rename(columns={0: 'geometry'})
ngaBoundary['country'] = 'Nigeria'

tzaBoundary = gpd.GeoSeries(cascaded_union(tzaMap['geometry']))
tzaBoundary = gpd.GeoDataFrame(tzaBoundary).rename(columns={0: 'geometry'})
tzaBoundary['country'] = 'Tanzania'

boundaries = gpd.GeoDataFrame(pd.concat([indBoundary,kenBoundary,ugaBoundary,bgdBoundary,ngaBoundary,tzaBoundary]))

# ----------------------------------------  Functions -------------------------------------------
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def voronoi_tesellation_box(boundary,lng,lat):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    boundary : GeoDataFrame, 
        Geometry of the country.
    lng : GeoSeries, 
        Longitud values of points. 
    lat : GeoSeries, 
        Longitud values of points. 
    Returns
    -------
    voronoid : GeaoDataFrames
        Geometries of Voronoi regions.
    """
    # array with points coordinates
    points = np.zeros((lng.shape[0],2))
    points[:,0] = lng
    points[:,1] = lat

    # compute Voronoi tesselation
    vor = Voronoi(points)
    
    # Reconstruct infinite voronoi regions in a 2D diagram to finite regions.
    regions, vertices = voronoi_finite_polygons_2d(vor)
    
    # build box from country boundary
    xmin = boundary.bounds.minx[0]
    xmax = boundary.bounds.maxx[0]
    ymin = boundary.bounds.miny[0]
    ymax = boundary.bounds.maxy[0]

    box = Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])

    voronoid = [] 
    for region in regions:
        polygon = vertices[region]
        # Clipping polygon
        poly = Polygon(polygon)
        voronoid.append(poly.intersection(box))
        
    voronoid = gpd.GeoDataFrame(geometry = voronoid)
    
    vor_lng = vor.points[:,0]
    vor_lat = vor.points[:,1]
    
    voronoid['lng'] = vor_lng
    voronoid['lat'] = vor_lat
    
    return voronoid    

def spatial_overlays(df1, df2):
    '''Compute overlay intersection of two 
        GeoPandasDataFrames df1 and df2
    '''
    df1 = df1.copy()
    df2 = df2.copy()
    df1['geometry'] = df1.geometry.buffer(0)
    df2['geometry'] = df2.geometry.buffer(0)

    # Spatial Index to create intersections
    spatial_index = df2.sindex
    df1['bbox'] = df1.geometry.apply(lambda x: x.bounds)
    df1['histreg']=df1.bbox.apply(lambda x:list(spatial_index.intersection(x)))
    pairs = df1['histreg'].to_dict()
    nei = []
    for i,j in pairs.items():
        for k in j:
            nei.append([i,k])
        
    pairs = gpd.GeoDataFrame(nei, columns=['idx1','idx2'], crs=df1.crs)
    pairs = pairs.merge(df1, left_on='idx1', right_index=True)
    pairs = pairs.merge(df2, left_on='idx2', right_index=True, suffixes=['_1','_2'])
    pairs['Intersection'] = pairs.apply(lambda x: (x['geometry_1'].intersection(x['geometry_2'])).buffer(0), axis=1)
    pairs = gpd.GeoDataFrame(pairs, columns=pairs.columns, crs=df1.crs)
    cols = pairs.columns.tolist()
    cols.remove('geometry_1')
    cols.remove('geometry_2')
    cols.remove('histreg')
    cols.remove('bbox')
    cols.remove('Intersection')
    dfinter = pairs[cols+['Intersection']].copy()
    dfinter.rename(columns={'Intersection':'geometry'}, inplace=True)
    dfinter = gpd.GeoDataFrame(dfinter, columns=dfinter.columns, crs=pairs.crs)
    dfinter = dfinter.loc[dfinter.geometry.is_empty==False]
    dfinter.drop(['idx1','idx2'], axis=1, inplace=True)
    return dfinter

def distances_map_cKDTree(boundary, pixel_size, points):
    
    xmin = int(np.floor(boundary.bounds.minx[0]))
    xmax = int(np.ceil(boundary.bounds.maxx[0]))
    ymin = int(np.floor(boundary.bounds.miny[0]))
    ymax = int(np.ceil(boundary.bounds.maxy[0]))
    
    x = np.linspace(xmin, xmax, int((xmax-xmin)/pixel_size)+1)
    y = np.linspace(ymin, ymax, int((ymax-ymin)/pixel_size)+1)
    
    tree = cKDTree(points)
    
    image = np.zeros((len(y),len(x)))
    
    for i in range(len(x)):
        for j in range(len(y)):
            image[j,i] = tree.query([x[i],y[j]])[0]
    return image


# -------------------------  Iterate over Sector, Country, and Type  ----------------------------
gdf = gpd.GeoDataFrame(columns=['id','geometry','count','max','mean']) 
for i in fsp['sector'].unique():
    for j in fsp[fsp['sector'] == i]['country'].unique():
        for k in fsp[(fsp['sector'] == i) & (fsp['country'] == j)]['type'].unique():
            print('Sector: ', i, ', Country: ', j, ', and Type: ', k)
            lng = fsp[(fsp['sector'] == i) & (fsp['country'] == j) & (fsp['type'] == k)]['lng']
            lat = fsp[(fsp['sector'] == i) & (fsp['country'] == j) & (fsp['type'] == k)]['lat'] 

            boundary = gpd.GeoDataFrame(boundaries[boundaries['country'] == j]['geometry'])

            if (i == 'Finance' and j == 'India' and k == 'Bank Customer Service Points'):
                lat.iloc[9265] = "{0:.4f}".format(lat.iloc[9265])

            if len(lat) >= 4:
                # ** Build a Voronoi tessellation from points **
                voronoid = voronoi_tesellation_box(boundary,lng,lat)

                # Coordinate reference system : WGS84
                boundary.crs = {'init': 'epsg:4326'}
                voronoid.crs = {'init': 'epsg:4326'}


                # ** Intersect voronoid with boundary **
                # Remove GeometryCollection and replace it by a nearby Polygon
                ind = voronoid[voronoid['geometry'].type == 'GeometryCollection'].index
                if len(voronoid[voronoid['geometry'].type == 'GeometryCollection']) > 0:
                    for p in range(len(voronoid[voronoid['geometry'].type == 'GeometryCollection'])):
                        voronoid.geometry.iloc[ind[p]] = voronoid.geometry.iloc[ind[p]-1]  
            
                voronoid= spatial_overlays(voronoid, boundary)


                # ** Nearest neighbour distance map **
                points = np.zeros((lng.shape[0],2))
                points[:,0] = lng
                points[:,1] = lat

                distance = distances_map_cKDTree(boundary, 0.025, points)

                # Saving raster data
                pixel_size = 0.025
                west = int(np.floor(boundary.bounds.minx[0])) - pixel_size/2
                north = int(np.ceil(boundary.bounds.maxy[0])) - pixel_size/2

                trans = rasterio.transform.from_origin(west, north, pixel_size, pixel_size)

                dataset = rasterio.open('/Users/ikersanchez/Vizzuality/PROIEKTUAK/i2i/Data/FSP_Maps/distance/distance_'+i+'_'+j+'_'+k.replace('/', ' ')+'.tif', 'w', driver='GTiff',
                                        height=distance.shape[0], width=distance.shape[1],
                                        count=1, dtype='float64',
                                        crs='EPSG:4326', transform=trans)

                distance = np.flip(distance,axis=0)

                dataset.write(distance, 1)

                dataset.close()


                # ** Zonal statistics **
                with rasterio.open('./data/distance.tif') as dataset:
                    myData=dataset.read(1)

                zs = zonal_stats(voronoid, '/Users/ikersanchez/Vizzuality/PROIEKTUAK/i2i/Data/FSP_Maps/distance/distance_'+i+'_'+j+'_'+k.replace('/', ' ')+'.tif',  all_touched=True)
                zs = gpd.GeoDataFrame(zs)
                voronoid_zs = voronoid.join(zs)

                # ** Voronoid table with the same ID as  point table **
                df = fsp[(fsp['sector'] == i) & (fsp['country'] == j) & (fsp['type'] == k)]
                df.reset_index(inplace=True)
                voronoid_zs['id'] = df['id']
                voronoid_zs = voronoid_zs[['id','geometry','count','max','mean']]

            else:
                voronoid_zs = gpd.GeoDataFrame(index = range(len(lat)),  columns=['id','geometry','count','max','mean']) 
                df = fsp[(fsp['sector'] == i) & (fsp['country'] == j) & (fsp['type'] == k)]
                df.reset_index(inplace=True)
                voronoid_zs['id'] = df['id']

            gdf = pd.concat([gdf,voronoid_zs])

            # ** Save voronoid table **
            gdf.to_csv('/Users/ikersanchez/Vizzuality/PROIEKTUAK/i2i/Data/FSP_Maps/FSP_voronoids.csv')



