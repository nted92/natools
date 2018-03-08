import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

import json
from geojson import Polygon
from scipy.spatial import ConvexHull
from shapely.geometry import Point, MultiPoint, mapping
from shapely.geometry.polygon import Polygon
from area import area

import random

from timezonefinder import TimezoneFinder

import gmaps
api_key = ""  # Google Maps
# Javascript API key - belongs to the account ??
gmaps.configure(api_key=api_key)

"""
NOTE: it was chosen to pass a point to the following functions as two 
arguments: a latitutde and a longitude. If working with tuples to represent 
points, the solution is to simply unpack them when passing arguments to the 
functions: e.g. *(lat,lon)
"""

"""
NOTE: shapely also contains a lot of the following capabilities
"""

"""
Change vocabulary so that distinction between 'polygon' and 'CONVEX HULL' is clear
(function names, variable names, and docstrings).

Also, seem to be recalculating ConvexHull too many times, everywhere, so see
what can get simplified here...
"""

# ==========================================
#               Geo constants
# ==========================================

MEAN_EARTH_RADIUS = 6371.0088 * 1000  # in meters
ELLIPSOID_PARAMETERS = {"a": 6378137, "b": 6356752.314245}  # WGS84 model

# ==========================================
#     Basic Earth related functionalities
# ==========================================


def get_local_earth_radius(latitude):
    """
    :param latitude: (decimal degrees)
    :return: the local radius (meters)
    
    It assumes an ellipsoidal model of the Earth (WGS 84).
    It doesn't account for the altitude (only on the surface of the Earth).
    """
    a, b = ELLIPSOID_PARAMETERS["a"], ELLIPSOID_PARAMETERS["b"]
    num = ((a**2) * np.cos(latitude))**2 + ((b**2) * np.sin(latitude))**2
    den = (a * np.cos(latitude))**2 + (b * np.sin(latitude))**2
    return np.sqrt(num / den)


def convert_great_circle_distance_to_radians(distance, latitude=None):
    """
    :param distance: quantity to convert (meters)
    :param latitude: optional (decimal degrees)
    :return: the converted quantity (radians)

    It is strictly valid only for small surface distances (up to a couple 
    kilometers).
    For greater distances, use the Vincenty's formulae.
    
    Details: for small distances, the Earth can be approximated locally by a 
    sphere. So the following formula holds: l = R * cos(theta), where l is the
    great circle distance, R is the local Earth's radius and theta is the 
    angle we want to return at the surface of the Earth. For very small 
    angles, cos(theta) ~ theta, so the formula becomes: theta ~ l / R.
    The local Earth radius R is calculated assuming a global ellipsoid model 
    of the Earth (WGS 84).
    """
    if latitude is None:
        return distance / MEAN_EARTH_RADIUS
    else:
        return distance / get_local_earth_radius(latitude)


def convert_great_circle_distance_to_degrees(distance, latitude=None):
    """
    :param distance: quantity to convert (meters)
    :param latitude: optional (decimal degrees)
    :return: the converted quantity (degrees)
    
    Please refer to the convert_great_circle_distance_to_radians method for 
    further explanations.
    """
    return np.rad2deg(convert_great_circle_distance_to_radians(distance=distance,
                                                               latitude=latitude))


def compute_great_circle_distance_between_two_points(lat1, lon1, lat2, lon2):
    """
    :param lat1: latitude of the first point (decimal degrees)
    :param lon1: longitude of the first point (decimal degrees)
    :param lat2: latitude of the second point (decimal degrees)
    :param lon2: longitude of the second point (decimal degrees)
    :return: the great-circle distance (meters)
    
    It assumes a spherical model of the earth. For an ellipsoidal model (
    WGS84), should use the Vincenty's formulae.
    
    NOTE 1: if you have tuples for each point, let's say point_1 = (lat1, 
    lon1) and point_2 = (lat2, lon2), the function can be called by passing 
    the following argument: *point1+point2. The star in Python unpacks 
    iterables.
    
    NOTE 2: if you are interested, the geopy package contains this distance, 
    as well as some more advanced ones like the Vincenty one, 
    etc. They can be found in geopy.distance
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = np.deg2rad([lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = MEAN_EARTH_RADIUS  # mean radius of earth in meters
    return c * r


def check_if_point_in_vicinity_of_place(point_lat, point_lon,
                                        place_lat, place_lon,
                                        radius=50):
    """
    :param point_lat: latitude of the point (decimal degrees)
    :param point_lon: longitude of the point (decimal degrees)
    :param place_lat: latitude of the place (decimal degrees)
    :param place_lon: longitude of the place (decimal degrees)
    :param radius: optional, vicinity tolerance (meters)
    :return: boolean

    It checks if a point is within a certain radius of a place (disc shape).
    """
    if compute_great_circle_distance_between_two_points(place_lat,
                                                        place_lon,
                                                        point_lat,
                                                        point_lon) < radius:
        return True
    else:
        return False


def get_centroid_of_points(points):
    """
    :param points: 2d numpy array of (lat, lon) points of shape (num_points, 2) 
    (decimal degrees)
    :return: (lat, lon) tuple of the true area-weighted geographic centroid 
    (decimal degrees)
    
    Behind the scenes, it translates each point into a vector in the 
    3D cartesian (x, y, z) space whose center is the center of the 
    Earth, computes the mean vector and project it back on the surface of 
    the earth. For different definitions of the centroid, please refer to 
    the following: http://www.geomidpoint.com/calculation.html
    
    Note: handling None values because other functions here can create such 
    value (e.g. polygon envelope)
    """
    if points is None:
        warnings.warn("None value was passed to get_centroid_of_points...")
        return None
    else:
        return MultiPoint(points).centroid.x, MultiPoint(points).centroid.y


def get_centermost_point_of_points(points):
    """
    :param points: 2d numpy array of (lat, lon) points of shape (num_points, 2) 
    (decimal degrees)
    :return: (lat, lon) tuple of the point, picked among the input points, 
    that is the closest to the true area-weighted geographic centroid of the 
    group of points (decimal degrees)
    
    Please refer to get_centroid_of_points for further explanations.
    
    Note: handling None values because other functions here can create such 
    value (e.g. polygon envelope)
    """
    if points is None:
        warnings.warn("None value was passed to get_centermost_point_of_points...")
        return None
    else:
        centroid = get_centroid_of_points(points)
        return tuple(min(points,
                         key=lambda point: compute_great_circle_distance_between_two_points(*tuple(point)+centroid)))


# ==========================================
#              Machine Learning
# ==========================================


def cluster_points(points, epsilon=100, min_samples=1, centroid=True):
    """
    :param points: 2d numpy array of (lat, lon) points of shape (num_points, 2) 
    (decimal degrees)
    :param epsilon: DBSCAN epsilon parameter (meters)
    :param min_samples: DBSCAN min_samples parameter (this includes the point itself)
    :param centroid: optional, whether to define clusters' centers with a true 
    centroid or with the centermost point (boolean)
    :return: a tuple of a pandas Dataframe containing the found clusters 
    described by their centroid and the points they are made of (no clusters
    found if empty), and the clusters labels per point for the input points
    (the noisy points that were not clusterd are labelled -1, so they can be
    easily retrieved by running the following simple code: points[clusters_labels_per_point == -1]

    It runs a DBSCAN algorithm over the data and returns the results in a 
    convenient way. It assumes a spherical model of the Earth.

    NOTE: a balance was found between handling locations that are all close 
    to each other (typically around a set of close cities, a few kilometers 
    apart), and handling locations that are scattered all around the globe. 
    A spherical model of the earth is thus used, with the great circle 
    distance as the distance metric between pairs of points. Those 
    simplifications create local distorsions in some cases.
    """
    if (points is not None) and (points.shape[0] > 0):
        points_in_radians = np.radians(points)
        epsilon_in_radians = convert_great_circle_distance_to_radians(epsilon)
        db = DBSCAN(eps=epsilon_in_radians, min_samples=min_samples,
                    algorithm='ball_tree', metric='haversine')
        db.fit(points_in_radians)
        clusters_labels_per_point = db.labels_  # label -1 is noise
        labels = np.unique(clusters_labels_per_point)
        labels = np.delete(labels, np.argwhere(labels == -1))
        if labels.shape[0] == 0:
            return pd.DataFrame(columns=["centroid_lat", "centroid_lon", "points"]), clusters_labels_per_point
        else:
            clusters_points = pd.Series([points[clusters_labels_per_point == n] for n in labels])
            if centroid:
                clusters_centers = clusters_points.map(get_centroid_of_points)
            else:
                clusters_centers = clusters_points.map(
                    get_centermost_point_of_points)
            clusters_dict = {
                "centroid_lat": [clusters_centers.ix[label][0] for label in labels],
                "centroid_lon": [clusters_centers.ix[label][1] for label in labels],
                "points": [clusters_points.ix[label] for label in labels]}
            clusters = pd.DataFrame(clusters_dict, index=labels)
            return clusters, clusters_labels_per_point
    else:
        warnings.warn("None value or empty array was passed to cluster_points...")
        return None, None


# ==========================================
#   Advanced Earth related functionalities
# ==========================================


def sample_regular_grid(lat1, lon1, lat2, lon2, spacing, verbose=False):
    """
    :param lat1: latitude of the first point (decimal degrees)
    :param lon1: longitude of the first point (decimal degrees)
    :param lat2: latitude of the second point (decimal degrees)
    :param lon2: longitude of the second point (decimal degrees)
    :param spacing: grid spacing (in meters)
    :param verbose: whether to plot the points (lat, lon) pairs or not
    :return: 3d numpy array, consisting of 2d numpy array representing the lines
    in the longitude direction (ordered inside from lon1 to lon2, and outside
    from lat1 to lat2), the 3rd dimension being the lat/lon thickness of the
    pairs of points.
    
    The two points define the rectangle, as being diametrically opposed.
    """
    n_lat = int(round(compute_great_circle_distance_between_two_points(lat2, lon1, lat1, lon1) / spacing))
    e_lat = (lat2 - lat1) / n_lat
    n_lon = int(round(compute_great_circle_distance_between_two_points(lat1, lon1, lat1, lon2) / spacing))
    e_lon = (lon2 - lon1) / n_lon

    points = []
    for i in range(n_lat + 1):
        line = []
        for j in range(n_lon + 1):
            line.append([lat1 + i * e_lat, lon1 + j * e_lon])
        points.append(line)

    if verbose:
        for line in points:
            for point in line:
                print(point[0], ",", point[1])

    return np.array(points)


def sample_points_uniformly_within_disk(center_lat, center_lon, radius=1000,
                                        num_points=100):
    """
    :param center_lat: the disk's center latitude (decimal degrees)
    :param center_lon: the disk's center longitude (decimal degrees)
    :param radius: optional, the disk's radius (meters)
    :param num_points: optional, the number of points to sample
    :return: 2d numpy array of points of shape (num_points, 2)

    The disk is an actual disk when near the equator, otherwise it starts
    looking like an elliptical disk stretching in the latitude direction,
    and shrinking in the longitude direction.

    For distances, an ellipsoidal model of the earth (WGS84) is used.
    """
    points = np.empty([num_points, 2])

    r = convert_great_circle_distance_to_degrees(radius, center_lat)
    for i in range(num_points):
        u, v = np.random.uniform(0, 1), np.random.uniform(0, 1)
        w = r * np.sqrt(u)  # random radius
        t = 2 * np.pi * v  # random angle
        x = w * np.cos(t)  # random x (in degrees)
        x1 = x  # could apply a transformation to x here
        y = w * np.sin(t)  # random y (in degrees)

        # Making sure that the point is valid
        point_lat = center_lat + x1
        point_lon = center_lon + y
        if (abs(point_lat) <= 90) and (abs(point_lon) <= 180):
            points[i, :] = [point_lat, point_lon]

    return points


def sample_points_uniformly_on_circle(center_lat, center_lon, radius=1000,
                                      num_points=100):
    """
    :param center_lat: the circle's center latitude (decimal degrees)
    :param center_lon: the circle's center longitude (decimal degrees)
    :param radius: optional, the circle's radius (meters)
    :param num_points: optional, the number of points to sample
    :return: 2d numpy array of points of shape (num_points, 2)

    The circle is an actual circle when near the equator, otherwise it starts
    looking like an ellipse stretching in the latitude direction, 
    and shrinking in the longitude direction.

    For distances, an ellipsoidal model of the earth (WGS84) is used.
    """
    points = np.empty([num_points, 2])

    r = convert_great_circle_distance_to_degrees(radius, center_lat)
    for k in range(num_points):
        theta = 2 * np.pi * (k / num_points)  # angle
        x, y = r * np.cos(theta), r * np.sin(theta)

        # Making sure that the point is valid
        point_lat = center_lat + x
        point_lon = center_lon + y
        if (abs(point_lat) <= 90) and (abs(point_lon) <= 180):
            points[k, :] = [point_lat, point_lon]

    return points


def sample_points_in_polygon(polygon, num_points):
    """
    CAREFUL: can be very long if the bounding box is much bigger than the polygon
    
    IMPROVEMENT: 
    https://cs.stackexchange.com/questions/14007/random-sampling-in-a-polygon
    https://math.stackexchange.com/questions/15624/distribute-a-fixed-number-of-points-uniformly-inside-a-polygon
    
    How to sample *uniformly* ?
    
    Add the None tests at the beginning and warnings
    """
    convex_hull = get_polygon_envelope_out_of_points(polygon)
    if convex_hull is None:
        warnings.warn("Cannot compute polygon in sample_points_uniformly_in_polygon...")
        return None
    else:
        shapely_polygon = convert_points_to_shapely_polygon(convex_hull)
        min_lat, min_lon, max_lat, max_lon = shapely_polygon.bounds

        points = np.array([]).reshape(-1, 2)

        while points.shape[0] < num_points:
            sample_lat = random.uniform(min_lat, max_lat)
            sample_lon = random.uniform(min_lon, max_lon)
            random_point = Point([sample_lat, sample_lon])
            if random_point.within(shapely_polygon):
                points = np.concatenate((points, np.array([[sample_lat, sample_lon]])))
        return points


def sample_points_on_polygon():
    """
    https://stackoverflow.com/questions/42023522/random-sampling-of-points-along-a-polygon-boundary
    """
    pass


def _get_polygon_envelope_out_of_points(points):
    """
    :param points: 2d numpy array of points of shape (num_points, 2) 
    (decimal lat/lon degrees). There should be at least 3 points.
    :return: scipy.spatial.ConvexHull object (or None)
    
    Only convex polygons are produced.
    """
    if (points is None) or (points.shape[0] < 3):
        warnings.warn("You should provide at least 3 points to compute a polygon envelop!")
        return None
    else:
        try:
            hull_object = ConvexHull(points)
            return hull_object
        except:
            warnings.warn("qhull input error: the points provided cannot produce a polygon")
            return None


def get_polygon_envelope_out_of_points(points, repeat_first=True):
    """
    :param points: 2d numpy array of points of shape (num_points, 2) 
    (decimal lat/lon degrees). There should be at least 3 points.
    :param repeat_first: optional, whether to append the first point at the 
    end in order to 'close the loop' (boolean)
    :return: 2d numpy array of points consisting of the polygon's 
    consecutive vertices ordered in the clockwise direction 
    
    A simple Convex Hull method is used.
    """
    envelope = _get_polygon_envelope_out_of_points(points)
    if envelope is not None:
        outline_points = [list(points[ind]) for ind in envelope.vertices]
        if repeat_first:
            outline_points.append(list(points[envelope.vertices[0]]))
        outline = np.array(outline_points)
        return outline
    else:
        return None


def get_polygon_envelope_out_of_points_for_geojson(points):
    """
    :param points: 2d numpy array of points of shape (num_points, 2) 
    (decimal lat/lon degrees). There should be at least 3 points.
    :return: list of (lat, lon) tuples consisting of the polygon's 
    consecutive vertices ordered in the clockwise direction, the first point 
    being repeated at the end

    A simple Convex Hull method is used.
    Formatted for the geojson and gmaps packages.
    """
    envelope = _get_polygon_envelope_out_of_points(points)
    if envelope is not None:
        outline = \
            [tuple(points[ind])[::-1] for ind in envelope.vertices] +\
            [tuple(points[envelope.vertices[0]])[::-1]]
        return outline
    else:
        return None


# @TODO: make it feed the following two very similar functions. Make it something even more general, as it is basically clustering + convex hull in the end
def _get_outer_and_inner_polygons_envelopes_out_of_points():
    pass


def get_outer_and_inner_polygons_envelopes_out_of_points(points, epsilon, min_points,
                                                         repeat_first=True):
    """
    :param points: 2d numpy array of points of shape (num_points, 2) 
    (decimal lat/lon degrees). There should be at least 3 points.
    :param epsilon: DBSCAN epsilon parameter (meters)
    :param min_points: DBSCAN min_points parameter
    :param repeat_first: optional, whether to append the first point at the 
    end in order to 'close the loop' (boolean)
    :return: tuple, consisting of one 2d numpy array of points containing the 
    outer polygon's consecutive vertices ordered in the clockwise  direction, 
    and one list of numpy arrays consisting of the inner polygons' 2d numpy 
    arrays, if any.
    
    A simple Convex Hull method is used.
    An inner DBSCAN is performed to find the inner polygons (regions of highest density).
    """
    outer_polygon = get_polygon_envelope_out_of_points(points, repeat_first)
    clusters_df, clusters_per_point = \
        cluster_points(points, epsilon=epsilon, min_samples=min_points, centroid=True)
    inner_polygons = \
        [get_polygon_envelope_out_of_points(clusters_df.loc[ind, "points"]) for ind in clusters_df.index]
    return outer_polygon, [poly for poly in inner_polygons if poly is not None]


def get_outer_and_inner_polygons_envelopes_out_of_points_for_geojson(points, epsilon, min_points):
    outer_polygon = get_polygon_envelope_out_of_points_for_geojson(points)
    clusters_df, clusters_per_point = \
        cluster_points(points, epsilon=epsilon, min_samples=min_points, centroid=True)
    inner_polygons = \
        [get_polygon_envelope_out_of_points_for_geojson(clusters_df.loc[ind, "points"]) for ind in clusters_df.index]
    return outer_polygon, [poly for poly in inner_polygons if poly is not None]


def convert_points_to_shapely_polygon(points):
    """
    :param points: either a 2d numpy array of the initial raw points of 
    shape (num_points, 2) from which the Convex Hull polygon should be 
    computed, or the polygon's envelope directly as a 2d numpy array of points 
    that are the polygon's consecutive vertices ordered in the clockwise 
    direction.
    :return: shapely.geometry.polygon.Polygon object, or None
    
    NOTE: looking for the Convex Hull of a Convex Hull is like applying the 
    identity function (i.e. f(x) = x) 
    """
    polygon = get_polygon_envelope_out_of_points_for_geojson(points)
    if polygon is None:
        return None
    else:
        new_polygon = [elt[::-1] for elt in polygon]
        shapely_polygon = Polygon(new_polygon)
        return shapely_polygon


def check_if_point_in_polygon(point_lat, point_lon, polygon):
    """
    :param point_lat: the point's latitude (decimal degrees)
    :param point_lon: the point's longitude (decimal degrees)
    :param polygon: either a 2d numpy array of the initial raw points of 
    shape (num_points, 2) from which the Convex Hull polygon should be 
    computed, or the polygon's envelope directly as a 2d numpy array of points 
    that are the polygon's consecutive vertices ordered in the clockwise 
    direction.
    :return: whether the point lies inside the polygon or not (boolean), 
    or None if didn't provide enough points for the polygon
    
    NOTE: looking for the Convex Hull of a Convex Hull is like applying the 
    identity function (i.e. f(x) = x) 
    """
    shapely_polygon = convert_points_to_shapely_polygon(polygon)
    if shapely_polygon is None:
        return None
    else:
        shapely_point = Point(point_lat, point_lon)
        return shapely_polygon.contains(shapely_point)


def compute_polygon_area(polygon):
    """
    :param polygon: either a 2d numpy array of the initial raw points of 
    shape (num_points, 2) from which the Convex Hull polygon should be 
    computed, or the polygon's envelope directly as a 2d numpy array of points 
    that are the polygon's consecutive vertices ordered in the clockwise 
    direction.
    :return: the polygon's area, in meters squared (m2), which only makes sense
    if the points are expressed as latitudes and longitudes

    NOTE: looking for the Convex Hull of a Convex Hull is like applying the 
    identity function (i.e. f(x) = x) 
    """
    polygon_obj = get_polygon_geojson_from_points(polygon)
    if polygon_obj is None:
        return 0
    else:
        polygon_obj = polygon_obj['features'][0]['geometry']
        return area(polygon_obj)


def compute_polygon_aspect_ratio(polygon, fast=True, plot=False):
    """
    :param polygon: either a 2d numpy array of the initial raw points of 
    shape (num_points, 2) from which the Convex Hull polygon should be 
    computed, or the polygon's envelope directly as a 2d numpy array of points 
    that are the polygon's consecutive vertices ordered in the clockwise 
    direction.
    :param fast: if True, will compute the eigenvectors on the ConvexHull, 
    otherwise it will sample a cloud of points inside the polygon
    :param plot: boolean
    :return: the polygon's aspect ratio, defined as the ratio between the biggest
    and the smallest eigenvalues of the polygon (only works because we are dealing
    with convex polygons)

    http://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/

    NOTE: looking for the Convex Hull of a Convex Hull is like applying the 
    identity function (i.e. f(x) = x) 
    """
    polygon_obj = get_polygon_envelope_out_of_points(polygon)
    if polygon_obj is None:
        return None
    else:
        if fast:
            xy = np.array([polygon_obj[:, 0].tolist(), polygon_obj[:, 1].tolist()])
            eigvals, eigvecs = np.linalg.eig(np.cov(xy))
            aspect_ratio = np.max(eigvals) / np.min(eigvals)
        else:
            aspect_ratios = []
            for i in range(5):
                inner_points = sample_points_in_polygon(polygon_obj, 1000)
                xy = np.array([inner_points[:, 0].tolist(), inner_points[:, 1].tolist()])
                eigvals, eigvecs = np.linalg.eig(np.cov(xy))
                aspect_ratios.append(np.max(eigvals) / np.min(eigvals))
            if len(aspect_ratios) == 0:
                aspect_ratio = None
            else:
                aspect_ratio = np.mean(aspect_ratios)

        if plot:
            fig, (ax1, ax2) = plt.subplots(nrows=2)
            x, y = xy
            center = xy.mean(axis=-1)
            for ax in [ax1, ax2]:
                ax.plot(x, y, 'ro')
                ax.axis('equal')

            for val, vec in zip(eigvals, eigvecs.T):
                val *= 2 * 1e4
                x, y = np.vstack(
                    (center + val * vec, center, center - val * vec)).T
                ax2.plot(x, y, 'b-', lw=3)

            plt.show()

        return aspect_ratio


# @TODO: Some functions should accept polygon envelopes directly, and not take back the points once again... it's re-doing unnecesary computations, even if the results is the same in the end...


# ==========================================
#               Visualisations
# ==========================================


def plot_polygon_envelope_out_of_points(points, fig_path=None):
    """
    :param points: 2d numpy array of points of shape (num_points, 2) 
    (decimal lat/lon degrees). There should be at least 3 points.
    :param fig_path: optional, path where to save the figure
    :return: None (but plots an interactive map, and can save to file)
    
    For high-quality image saving, should use the 'Agg' matplotlib backend.
    """
    if points.shape[0] < 3:
        warnings.warn("You should provide at least 3 points!")
    else:
        fig, ax = plt.subplots()
        try:
            hull_object = ConvexHull(points)
        except:
            warnings.warn("qhull input error: the points can't produce a polygon")
            return
        ax.plot(points[:, 0], points[:, 1], 'o')
        for simplex in hull_object.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 'r--')
        if fig_path is not None:
            fig.tight_layout()
            fig.savefig(fig_path, format='png')
            plt.close(fig)


def plot_outer_and_inner_polygons_envelopes_out_of_points(points, fig_path=None):
    pass


def get_polygon_geojson_from_points(points):
    """
    :param points: 2d numpy array of points of shape (num_points, 2) 
    (decimal lat/lon degrees). There should be at least 3 points.
    :return: the geojson dictionary for gmaps, or None if less than 3 points
    
    NOTE: if None is returned, it will be handled by the 
    plot_places_on_gmaps function to avoid raising a gmaps error. For the 
    data to be displayed anyway, it shouldn't be passed to gmaps as a 
    polygon_geojson type altogether, but as a 'points' type for the symbol 
    layer.
    """
    polygon_envelope = get_polygon_envelope_out_of_points_for_geojson(points)
    if polygon_envelope is not None:
        geometry_json = json.loads(json.dumps(mapping(Polygon(polygon_envelope))))
        geo_json_object = \
            {"type": "FeatureCollection",
             "features": [{'type': 'Feature',
                           'geometry': geometry_json,
                           'properties': {"hitMap": points.shape[0]}}]}
        return geo_json_object
    else:
        return None


def get_outer_and_inner_polygons_geojsons_from_points(points):
    pass


def get_circle_geojson_for_point(point_lat, point_lon, radius, num_samples=50):
    """
    :param point_lat: point latitude (decimal degrees)
    :param point_lon: point longitude (decimal degrees)
    :param radius: circle's radius (meters)
    :param num_samples: optional, number of points to samples (have to be more 
    than 3)
    :return: the geojson dictionary for gmaps, or None if less than 3 points
    
    NOTE: if None is returned, it will be handled by the 
    plot_places_on_gmaps function to avoid raising a gmaps error. For the 
    data to be displayed anyway, it shouldn't be passed to gmaps as a 
    polygon_geojson type altogether, but as a 'points' type for the symbol 
    layer.
    """
    if num_samples >= 3:
        circle_samples = \
            sample_points_uniformly_on_circle(point_lat, point_lon,
                                              radius, num_samples)
        circle_samples = np.concatenate((circle_samples, np.array([circle_samples[0]])))
        circle_envelope = [tuple(circle_samples[ind])[::-1]
                           for ind in range(circle_samples.shape[0])]
        geometry_json = json.loads(json.dumps(mapping(Polygon(circle_envelope))))
        geo_json_object = \
            {"type": "FeatureCollection",
             "features": [{'type': 'Feature',
                           'geometry': geometry_json,
                           'properties': {}}]}
        return geo_json_object
    else:
        warnings.warn("You should provide at least 3 points!")
        return None


def plot_places_on_gmaps(polygon_geojsons=None, circle_geojsons=None,
                         heatmaps=None, points=None):
    """
    :param polygon_geojsons: list of polygon_geojsons for geojson_layer gmaps
    objects
    :param circle_geojsons: list of circle_geojsons for geojson_layer gmaps
    objects
    :param heatmaps: list of 2d numpy array of points of shape (num_points, 2)
    (decimal lat/lon degrees) for heatmap_layer gmaps objects
    :param points: 2d numpy array of points of shape (num_points, 2)
    (decimal lat/lon degrees), for a symbol_layer gmaps object
    :return: the gmaps.figure() object, which is a Google Maps interactive map

    gmaps documentation: http://jupyter-gmaps.readthedocs.io/en/latest/gmaps.html

    NOTE: when trying to create the polygon and circle geojsons, make sure 
    there are more than 3 points being provided, otherwise they won't 
    appear. In this case, should put them in the 'points' category.

    NOTE: the heatmap colour levels are independent from one heatmap object to 
    another

    NOTE: when showing map in jupyter notebook, the widget's "save" button 
    doesn't work (saves a blank image)

    WARNING: should create a Neura Google Maps Javascript API key. The one
    it is currently using is hard-coded in this package.
    """
    fig = gmaps.figure()

    if polygon_geojsons is not None:
        for geojson in polygon_geojsons:
            if geojson is not None:
                try:
                    geojson_layer = gmaps.geojson_layer(geojson,
                                                        fill_color=['red'],
                                                        stroke_color=['red'],
                                                        fill_opacity=0.1)
                    fig.add_layer(geojson_layer)
                except:
                    warnings.warn("failed to add polygon")
                    continue

    if circle_geojsons is not None:
        for geojson in circle_geojsons:
            if geojson is not None:
                try:
                    geojson_layer = gmaps.geojson_layer(geojson, fill_opacity=0.1)
                    fig.add_layer(geojson_layer)
                except:
                    warnings.warn("failed to add circle")
                    continue

    if heatmaps is not None:
        for heatmap in heatmaps:
            try:
                heatmap_layer = gmaps.heatmap_layer(heatmap)
                heatmap_layer.point_radius = 15
                heatmap_layer.gradient = [
                    (255, 0, 0, 0),
                    (255, 0, 0, 0.7),
                    (255, 0, 0, 0.99)
                ]
                fig.add_layer(heatmap_layer)
            except:
                warnings.warn("failed to add heatmap")
                continue

    if points is not None:
        try:
            symbol_layer = gmaps.symbol_layer(points,
                                              fill_color="rgba(0, 0, 255, 0.5)",
                                              stroke_color="rgba(0, 0, 255, 0.5)")
            fig.add_layer(symbol_layer)
        except:
            warnings.warn("failed to add points")

    return fig


# ==========================================
#          Space and time coupling
# ==========================================


def get_timezone_finder_object():
    return TimezoneFinder()


def get_timezone_name_from_point(lat, lon, timezone_finder_object,
                                 tz_back_up=None):
    """
    :param lat: point latitude (decimal degrees)
    :param lon: point longitude (decimal degrees)
    :param timezone_finder_object: timezonefinder.TimezoneFinder() object, 
    can get it and initialize it only once from the get_timezone_finder_object 
    function
    :param tz_back_up: a fallback timezone name just in case
    :return: the best timezone name it could find (string), None otherwise
    
    It performs an offline lookup. It doesn't require an internet connection.
    
    NOTE: could even try to increase the search radius when it is still 
    None, like delta_degree=3 etc.
    """
    lat, lon = float(lat), float(lon)
    try:
        timezone_name = timezone_finder_object.timezone_at(lat=lat, lng=lon)
        if timezone_name is None:
            timezone_name = timezone_finder_object.closest_timezone_at(lat=lat, lng=lon)
    except ValueError:
        timezone_name = None
    if (timezone_name is None) and (tz_back_up is not None):
        timezone_name = tz_back_up
    return timezone_name


# ==========================================
#           Neura's data specifics
# ==========================================


def get_mongo_geo_query_dict(lat, lon, radius):
    """
    :param lat: latitude of the central point (decimal degrees)
    :param lon: longitude of the central point (decimal degrees)
    :param radius: (meters)
    :return: a dictionary

    It produces the dictionary to insert in a Mongo query in order to match 
    the documents that contain a 'location' field that are within a certain 
    radius of the given (lat, lon) point.
    """
    return {"$geoWithin": {"$centerSphere": [[lon, lat],
                                             convert_great_circle_distance_to_radians(
                                                 radius, lat)]}}

# TODO
# user_geologs (use Elastic search..?) : have wrapper that know if to query
# this or this database...
# user_position_at_time (use Elastic search..?)
# TZ tools: user timezone for dateInt, for timestamp, also use AR, etc.


# ==========================================
#                   Others
# ==========================================
pass
# TODO
# + look for other things in other projects, etc.


if __name__ == "__main__":
    pass
    # ========================
    #  Haversine vs. Vincenty
    # ========================

    # from geopy.distance import vincenty
    #
    # point_1 = (32.169500, 34.807121)
    # point_2 = (32.166500, 34.809121)
    #
    # print(compute_great_circle_distance_between_two_points(*point_1+point_2))
    #
    # print(vincenty(point_1, point_2).meters)

    # ========================
    #     Centroid methods
    # ========================

    # # The following accounts for the error of both the "circle/disk sampling
    # # process" (which doesn't sample a perfectly shaped circle or disk),
    # # and the centroid computation method.
    #
    # true_center = (32.169500, 34.807121)
    #
    # points_circle = sample_points_uniformly_on_circle(*true_center,
    #                                                   radius=100,
    #                                                   num_points=10)
    # points_disk = sample_points_uniformly_within_disk(*true_center,
    #                                                   radius=100,
    #                                                   num_points=10)
    #
    # print("CIRCLE")
    # centroid = get_centroid_of_points(points_circle)
    # centroid_error = compute_great_circle_distance_between_two_points(
    #     *true_center+centroid)
    #
    # centermost = get_centermost_point_of_points(points_circle)
    # centermost_error = compute_great_circle_distance_between_two_points(
    #     *true_center+centermost)
    #
    # print("Centroid: {}, error {} meters".format(centroid, centroid_error))
    # print("Centermost: {}, error {} meters".format(centermost, centermost_error))
    # print()
    #
    # print("DISK")
    # centroid = get_centroid_of_points(points_disk)
    # centroid_error = compute_great_circle_distance_between_two_points(
    #     *true_center+centroid)
    #
    # centermost = get_centermost_point_of_points(points_disk)
    # centermost_error = compute_great_circle_distance_between_two_points(
    #     *true_center+centermost)
    #
    # print("Centroid: {}, error {} meters".format(centroid, centroid_error))
    # print("Centermost: {}, error {} meters".format(centermost, centermost_error))

    # ========================
    #         Polygon
    # ========================

    point_to_test = (32.161212, 34.807579)
    polygon = \
        np.array([[32.15904295, 34.81977801],
                  [32.15757849, 34.81131917],
                  [32.15859842, 34.81007924],
                  [32.1612109, 34.80762882],
                  [32.16370818, 34.80620902],
                  [32.16992767, 34.80652106],
                  [32.17182619, 34.8017596],
                  [32.17332121, 34.81008394],
                  [32.17430054, 34.81312372],
                  [32.17428625, 34.81640221],
                  [32.17420266, 34.81731306],
                  [32.17297216, 34.81912149],
                  [32.16762784, 34.82243909],
                  [32.16582332, 34.82302114],
                  [32.16157795, 34.82136284],
                  [32.15904295, 34.81977801]])
    print(check_if_point_in_polygon(*point_to_test, polygon))
