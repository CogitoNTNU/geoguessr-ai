import pandas as pd
import numpy as np
import math

#Read datafile
road = pd.read_csv('backend/roadNetworkSize.txt')
totalPoints = 100000 #total points you want distrubuted

countries = road['country'].tolist()
road_network_size = road['road_km'].tolist()

#Calculate distribution of points
total_road = np.sum(road_network_size)

country_points = []
for i in range(len(countries)):
    country_points.append(math.ceil((road_network_size[i]*totalPoints)/total_road))

#Sort the country list
country_point_pairs = list(zip(countries, country_points))
country_point_pairs.sort(key=lambda x: x[0]) 

#Print sorted results
for country, points in country_point_pairs:
    print(country, ": ", points)






