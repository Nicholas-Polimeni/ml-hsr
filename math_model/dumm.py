from colour import Color
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
from multiprocessing import Pool

# import streamlit as st

GEONAME_ID = "metro"
NAME = "metro"
POPULATION = "population"
COU_NAME_EN = "country_code"
LATITUDE = "latitude"
LONGITUDE = "longitude"
DATASET_CSV_PATH = "../data/metro_regions.csv"


def get_dataset(csv_name):
    dataset = pd.read_csv(csv_name)
    dataset = dataset.drop_duplicates()

    # pick columns to use
    columns = [
        # GEONAME_ID,
        NAME,
        POPULATION,
        COU_NAME_EN,
        LATITUDE,
        LONGITUDE,
    ]
    dataset = dataset[columns]
    return dataset


def get_dist(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in km
    dLat = np.deg2rad(lat2 - lat1)
    dLon = np.deg2rad(lon2 - lon1)
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(np.deg2rad(lat1)) * np.cos(
        np.deg2rad(lat2)
    ) * np.sin(dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c  # Distance in km
    return d


def get_max_dist(dataset_item, dataset):
    lat1 = dataset_item[LATITUDE]
    lon1 = dataset_item[LONGITUDE]
    city_max_dist = 0
    for _, row in dataset.iterrows():
        lat2 = row[LATITUDE]
        lon2 = row[LONGITUDE]
        dist = get_dist(lat1, lon1, lat2, lon2)
        if dist > city_max_dist:
            city_max_dist = dist
    return city_max_dist


def get_city_item(name_or_id, dataset):
    def get_city_with_max_pop(subset):
        return subset[subset[POPULATION] == subset[POPULATION].max()].iloc[0]

    if isinstance(name_or_id, int):
        subset = dataset[dataset[GEONAME_ID] == name_or_id]
    else:
        subset = dataset[dataset[NAME].str.contains(name_or_id) == True]
    if len(subset) == 0:
        raise Exception(f"City {name_or_id} not found")
    return get_city_with_max_pop(subset)


def get_viability(city_1, city_2, dataset, dataset_statistics, A=1, B=1, verbose=False):
    min_pop, max_pop, avg_pop, min_dist, max_dist = (
        dataset_statistics["min_pop"],
        dataset_statistics["max_pop"],
        dataset_statistics["avg_pop"],
        dataset_statistics["min_dist"],
        dataset_statistics["max_dist"],
    )
    pop_1 = city_1[POPULATION]
    pop_2 = city_2[POPULATION]
    sum_pop = pop_1 + pop_2
    dist = get_dist(
        city_1[LATITUDE], city_1[LONGITUDE], city_2[LATITUDE], city_2[LONGITUDE]
    )
    normalized_pop = (sum_pop - min_pop) / (max_pop + avg_pop - min_pop)
    normalized_dist = (dist - min_dist) / (max_dist - min_dist)
    if verbose:
        print(sum_pop, normalized_pop, dist, normalized_dist)
    return A * normalized_pop - B * normalized_dist


def find_most_viable_city(city_name, dataset):
    city = get_city_item(city_name, dataset)
    max_viability = 0
    max_viable_city = None
    for _, row in dataset.iterrows():
        if row[GEONAME_ID] != city[GEONAME_ID]:
            viability = get_viability(city, row, dataset)
            if viability > max_viability:
                max_viability = viability
                max_viable_city = row
    return max_viable_city


def find_top_viable_cities(city_name, dataset, dataset_statistics, top_n=5, A=1, B=1):
    city = get_city_item(city_name, dataset)
    viabilities = []
    for _, row in dataset.iterrows():
        if row[GEONAME_ID] != city[GEONAME_ID]:
            viability = get_viability(city, row, dataset, dataset_statistics, A, B)
            viabilities.append((viability, row))
    viabilities.sort(key=lambda x: x[0], reverse=True)
    return viabilities[:top_n]


def get_viable_cities_paths(
    city_1_name, dataset, dataset_statistics, top_n=5, A=1, B=1
):
    city_1 = get_city_item(city_1_name, dataset)
    most_viable_cities = find_top_viable_cities(
        city_1_name, dataset, dataset_statistics, top_n, A, B
    )
    paths_dict = []
    for city in most_viable_cities:
        city_2 = city[1]
        paths_dict.append(
            {
                "path": [
                    [city_1[LONGITUDE], city_1[LATITUDE]],
                    [city_2[LONGITUDE], city_2[LATITUDE]],
                ],
                "viability": f"{city[0] * 100}%",
                "city_2": city_2[NAME],
            }
        )
    red = Color("red")
    colors = list(red.range_to(Color("green"), len(paths_dict)))
    for i, path in enumerate(paths_dict):
        path["color"] = colors[i].hex_l
    return pd.DataFrame(paths_dict)


dataset = get_dataset(DATASET_CSV_PATH)

# remove empty rows
dataset = dataset.dropna()
print(len(dataset))

# dataset = dataset[dataset[COU_NAME_EN] == "United States"]
dataset_statistics = {
    "min_pop": dataset[POPULATION].min(),
    "max_pop": dataset[POPULATION].max(),
    "avg_pop": dataset[POPULATION].mean(),
    "min_dist": 0,
    "max_dist": get_max_dist(get_city_item("Atlanta", dataset), dataset),
}

# Text box input
user_city_input = "Atlanta"
user_A_input = 1
user_B_input = 1
df1 = get_viable_cities_paths(
    user_city_input,
    dataset,
    dataset_statistics,
    5,
    A=float(user_A_input),
    B=float(user_B_input),
)


def get_most_viable_data(city_item):
    city = city_item
    max_viable_cities = find_top_viable_cities(
        city[NAME], dataset, dataset_statistics=dataset_statistics
    )
    return {"source": city, "max_viable_cities": max_viable_cities}


def get_most_viable_for_all(dataset):
    max_viable_cities = []
    with Pool(4) as p:
        max_viable_cities = p.map(get_most_viable_data, (row for _, row in dataset.iterrows()))
    # for _, city_item in tqdm(dataset.iterrows(), total=len(dataset)):
    #     max_viable_cities.append(get_most_viable_data(city_item))
    return max_viable_cities


def get_best_of_best(dataset, top_n=5):
    max_viable_cities = get_most_viable_for_all(dataset)
    max_viable_cities_list = []
    for city_source in max_viable_cities:
        city_source_item = city_source["source"]
        for viable_city in city_source["max_viable_cities"]:
            max_viable_cities_list.append(
                {
                    "source": city_source_item[NAME],
                    "source_lat": city_source_item[LATITUDE],
                    "source_lon": city_source_item[LONGITUDE],
                    "destination": viable_city[1][NAME],
                    "destination_lat": viable_city[1][LATITUDE],
                    "destination_lon": viable_city[1][LONGITUDE],
                    "viability": viable_city[0],
                }
            )
    max_viable_cities_df = pd.DataFrame(max_viable_cities_list)
    max_viable_cities_df = max_viable_cities_df.sort_values(
        by=["viability"], ascending=False
    )
    best_of_best = max_viable_cities_df.head(top_n)
    return best_of_best

if __name__ == "__main__":
    bb = get_best_of_best(dataset, 5)
    print(bb)
