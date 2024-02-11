from colour import Color
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
from multiprocessing import Pool

# import streamlit as st


def get_dataset(csv_name):
    dataset = pd.read_csv(csv_name)
    dataset = dataset.drop_duplicates()

    # pick columns to use
    columns = [
        "geoname_id",
        "name",
        "population",
        "cou_name_en",
        "latitude",
        "longitude",
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
    lat1 = dataset_item["latitude"]
    lon1 = dataset_item["longitude"]
    city_max_dist = 0
    for _, row in dataset.iterrows():
        lat2 = row["latitude"]
        lon2 = row["longitude"]
        dist = get_dist(lat1, lon1, lat2, lon2)
        if dist > city_max_dist:
            city_max_dist = dist
    return city_max_dist


def get_city_item(name_or_id, dataset):
    def get_city_with_max_pop(subset):
        return subset[subset["population"] == subset["population"].max()].iloc[0]

    if isinstance(name_or_id, int):
        subset = dataset[dataset["geoname_id"] == name_or_id]
    else:
        subset = dataset[dataset["name"].str.contains(name_or_id) == True]
    return get_city_with_max_pop(subset)


def get_viability(city_1, city_2, dataset, dataset_statistics, A=1, B=1, verbose=False):
    min_pop, max_pop, avg_pop, min_dist, max_dist = (
        dataset_statistics["min_pop"],
        dataset_statistics["max_pop"],
        dataset_statistics["avg_pop"],
        dataset_statistics["min_dist"],
        dataset_statistics["max_dist"],
    )
    pop_1 = city_1["population"]
    pop_2 = city_2["population"]
    sum_pop = pop_1 + pop_2
    dist = get_dist(
        city_1["latitude"], city_1["longitude"], city_2["latitude"], city_2["longitude"]
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
        if row["geoname_id"] != city["geoname_id"]:
            viability = get_viability(city, row, dataset)
            if viability > max_viability:
                max_viability = viability
                max_viable_city = row
    return max_viable_city


def find_top_viable_cities(city_name, dataset, dataset_statistics, top_n=5, A=1, B=1):
    city = get_city_item(city_name, dataset)
    viabilities = []
    for _, row in dataset.iterrows():
        if row["geoname_id"] != city["geoname_id"]:
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
                    [city_1["longitude"], city_1["latitude"]],
                    [city_2["longitude"], city_2["latitude"]],
                ],
                "viability": f"{city[0] * 100}%",
                "city_2": city_2["name"],
            }
        )
    red = Color("red")
    colors = list(red.range_to(Color("green"), len(paths_dict)))
    for i, path in enumerate(paths_dict):
        path["color"] = colors[i].hex_l
    return pd.DataFrame(paths_dict)


dataset = get_dataset("../data/city_proper_over_1000.csv")

# remove empty rows
dataset = dataset.dropna()
print(len(dataset))

dataset = dataset[dataset["cou_name_en"] == "United States"]
dataset_statistics = {
    "min_pop": dataset["population"].min(),
    "max_pop": dataset["population"].max(),
    "avg_pop": dataset["population"].mean(),
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
    print(city_item["name"])
    city = city_item
    max_viable_cities = find_top_viable_cities(
        city["name"], dataset, dataset_statistics=dataset_statistics
    )
    return {"genome_id": city["geoname_id"], "max_viable_cities": max_viable_cities}


from tqdm.contrib.concurrent import process_map


def get_most_viable_for_all(dataset):
    max_viable_cities = process_map(
        get_most_viable_data, dataset.to_dict("records"), max_workers=5, chunksize=1
    )
    return max_viable_cities


if __name__ == "__main__":
    max_viable_cities = get_most_viable_for_all(dataset)
    print(max_viable_cities)
