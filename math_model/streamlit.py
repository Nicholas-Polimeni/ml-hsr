from colour import Color
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


def draw_all(path_df):
    # path_df["path"] = path_df["path"].apply(lambda x: eval(x))

    st.dataframe(path_df)

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    path_df["color"] = path_df["color"].apply(hex_to_rgb)

    view_state = pdk.ViewState(latitude=37.782556, longitude=-122.3484867, zoom=2)

    layer = pdk.Layer(
        type="PathLayer",
        data=path_df,
        pickable=True,
        get_color="color",
        width_scale=20,
        width_min_pixels=2,
        get_path="path",
        get_width=5,
    )

    r = pdk.Deck(
        layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}"}
    )

    st.pydeck_chart(r)


def get_dataset(csv_name):
    dataset = pd.read_csv(csv_name, delimiter=";")
    dataset = dataset.drop_duplicates()

    # pick columns to use
    dataset["Latitude"] = dataset["Coordinates"].str.split(",").str[0].astype(float)
    dataset["Longitude"] = dataset["Coordinates"].str.split(",").str[1].astype(float)
    columns = [
        "Geoname ID",
        "Name",
        "Population",
        "Country name EN",
        "Latitude",
        "Longitude",
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
    lat1 = dataset_item["Latitude"]
    lon1 = dataset_item["Longitude"]
    city_max_dist = 0
    for _, row in dataset.iterrows():
        lat2 = row["Latitude"]
        lon2 = row["Longitude"]
        dist = get_dist(lat1, lon1, lat2, lon2)
        if dist > city_max_dist:
            city_max_dist = dist
    return city_max_dist


def get_city_item(name_or_id, dataset):
    def get_city_with_max_pop(subset):
        return subset[subset["Population"] == subset["Population"].max()].iloc[0]

    if isinstance(name_or_id, int):
        subset = dataset[dataset["Geoname ID"] == name_or_id]
    else:
        subset = dataset[dataset["Name"].str.contains(name_or_id) == True]
    return get_city_with_max_pop(subset)


def get_viability(city_1, city_2, dataset, dataset_statistics, A=1, B=1, verbose=False):
    min_pop, max_pop, avg_pop, min_dist, max_dist = (
        dataset_statistics["min_pop"],
        dataset_statistics["max_pop"],
        dataset_statistics["avg_pop"],
        dataset_statistics["min_dist"],
        dataset_statistics["max_dist"],
    )
    pop_1 = city_1["Population"]
    pop_2 = city_2["Population"]
    sum_pop = pop_1 + pop_2
    dist = get_dist(
        city_1["Latitude"], city_1["Longitude"], city_2["Latitude"], city_2["Longitude"]
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
        if row["Geoname ID"] != city["Geoname ID"]:
            viability = get_viability(city, row, dataset)
            if viability > max_viability:
                max_viability = viability
                max_viable_city = row
    return max_viable_city


def find_top_viable_cities(city_name, dataset, dataset_statistics, top_n=5, A=1, B=1):
    city = get_city_item(city_name, dataset)
    viabilities = []
    for _, row in dataset.iterrows():
        if row["Geoname ID"] != city["Geoname ID"]:
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
                    [city_1["Longitude"], city_1["Latitude"]],
                    [city_2["Longitude"], city_2["Latitude"]],
                ],
                "viability": f"{city[0] * 100}%",
                "city_2": city_2["Name"],
            }
        )
    red = Color("red")
    colors = list(red.range_to(Color("green"), len(paths_dict)))
    for i, path in enumerate(paths_dict):
        path["color"] = colors[i].hex_l
    return pd.DataFrame(paths_dict)


dataset = get_dataset("geonames.csv")
dataset_statistics = {
    "min_pop": dataset["Population"].min(),
    "max_pop": dataset["Population"].max(),
    "avg_pop": dataset["Population"].mean(),
    "min_dist": 0,
    "max_dist": get_max_dist(get_city_item("Atlanta", dataset), dataset),
}

# Text box input
user_city_input = st.text_input("Enter the origin city:", "Atlanta")
user_A_input = st.slider(
    "How important is the number of people connected?",
    min_value=0.0,
    max_value=5.0,
    step=0.1,
    value=1.0,
)
user_B_input = st.slider(
    "How important is a short distance?",
    min_value=0.0,
    max_value=5.0,
    step=0.1,
    value=1.0,
)
df1 = get_viable_cities_paths(
    user_city_input,
    dataset,
    dataset_statistics,
    5,
    A=float(user_A_input),
    B=float(user_B_input),
)
print(df1.head())

draw_all(df1)