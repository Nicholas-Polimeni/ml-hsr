from colour import Color
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from multiprocess import Pool

GEONAME_ID = "metro"
NAME = "metro"
POPULATION = "population"
COU_NAME_EN = "country_code"
LATITUDE = "latitude"
LONGITUDE = "longitude"
DATASET_CSV_PATH = "../data/metro_regions.csv"


def draw_all(dataset, dataset_statistics):
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
    path_df = get_viable_cities_paths(
        user_city_input,
        dataset,
        dataset_statistics,
        5,
        A=float(user_A_input),
        B=float(user_B_input),
    )

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
                "city_2": city_2[NAME],
                "viability": f"{city[0] * 100}%",
                "path": [
                    [city_1[LONGITUDE], city_1[LATITUDE]],
                    [city_2[LONGITUDE], city_2[LATITUDE]],
                ],
            }
        )
    red = Color("red")
    colors = list(red.range_to(Color("green"), len(paths_dict)))
    for i, path in enumerate(paths_dict):
        path["color"] = colors[i].hex_l
    return pd.DataFrame(paths_dict)


def get_most_viable_data(data_bundle):
    city_item, dataset, dataset_statistics = (
        data_bundle["city_item"],
        data_bundle["dataset"],
        data_bundle["dataset_statistics"],
    )
    max_viable_cities = find_top_viable_cities(
        city_item[NAME], dataset, dataset_statistics=dataset_statistics
    )
    return {"source": city_item, "max_viable_cities": max_viable_cities}


def get_most_viable_for_all(dataset, dataset_statistics):
    max_viable_cities = []
    with Pool(4) as p:
        max_viable_cities = p.map(
            get_most_viable_data,
            (
                {
                    "city_item": row,
                    "dataset": dataset,
                    "dataset_statistics": dataset_statistics,
                }
                for _, row in dataset.iterrows()
            ),
        )
    # for _, city_item in tqdm(dataset.iterrows(), total=len(dataset)):
    #     max_viable_cities.append(get_most_viable_data(city_item))
    return max_viable_cities


def get_best_of_best(dataset, dataset_statistics, top_n=5):
    max_viable_cities = get_most_viable_for_all(dataset, dataset_statistics)
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
    dataset = get_dataset(DATASET_CSV_PATH)
    dataset_statistics = {
        "min_pop": dataset[POPULATION].min(),
        "max_pop": dataset[POPULATION].max(),
        "avg_pop": dataset[POPULATION].mean(),
        "min_dist": 0,
        "max_dist": get_max_dist(get_city_item("Atlanta", dataset), dataset),
    }

    

    tab1, tab2 = st.tabs(["Math Model", "DL Model"])

    with tab1:
        st.title("Math Model")
        draw_all(dataset, dataset_statistics)

    with tab2:
        st.title("DL Model")
        bb = get_best_of_best(dataset, dataset_statistics, 5)
        print(bb)
