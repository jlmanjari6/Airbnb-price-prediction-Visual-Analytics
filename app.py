from flask import Flask, render_template, request
from flask import send_file
import numpy as np
import pandas as pd
import folium
import os
from folium.plugins import FastMarkerCluster
from branca.colormap import LinearColormap

app = Flask(__name__)


@app.route('/', methods=['GET'])
def render_home():
    neighbourhood_groups = []
    neighborhoods = []
    df_nb = pd.read_csv('listings_NB.csv')
    # preprocess default option "New Brunswick"
    df_nb = get_preprocessed_dfnb(df_nb)
    # fill drop downs for default option "New Brunswick"
    neighbourhood_groups = df_nb.neighbourhood_group.unique()
    neighborhoods = (df_nb.loc[df_nb['neighbourhood_group'] == neighbourhood_groups[0], 'neighbourhood']).unique()

    df = get_preprocessed_df();
    lats2019 = df['latitude'].tolist()
    lons2019 = df['longitude'].tolist()
    locations = list(zip(lats2019, lons2019))
    map1 = folium.Map(location=[55.585901, -105.750596], zoom_start=2.25)
    FastMarkerCluster(data=locations).add_to(map1)
    map1.save(outfile="./templates/dashboard-map1.html")
    return render_template('index.html', neighbourhood_groups=neighbourhood_groups, neighborhoods=neighborhoods)


@app.route('/templates/dashboard-map1.html')
def show_map():
    return send_file(os.path.join('./templates/', "dashboard-map1.html"))


@app.route('/Trends', methods=['GET'])
def render_trends():
    provinces = []
    neighborhoods = []
    df = get_preprocessed_df();
    # fill drop downs for default option "New Brunswick"
    provinces = df.province.unique()
    neighborhoods = (df.loc[df['province'] == provinces[0], 'neighbourhood']).unique()
    print(provinces[0])
    # room_types = df.room_type.unique()
    # minimum_nights = df.minimum_nights.unique()
    # availability_30 = df.availability_30.unique()
    # ratings = df.ratings.unique()
    # bedrooms = df.bedrooms.unique()
    return render_template('trends.html', provinces=provinces, neighborhoods=neighborhoods)


def get_preprocessed_dfnb(temp_df):
    ndf = temp_df.drop(['host_id', 'host_name', 'last_review', 'calculated_host_listings_count', 'reviews_per_month'],
                       axis=1)
    return ndf


def get_preprocessed_df():
    df_nb = pd.read_csv("NewBrunswick.csv")
    df_nb['province'] = "New Brunswick"
    df_bc = pd.read_csv("Victoria_BC.csv")
    df_bc['province'] = "British Columbia"
    df_tr = pd.read_csv("Toronto.csv")
    df_tr['province'] = "Ontario"
    df = pd.concat([df_nb, df_bc, df_tr])
    df = df[
        ['id', 'name', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'latitude', 'longitude', 'room_type',
         'minimum_nights', 'availability_30', 'availability_365', 'review_scores_rating', 'accommodates', 'bathrooms',
         'bedrooms', 'beds', 'amenities', 'province', 'price']]
    df = df.rename(
        columns={"neighbourhood_cleansed": "neighbourhood", "neighbourhood_group_cleansed": "neighbourhood_group",
                 "review_scores_rating": "ratings"})
    df = df.dropna(axis=0, subset=['bedrooms', 'bathrooms', 'beds'])

    df['ratings'] = df['ratings'].replace(0, np.NaN)
    df['ratings'] = df.groupby(['neighbourhood', 'room_type'])['ratings'].transform(lambda x: x.fillna(x.mean()))
    df = df[pd.notna(df['ratings'])]
    df['price'] = df['price'].str.replace('$', '')
    df['price'] = df['price'].str.replace(',', '')
    df["price"] = pd.to_numeric(df["price"])
    df.bedrooms = df.bedrooms.replace({0: np.NaN})
    df['bedrooms'] = df.groupby(['room_type', 'beds'])['bedrooms'].transform(lambda x: x.fillna(x.mean()))
    df = df[pd.notna(df['bedrooms'])]
    df = df.astype({"bedrooms": int})
    return df


@app.route('/PredictPrice', methods=['GET', 'POST'])
def predict_price():
    selected_province = request.form.get('provinceID')
    print(selected_province)


if __name__ == '__main__':
    app.run(debug=True)
