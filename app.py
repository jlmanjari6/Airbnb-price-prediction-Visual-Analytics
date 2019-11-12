import json
import os

import folium
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from flask import Flask, render_template, request, jsonify
from folium.plugins import FastMarkerCluster

app = Flask(__name__)


if __name__ == '__main__':
    app.run(debug=True)


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


# **************************************************** Dashboard page **************************************************
@app.route('/', methods=['GET'])
def render_home():
    neighbourhood_groups = []
    neighborhoods = []

    df = get_preprocessed_df()
    lats2019 = df['latitude'].tolist()
    lons2019 = df['longitude'].tolist()
    locations = list(zip(lats2019, lons2019))
    map1 = folium.Map(location=[55.585901, -105.750596], zoom_start=3)
    FastMarkerCluster(data=locations).add_to(map1)
    map1.save(outfile="./static/dashboard-map2.html")
    return render_template('index.html', neighbourhood_groups=neighbourhood_groups, neighborhoods=neighborhoods)


# ****************************************************** Trends page ***************************************************
@app.route('/Trends', methods=['GET'])
def render_trends():
    provinces = []
    neighborhoods = []
    df = get_preprocessed_df()
    # fill drop downs for default option "New Brunswick"
    provinces = df.province.unique()
    x = "bedrooms"
    y = "room_type"
    dfg = df[df.province == provinces[0]]
    plot_generated = get_plot(df, dfg, x, y)
    return render_template('trends.html', provinces=provinces, plot=plot_generated)


def get_plot(df, dfg, x, y):
    y_values = list(getattr(df, y).unique())  # y
    x_values = list(getattr(df, x).unique())  # x
    dataa = []
    for val in y_values:
        # item = ""
        df_temp = dfg[(getattr(dfg, y)) == val]
        ls = []
        for r in x_values:
            ls.append(len(df_temp[getattr(dfg, x) == r]))
        item = go.Bar(name=str(val), x=x_values, y=ls)
        dataa.append(item)
    fig = go.Figure(data=dataa)
    y_title = 'Number of Airbnbs by ' + y
    plot_title = "Number of Airbnbs by " + x + " and " + y + " parameters of " + dfg.province[0]
    # Change the bar mode
    fig.update_layout(barmode='stack', title=plot_title, width=1100, height=500,
                      xaxis_title=x,
                      yaxis_title=y_title)
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json


@app.route('/GenerateChart', methods=['GET', 'POST'])
def generate_plot():
    selected_province = request.form.get('provinceID')
    x = request.form.get('xParams')
    y = request.form.get('yParams')
    df = get_preprocessed_df()
    provinces = df.province.unique()
    dfg = df[df.province == selected_province]
    plot_generated = get_plot(df, dfg, x, y)
    return render_template('trends.html', plot=plot_generated, provinces=provinces)


# ************************************************ Find Airbnbs page ***************************************************


@app.route('/FindAirbnbs', methods=['GET'])
def render_airbnbs():
    provinces = []
    neighborhoods = []
    df = get_preprocessed_df()
    # fill drop downs for default option "New Brunswick"
    provinces = df.province.unique()

    # filtering province
    p_df = df.loc[df.province == provinces[0]]
    neighborhoods = p_df.neighbourhood.unique()

    # filtering neighbourhood
    n_df = p_df.loc[p_df.neighbourhood == neighborhoods[0]]
    roomtypes = n_df.room_type.unique()

    # filtering room type
    r_df = n_df[(n_df.room_type == roomtypes[0])]

    generate_marker_latlong(r_df)
    return render_template('find-airbnbs.html', provinces=provinces, neighborhoods=neighborhoods, roomtypes = roomtypes)


def generate_marker_latlong(r_df):
    latitudes_list = list(r_df.latitude)
    longitudes_list = list(r_df.longitude)
    map2 = folium.Map(location=[latitudes_list[0], longitudes_list[0]])
    for index, row in r_df.iterrows():
        tooltip = 'Click here to know more!'
        pop_string = "<b>Name: </b>" + row['name'] + "<br><br>" + "<b>Minimum nights: </b>" + str(
            row['minimum_nights']) + "<br><br>" + "<b>Ratings: </b>" + str(
            row['ratings']) + "<br><br>" + "<b>Bedrooms: </b>" + str(
            row['bedrooms']) + "<br><br>" + "<b>Bathrooms: </b>" + str(
            row['bathrooms']) + "<br><br>" + "<b>Amenities: </b>" + row['amenities']
        folium.Marker([row['latitude'], row['longitude']], popup=pop_string, tooltip=tooltip).add_to(map2)
    map2.save(outfile="./static/find-airbnbs-map2.html")


@app.route('/GenerateMarkers', methods=['GET', 'POST'])
def generate_markers():
    df = get_preprocessed_df()

    selected_province = request.form.get('provinceID')
    selected_neighbourhood = request.form.get('neighbourhoods')
    selected_roomtype = request.form.get('roomtypes')

    provinces = df.province.unique()

    # filtering province
    p_df = df.loc[df.province == selected_province]
    neighborhoods = p_df.neighbourhood.unique()

    # filtering neighbourhood
    n_df = p_df.loc[p_df.neighbourhood == selected_neighbourhood]
    roomtypes = n_df.room_type.unique()

    # filtering room type
    r_df = n_df[(n_df.room_type == selected_roomtype)]

    generate_marker_latlong(r_df)
    return render_template('find-airbnbs.html', provinces=provinces, neighborhoods=neighborhoods, roomtypes = roomtypes)


@app.route('/get_neighbourhood/<provinceID>')
def get_neighbourhood(provinceID):
    print("enter")
    df = get_preprocessed_df()
    p_df = df.loc[df.province == provinceID]
    neighborhoods = list(p_df.neighbourhood.unique())
    return jsonify(neighborhoods)


@app.route('/get_roomtypes/<neighbourhoodID>')
def get_roomtypes(neighbourhoodID):
    print("enter")
    df = get_preprocessed_df()
    p_df = df.loc[df.neighbourhood == neighbourhoodID]
    roomtypes = list(p_df.room_type.unique())
    return jsonify(roomtypes)


# *********************************************** Predict price page ***************************************************


@app.route('/PricePrediction', methods=['GET'])
def render_price_prediction():

    return render_template('predict-price.html')



