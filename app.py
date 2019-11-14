import json

import folium
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from flask import Flask, render_template, request, jsonify
from folium.plugins import FastMarkerCluster
from sklearn import preprocessing
import xgboost as xgb


app = Flask(__name__)


if __name__ == '__main__':
    app.run(debug=True)


class MainClass:
    def __init__(self):
        self.cleaned_df = pd.read_csv("cleaned_df.csv")


class PricePrediction:
    def __init__(self):
        self.preprocessed_df = pd.read_csv("preprocessed_df.csv")


main_class_obj = MainClass()
# **************************************************** Dashboard page **************************************************
@app.route('/', methods=['GET'])
def render_home():
    neighbourhood_groups = []
    neighborhoods = []

    df = main_class_obj.cleaned_df
    lats2019 = df['latitude'].tolist()
    lons2019 = df['longitude'].tolist()
    locations = list(zip(lats2019, lons2019))
    map1 = folium.Map(location=[55.585901, -105.750596], zoom_start=3.3)
    FastMarkerCluster(data=locations).add_to(map1)
    map1.save(outfile="./static/dashboard-map5.html")
    return render_template('index.html', neighbourhood_groups=neighbourhood_groups, neighborhoods=neighborhoods)


# ****************************************************** Trends page ***************************************************
@app.route('/Trends', methods=['GET'])
def render_trends():
    df = main_class_obj.cleaned_df
    # fill drop downs for default option "New Brunswick"
    provinces = df.province.unique()
    x = "bedrooms"
    y = "room_type"
    dfg = df[df.province == provinces[0]]
    plot_generated = get_plot(df, dfg, x, y, provinces[0])
    return render_template('trends.html', provinces=provinces, plot=plot_generated)


def get_plot(df, dfg, x, y, selected_province):
    y_values = list(getattr(df, y).unique())
    x_values = list(getattr(df, x).unique())
    dataa = []
    for val in y_values:
        df_temp = dfg[(getattr(dfg, y)) == val]
        ls = []
        for r in x_values:
            ls.append(len(df_temp[getattr(dfg, x) == r]))
        item = go.Bar(name=str(val), x=x_values, y=ls)
        dataa.append(item)
    fig = go.Figure(data=dataa)
    y_title = 'Number of Airbnbs by ' + y
    plot_title = "Number of Airbnbs by " + x + " and " + y + " parameters of " + selected_province
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
    df = main_class_obj.cleaned_df
    provinces = df.province.unique()
    dfg = df[df.province == selected_province]
    plot_generated = get_plot(df, dfg, x, y, selected_province)
    return render_template('trends.html', plot=plot_generated, provinces=provinces, sel_x=x, sel_y=y ,
                           sel_province=selected_province)


# ************************************************ Find Airbnbs page ***************************************************


@app.route('/FindAirbnbs', methods=['GET'])
def render_airbnbs():
    provinces = []
    neighborhoods = []
    df = main_class_obj.cleaned_df
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

    response_map = generate_marker_latlong(r_df)
    return render_template('find-airbnbs.html', provinces=provinces, neighborhoods=neighborhoods, roomtypes = roomtypes
                           , response_map=response_map)


def generate_marker_latlong(r_df):
    latitudes_list = list(r_df.latitude)
    longitudes_list = list(r_df.longitude)
    map2 = folium.Map(location=[latitudes_list[0], longitudes_list[0]] , zoom_start='12', width='100%',height='75%')
    for index, row in r_df.iterrows():
        tooltip = 'Click here to know more!'
        pop_string = "<b>Name: </b>" + row['name'] + "<br><br>" + "<b>Minimum nights: </b>" + str(
            row['minimum_nights']) + "<br><br>" + "<b>Ratings: </b>" + str(
            row['ratings']) + "<br><br>" + "<b>Bedrooms: </b>" + str(
            row['bedrooms']) + "<br><br>" + "<b>Bathrooms: </b>" + str(
            row['bathrooms']) + "<br><br>" + "<b>Amenities: </b>" + row['amenities']
        folium.Marker([row['latitude'], row['longitude']], popup=pop_string, tooltip=tooltip).add_to(map2)
    return map2


@app.route('/GenerateMarkers', methods=['GET', 'POST'])
def generate_markers():
    output = get_filtered_df()
    r_df = output[0]
    provinces = output[1]
    neighborhoods = output[2]
    roomtypes = output[3]
    response_map = generate_marker_latlong(r_df)
    return render_template('find-airbnbs.html', provinces=provinces, neighborhoods=neighborhoods, roomtypes = roomtypes
                           , response_map=response_map)


def get_filtered_df():
    df = main_class_obj.cleaned_df

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
    return r_df, provinces, neighborhoods, roomtypes


@app.route('/get_neighbourhood/<provinceID>')
def get_neighbourhood(provinceID):
    df = main_class_obj.cleaned_df
    p_df = df.loc[df.province == provinceID]
    neighborhoods = list(p_df.neighbourhood.unique())
    return jsonify(neighborhoods)


@app.route('/get_roomtypes/<neighbourhoodID>')
def get_roomtypes(neighbourhoodID):
    df = main_class_obj.cleaned_df
    p_df = df.loc[df.neighbourhood == neighbourhoodID]
    roomtypes = list(p_df.room_type.unique())
    return jsonify(roomtypes)


# *********************************************** Predict price page ***************************************************


@app.route('/PricePrediction', methods=['GET'])
def render_price_prediction():
    price_prediction_obj = PricePrediction()
    ndf = price_prediction_obj.preprocessed_df
    provinces = ndf.province.unique()
    accommodates = sorted(ndf.accommodates.unique())
    bedrooms = sorted(ndf.bedrooms.unique())
    minimum_nights = sorted(ndf.minimum_nights.unique())
    return render_template('predict-price.html', provinces=provinces, accommodates=accommodates, bedrooms=bedrooms,
                           minimum_nights=minimum_nights, plot='NA',price='')


@app.route('/PredictPrice', methods=['GET', 'POST'])
def predict_price():
    # to refill the dropdown values
    price_prediction_obj = PricePrediction()
    ndf = price_prediction_obj.preprocessed_df
    provinces = ndf.province.unique()
    accommodates = sorted(ndf.accommodates.unique())
    bedrooms = sorted(ndf.bedrooms.unique())
    minimum_nights = sorted(ndf.minimum_nights.unique())

    # to predict the price
    selected_province = request.form.get('provinceID')
    selected_neighbourhood = request.form.get('neighbourhoodID')
    selected_roomtype = request.form.get('roomtypeID')
    selected_accommodates = request.form.get('accommodateID')
    selected_bedrooms = request.form.get('bedroomsID')
    selected_min_nights = request.form.get('minimumID')

    # filtering province
    p_df = ndf.loc[ndf.province == selected_province]
    neighborhoods = p_df.neighbourhood.unique()
    sel_neighbourhood = selected_neighbourhood

    # filtering neighbourhood
    n_df = p_df.loc[p_df.neighbourhood == selected_neighbourhood]
    roomtypes = n_df.room_type.unique()
    sel_roomtype = selected_roomtype

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(ndf['province'])
    le_province_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    selected_province = le_province_mapping[str(selected_province)]
    ndf['province'] = label_encoder.transform(ndf['province'])

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(ndf['room_type'])
    le_room_type_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    selected_roomtype = le_room_type_mapping[selected_roomtype]
    ndf['room_type'] = label_encoder.transform(ndf['room_type'])

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(ndf['neighbourhood'])
    le_neighbourhood_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    selected_neighbourhood = le_neighbourhood_mapping[selected_neighbourhood]
    ndf['neighbourhood'] = label_encoder.transform(ndf['neighbourhood'])

    data = {}
    tdf = pd.DataFrame(data)

    neighbourhood_list = (ndf[ndf.province == selected_province].neighbourhood.unique())
    for val in neighbourhood_list:
        data = {'neighbourhood': [val], 'province': [selected_province], 'room_type': [selected_roomtype],
                'accommodates': [int(selected_accommodates)]}
        tdf = tdf.append(pd.DataFrame(data), ignore_index=True)

    sel_min = ''
    sel_bedrooms = ''
    if selected_bedrooms != 'b0' and selected_min_nights != 'm0':
        tdf['minimum_nights'] = int(selected_min_nights)
        tdf['bedrooms'] = int(selected_bedrooms)
        sel_min = int(selected_min_nights)
        sel_bedrooms = int(selected_bedrooms)

    elif selected_bedrooms != 'b0' and selected_min_nights == 'm0':
        tdf['bedrooms'] = int(selected_bedrooms)
        sel_bedrooms = int(selected_bedrooms)

    elif selected_bedrooms == 'b0' and selected_min_nights != 'm0':
        tdf['minimum_nights'] = int(selected_min_nights)
        sel_min = int(selected_min_nights)

    xgb_reg = xgb.XGBRegressor(max_depth= 5, min_child_weight=24)
    xgb_reg.fit(ndf[tdf.columns], ndf.price)
    tdf['price'] = xgb_reg.predict(tdf)
    output = tdf.loc[tdf.neighbourhood == selected_neighbourhood, 'price'].iloc[0]
    ls1 = (tdf[tdf.price <= output]).nlargest(5, "price")
    ls2 = (tdf[tdf.price > output]).nsmallest(5, "price")
    ls = ls1.append(ls2)
    recommended_neighbourhoods = {}
    for row in ls.iterrows():
        for key, value in le_neighbourhood_mapping.items():
            if value == int(row[1].neighbourhood):
                recommended_neighbourhoods[key] = round(row[1].price, 2)

    plot_generated = get_price_plots(recommended_neighbourhoods)
    return render_template('predict-price.html', provinces=provinces, neighbourhoods=neighborhoods,
                           roomtypes=roomtypes, accommodates=accommodates, bedrooms=bedrooms,
                           minimum_nights=minimum_nights, price=round(output,2),
                           recommended_neighbourhoods=recommended_neighbourhoods, plot=plot_generated,
                           sel_province=request.form.get('provinceID'),
                           sel_neighbourhood=sel_neighbourhood, sel_roomtype=sel_roomtype,sel_min=sel_min,
                           sel_bedrooms=sel_bedrooms, sel_accommodates=int(request.form.get('accommodateID')))


def get_price_plots(recommended_neighbourhoods):
    x = list(recommended_neighbourhoods.keys())
    y = list(recommended_neighbourhoods.values())
    fig = go.Figure(data=[go.Bar(
            x=x, y=y,
            width=0.7,
            text=y,
            textposition='auto',
    )])
    fig.update_traces(marker_color='#C01A4A', marker_line_color='rgb(8,48,107)',
                      marker_line_width=0.3, opacity=0.8)
    fig.update_layout(title="Other neighbourhoods of similar price range:", width=1100, height=600,
                      xaxis_title="Neighbourhood",
                      yaxis_title="Price")
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json



