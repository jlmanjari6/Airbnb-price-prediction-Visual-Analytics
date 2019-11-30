<h1> Visual Analytics Project -  Airbnb Price Prediction </h1>


<h1> Description </h1>

As part of this project, we aim to develop a web application where, users can get an efficient platform to know more about Airbnbs and get a predicted price based on the requirements provided by the user. Also, to show the performance of our price prediction model to the user to help perceive on how much he/she can rely on the predictions. More importantly, our web application aims to provide Airbnb recommendations to the user based on real time distance and time taken to travel to any place of interest in Canada. This way we are providing recommendations about the best and nearest Airbnb’s accessible to the user which benefits the users in more reliable way.


<h1> Technical stack </h1>

Python 3+, Flask, JQuery, Bootstrap


<h1> Functionalities </h1>

<ul>
  <li>Exploratory data analysis to show trends </li>
  <li>Airbnb price prediction - Machine learning with XGBoost</li>
  <li>Airbnb recommendations - Clustering with K-Means</li>
</ul>

<h1> Steps involved </h1>
<ul>
  <li> Data cleaning and preprocessing </li>
  <li> Machine learning </li>
  <li> Evaluation of models </li>
  <li> Application development </li>
  <li> Visualization </li>
  <li> Testing </li>
</ul>

<h1> Modules developed </h1>

<b> Trends – Number of Airbnbs:</b>
<p>In this module, we are providing stacked bar charts to the user to view total number of Airbnbs for the selected user inputs. Each bar chart holds number of Airbnbs on the y-axis and user may select one parameter for x-axis, and desired category required to segment each bar to represent different types of selected category with different colors. The category for segmentation of bars and parameter for x-axis might be one among the 4 given options - bedrooms, room type, neighborhood, number of Airbnbs available in next 30 days. In this way, this module lets the user to select 16 different combinations producing 16 different bar charts. <p>
<p>We chose this visualization module because user can be able to see the number of Airbnbs available in a desired province based on various combinations of parameters through one single visualization. From the image below, it can be inferred from the tooltip that there are 12 Airbnbs of private room type in New Brunswick with an availability of 12 days. This Plotly chart allows the user to zoom any particular point in the bar chart, slide the chart horizontally and vertically to see the results clearly.</p>

<b> Dashboard – Airbnb clusters: </b>
<p>In this module, we are providing the map of Canada with Airbnbs clustered based on zoom level of the map. The Airbnbs are clustered irrespective of the province, neighborhood or any other parameter and user can be able to see the Airbnb markers all over the Canada using zoom functionality. As the user zooms into the map, Airbnbs at that level keeps decluttered until a point where single Airbnb markers are visible. </p>
<p>We chose this visualization module as a landing page to the application and as an overview of the locations of Airbnbs. Through this module, user can see Airbnb locations all over the Canada. </p>

<b>Find Airbnbs – Airbnbs on map</b>
<p>In this module, we are providing a map with markers of Airbnbs of selected room type in a neighborhood of desired province. Based on the  user choice, the map is displayed with markers along with tool tip saying “Click here to know more” and on click of any particular marker, a pop up will be displayed on the marker with the details of that Airbnb such as name, minimum nights to stay, ratings, number of bedrooms, number of bathrooms and the amenities being provided by the Airbnb. </p>
<p>We chose this visualization module because user might want to see all the locations of Airbnbs of one room type that exist in a desired neighborhood of a province. With this module, we are allowing the user to see all the Airbnbs that exist in a selected location along with the details of each Airbnb which will be displayed on click of each marker icon.</p>

<b>Predict price – price prediction </b>
<p>In this module, we are providing the predicted price to the user by taking the user inputs into consideration. The user will be provided with 6 parameters among which the parameters province, neighborhood, room type and accommodates are mandatory whereas the parameters bedrooms and minimum number of nights to stay are optional. Based on the user inputs, we train the machine learning model dynamically and display the predicted price to the user as shown in the image below. Along with the predicted price for the selected combination of inputs, we also display two bar charts – for neighborhood and for room type which are explained as follows. For the same set of parameters i.e., same province, neighborhood, accommodates, minimum nights to stay (if selected), bedrooms (if selected), we display a bar chart with predicted prices of other room types along with selected room type. Similarly, we display a bar chart with at most 10 neighborhoods which are in the predicted price range assuming all the other parameters are selected as it is. Apart from the predictions, for the user to gain some insight into the machine learning, we provided an accordion on clicking which two plots – Actual Vs Predicted price scatter plot with correlation line and feature importance bar chart of selected parameters with their contributions towards the prediction, appear. </p>
<p>This module is developed completely for dynamic visualizations including dynamic selection of parameters to train the model, predicted price based on the user inputs, two bar charts with predicted prices for other neighborhoods of similar price range and other room types, and to know about machine learning model performance, two types of plots - correlation scatter plot of Actual Vs Predicted price and feature importance bar chart. Through this module, we not only provide the predicted price of selected inputs, but also intend to provide some recommendations about other room types and neighborhoods with similar price range to make it flexible to the user. We also intend to provide insights into machine learning to perceive how far the user can rely on the predictions. By making the visualizations dynamic, we intended to provide flexibility to the user to train the model with number of parameters of his/her choice rather than keeping them constant.</p>

<b>Clustering – Airbnb clusters and recommendations:</b>
<p>The workflow of giving recommendations to user begins with selecting a place of interest from the user. We have given the users an option to search and select their place of interest. Google maps search API is used to serve this purpose. A blue marker is placed on the map to indicate the place of interest of the user. To find the real-time distance and time taken to reach the Airbnb from point of interest, we have used Google Distance Matrix API. This Google API by accepting multiple sources and multiple destinations, returns the real-time distance and time taken to reach the place of interest from each Airbnb. There are some advanced options in these APIs with using which we can calculate the traffic conditions and select transport options like bus. We intend to use these advanced features as part of our future work.</p>
<p>Once the clustering is completed, the data is sent to the user interface of the application. Currently, we are showing nearest 50 Airbnbs to the user. As part of future work, we intend to make this configurable. We have used Google maps to plot the nearest Airbnbs and some recommendations. The 50 nearest Airbnbs are listed as red markers on the Google map. Once the user selects any Airbnb, all the other Airbnbs that are similar to the one selected (belongs to one particular cluster) will be highlighted with green markers as shown in the above image. Also, a summary of the recommended Airbnbs will be displayed on the right-hand side of the screen. The summary includes the name of the Airbnb, distance from the place of interest, time taken to reach the Airbnb and price of stay. The user can scroll through the recommendations and select any Airbnb to get it highlighted on the screen instead of manually selecting every Airbnb for information.</p>

<h1> Future work </h1>
<ul>
<li>As of now, we have calculated the distance and time taken assuming the user uses their own transportation. If the user wants to use public transport or to go by walk, there must be an option to prefer distance over time taken. To serve this purpose, we would like to provide an option to the user in future to choose distance, time taken or both as inputs to clustering.</li>
<li>In future, we also intend to include all the provinces in Canada for our application. The code is implemented in such a way that it does not require much changes to accommodate more data. </li>
<li>	We would also like to depict more trends on the price variations among different provinces/neighborhoods. </li>
<li>	Finally, we also intend to incorporate Airbnb trends by considering exact date of accommodation, real time traffic and navigation.</li>
  </ul>







