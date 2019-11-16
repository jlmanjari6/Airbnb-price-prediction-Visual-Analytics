var map, autocomplete, place;
var prevClusterId = -1, prevInfoWindow = -1, newSearch = true;
var marker_default, marker_cluster, marker_pin, marker_interest;
var clusterSet = new Set();
var clusterWiseMarker = {};

var initData = function() {
    console.log(window.location.pathname);
    marker_default = new google.maps.MarkerImage('../static/red-pin.png', new google.maps.Size(30, 30), null, null, new google.maps.Size(30, 30));
    marker_cluster = new google.maps.MarkerImage('../static/marker_cluster.png', new google.maps.Size(35, 35), null, null, new google.maps.Size(35, 35));
    marker_pin = new google.maps.MarkerImage('../static/blue-pin.png', new google.maps.Size(50, 50),null, null, new google.maps.Size(50, 50));
};

var initMap = function() {
    // New Brunswick lat long position
    initData();
    var initialLocation = {
        lat: 46.5653,
        lng: -66.4619
    };
    map = new google.maps.Map(
        document.getElementById('mapid'), {
            zoom: 7,
            streetViewControl: false,
            center: initialLocation
        });

    autocomplete = new google.maps.places.Autocomplete((
        document.getElementById('autocomplete')), {});
    autocomplete.addListener('place_changed', onPlaceChanged);

};

var onPlaceChanged = function() {
    place= autocomplete.getPlace();
    if (place.geometry) {
        map.panTo(place.geometry.location);
        map.setZoom(15);
        clearMarkers();
        createPlaceOfInterest(place.geometry.location);
    } else {
        document.getElementById('autocomplete').placeholder = 'Enter place of interest';
    }
};

var popContent = function(airbnbObj){
    return '<h5>'+airbnbObj.name+'</h5>'+
    '</br><span>Room type :</span>&nbsp;<span>'+airbnbObj.roomType+'</span>'+
    '</br><span>Distance :</span>&nbsp;<span>'+airbnbObj.distance+' km</span>'+
    '</br><span>Time taken :</span>&nbsp;<span>'+airbnbObj.timeTaken+' s</span>'+
    '</br><span>Price :</span>&nbsp;<span>'+airbnbObj.price+' CAD</span>'
};

// Sets the map on all markers in the array.
var clearMarkers = function () {
    if(newSearch){
        return;
    }
    clusterSet.forEach(clusterId=>{
        clusterWiseMarker[clusterId].forEach(marker => {
            marker.setMap(null);
        });
    });
    marker_interest.setMap(null);
}

var createPlaceOfInterest = function(placeObj){
    newSearch = false;
    marker_interest = new google.maps.Marker({
        position: {
            lat: placeObj.lat(),
            lng: placeObj.lng()
        },
        map: map,
        icon: marker_pin
    });

    $.get("http://127.0.0.1:5000/getData?lat="+placeObj.lat()+"&lng="+placeObj.lng(), function(data, status){
        clusterSet = new Set();
        clusterWiseMarker = {};

        var jsonData = JSON.parse(data);
        jsonData.forEach(data => {

            if (!clusterSet.has(data.cluster)) {
                clusterSet.add(data.cluster);
                clusterWiseMarker[data.cluster] = [];
            }
            var clusterId = data.cluster;

            var marker = new google.maps.Marker({
                position: {
                    lat: data.latitude,
                    lng: data.longitude
                },
                map: map,
                icon: marker_default
            });

            // highlight cluster on mouse click
            // remove previous cluster highlight and pop up.
            google.maps.event.addDomListener(marker, 'click', function() {
                if(prevClusterId !== -1){
                    clusterWiseMarker[prevClusterId].forEach(marker => {
                        marker.setIcon(marker_default);
                    });
                    prevInfoWindow.close();
                }

                // pop up
                var infowindow = new google.maps.InfoWindow({
                  content: popContent(data)
                });
                prevInfoWindow = infowindow;
                infowindow.open(map, marker);
                prevClusterId = clusterId;
                clusterWiseMarker[clusterId].forEach(marker => {
                    marker.setIcon(marker_cluster);
                });
                createAirbnbDetails(jsonData, clusterId);
            });

            clusterWiseMarker[data.cluster].push(marker);
        });
    });
};

var createAirbnbDetails = function(jsonData, clusterId){
    var airbnbs = jsonData.filter(data=>{
        return data.cluster === clusterId;
    });
    var clusterDetails = $('#cluster-details');
    airbnbs.forEach(airbnb=>{
        var div = document.createElement("DIV");
        var name = document.createElement("h6");
        var distance = document.createElement("p");
        var time = document.createElement("p");
        var price = document.createElement("p");

        distance.innerText = 'Distance :  '+ airbnb.distance + 'km';
        time.innerText = 'Time : '+ parseInt(airbnb.timeTaken/60) + 'min';
        price.innerText = 'Price :  '+ airbnb.price + ' CAD';
        name.innerText = airbnb.name;
        div.appendChild(name);
        div.appendChild(distance);
        div.appendChild(time);
        div.appendChild(price);
        clusterDetails.append(div);
    });
};

var loadUrl = function() {
    $('#googleMaps').attr('src', 'https://maps.googleapis.com/maps/api/js?key=' + googleMapAPIKey + '&libraries=places&callback=initMap');
};
