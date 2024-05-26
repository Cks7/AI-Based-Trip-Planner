from flask import Flask, render_template, request, redirect, session, url_for
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances, cosine_similarity
from math import radians

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the combined dataset
combined_data = pd.read_csv('dataset_without_duplicates.csv')

# Remove duplicate entries
combined_data.drop_duplicates(inplace=True)

# Mock user database (you can replace this with a real database)
users = {'user1': 'password1', 'user2': 'password2'}

def calculate_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dist = haversine_distances([[lat1, lon1], [lat2, lon2]]) * 6371000  # Earth radius in meters
    return dist[1, 0]

def get_recommendations(filtered_data, selected_restaurant):
    filtered_data['distance_to_restaurant'] = filtered_data.apply(
        lambda row: calculate_distance(
            selected_restaurant['Latitude_x__Restaurant'],
            selected_restaurant['Longitude_x__Restaurant'],
            row['Latitude_Hotel'],
            row['Longitude_Hotel']
        ),
        axis=1
    )

    ranked_hotels = filtered_data[filtered_data['Hotel_name'].notnull()].sort_values(
        by=['distance_to_restaurant', 'mmt_review_score_Hotel', 'hotel_star_rating_Hotel']
    )

    nearest_hotel = ranked_hotels.iloc[0].to_dict() if not ranked_hotels.empty else None

    filtered_data['distance_to_restaurant'] = filtered_data.apply(
        lambda row: calculate_distance(
            selected_restaurant['Latitude_x__Restaurant'],
            selected_restaurant['Longitude_x__Restaurant'],
            row['Latitude_place_0_x'],
            row['Longitude_place_0_x']
        ),
        axis=1
    )

    ranked_places = filtered_data[filtered_data['Name_Place'].notnull()].sort_values(by='distance_to_restaurant')

    nearest_place = ranked_places.iloc[0].to_dict() if not ranked_places.empty else None

    hotel_profiles = filtered_data[['mmt_review_score_Hotel', 'hotel_star_rating_Hotel', 'budget_level']]
    place_profiles = filtered_data[['Rating_Place', 'Ratings_out_of_5_Restaurant', 'budget_level']]

    selected_hotel_profile = selected_restaurant[['mmt_review_score_Hotel', 'hotel_star_rating_Hotel', 'budget_level']]

    hotel_similarity = cosine_similarity([selected_hotel_profile], hotel_profiles)
    place_similarity = cosine_similarity([selected_hotel_profile], place_profiles)

    top_hotels = ranked_hotels.iloc[hotel_similarity.argsort()[0][::-1][:4]].to_dict(orient='records')
    top_places = ranked_places.iloc[place_similarity.argsort()[0][::-1][:4]].to_dict(orient='records')

    return nearest_hotel, nearest_place, top_hotels, top_places

def get_budget_level(budget):
    if 0 < budget < 1000:
        return 0
    if budget <= 1000:
        return 1
    elif 1000 < budget <= 2000:
        return 2
    elif 2000 < budget <= 3000:
        return 3
    elif 3000 < budget <= 4000:
        return 4
    else:
        return 5

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_budget = int(request.form['budget'])
        user_hotel_rating = int(request.form['hotel_rating'])
        user_hotel_star_rating = int(request.form['hotel_star_rating'])
        user_restaurant_rating = int(request.form['restaurant_rating'])
        num_days = int(request.form['num_days'])

        budget_level = get_budget_level(user_budget)

        filtered_data = combined_data[
            (combined_data['budget_level'] == budget_level) &
            (combined_data['mmt_review_score_Hotel'] >= user_hotel_rating) &
            (combined_data['hotel_star_rating_Hotel'] == user_hotel_star_rating) &
            (combined_data['Ratings_out_of_5_Restaurant'] >= user_restaurant_rating)
        ]

        if filtered_data.empty:
            return render_template('index.html', error="No suitable recommendations found based on the provided criteria.")

        recommendations = []
        for day in range(1, num_days + 1):
            selected_restaurant = filtered_data.sample(n=1).iloc[0]
            nearest_hotel, nearest_place, top_hotels, top_places = get_recommendations(filtered_data, selected_restaurant)
            recommendations.append({
                'day': day,
                'restaurant': selected_restaurant.to_dict(),
                'nearest_hotel': nearest_hotel,
                'nearest_place': nearest_place,
                'top_hotels': top_hotels,
                'top_places': top_places
            })

        return render_template('recommendations.html', recommendations=recommendations)

    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return render_template('register.html', error='Username already exists.')
        users[username] = password
        return redirect(url_for('login'))
    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)
