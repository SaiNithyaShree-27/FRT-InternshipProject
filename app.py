import pandas as pd
from flask import Flask, request, render_template
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/check')
def check():
    return render_template('check.html')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

# Load the dataset
data = pd.read_csv('food_data.csv')

# Split the data into features (X) and target (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# Create an Extra Trees classifier with default parameters
food_model = ExtraTreesClassifier()

# Fit the model on the data
food_model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    blood_sugar =request.form['blood_sugar']
    food = food_model.predict([[blood_sugar]])
    food_list = food[0].split("\n")

    return render_template('predict.html', food=food_list)



if __name__ == '__main__':
    app.run(debug=True)
