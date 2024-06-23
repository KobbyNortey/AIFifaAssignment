### FIFA Player Overall Prediction Web App

This project is a web application for predicting the overall rating of FIFA players using machine learning models. The application allows users to input player attributes and receive a predicted overall rating based on those attributes.

### Features

- Input player attributes such as age, height, weight, etc.
- Predict the overall rating of a player using machine learning models.

### Installation

1. Clone the repository: `git clone https://github.com/KobbyNortey/AIFifaAssignment`
2. Navigate to the project directory: `cd fifa-player-prediction`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Flask web application: `python app.py`
5. Open your browser and navigate to `http://localhost:5000` to use the web application.

### Usage

1. Enter the player attributes in the input form.
2. Click the "Predict" button to see the predicted overall rating.
3. Compare the predicted rating with the actual rating.

### Models Used

- K-Nearest Neighbors (KNN)
- Linear Regression
- Support Vector Regression (SVR)
- Random Forest Regression
- Gradient Boosting Regression

### File Descriptions

- `app.py`: Flask web application code.
- `templates/`: HTML templates for the web application.
- `static/`: Static files (e.g., CSS, JavaScript) for the web application.
- `Data/`: Folder containing the FIFA player datasets.
- `models/`: Saved machine learning models.
