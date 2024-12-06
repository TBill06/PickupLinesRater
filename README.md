# Pickup Line Rating App

This project is a web application that rates pickup lines using various machine learning and rule-based systems. The application is built using Flask, Python for the backend and React for the frontend. It is used to benchmark 6 NLP systems including Bag of Words, VADER, Naive Bayes, Markov Chains, BERT, and GPT.

## Features

- Rate a pickup line using a selected rating system.
- Get ratings from all available systems and display them in a bar chart.
- Models are trained on a HuggingFace flirty_or_not Dataset.

## Technologies Used

- **Backend**: Flask, Python 
- **Frontend**: React, Axios, Chart.js
- **Machine Learning Libraries**: scikit-learn, transformers, vaderSentiment, torch, datasets

## Setup Instructions

### Backend

1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Flask server**:
    ```bash
    python server.py
    ```

### Frontend

1. **Navigate to the frontend directory**:
    ```bash
    cd frontend
    ```

2. **Install the required packages**:
    ```bash
    npm install
    ```

3. **Run the React application**:
    ```bash
    npm start
    ```

## Usage

1. **Start the backend server**:
    ```bash
    python server.py
    ```

2. **Start the frontend application**:
    ```bash
    npm start
    ```

3. **Open your browser and navigate to** `http://localhost:3000`.

4. **Enter a pickup line** or select one from the dropdown.

5. **Select a rating system** and click on "Get Rating" to get the rating for the selected system.

6. **Click on "Get All Ratings and Plot"** to get ratings from all systems and display them in a bar chart.

## Contributing

Feel free to contribute to this project by submitting a pull request or opening an issue.

## License

This project is licensed under the MIT License.
