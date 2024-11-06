import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import requests
import pickle
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import genetic
import matplotlib as plt

# Database Initialization and Data Loading
def init_db():
    conn = sqlite3.connect('bitcoin_prices.db', check_same_thread=False)
    return conn

def load_training_data():
    conn = init_db()
    # Load the initial CSV data into the database if it's empty
    df = pd.read_csv('preprocessed_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.to_sql('bitcoin_prices', conn, if_exists='replace', index=False)
    return "Training data loaded into database successfully!"

def fetch_bitfinex_data(symbol='BTCUSD', limit=30):
    url = f'https://api-pub.bitfinex.com/v2/candles/trade:1D:t{symbol}/hist'
    params = {'limit': limit, 'sort': -1}  # Fetch latest data
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise ValueError("Failed to fetch data from Bitfinex API.")
    
    data = response.json()
    
    # Convert data to DataFrame and rename columns to match the database
    df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'Price', 'High', 'Low', 'Vol'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.date
    df = df[['Date', 'Price', 'Open', 'High', 'Low', 'Vol']].sort_values(by='Date')
    print(df)
    return df

def update_database(conn, data_df):
    
    latest_date_in_db = pd.read_sql('SELECT MAX(Date) as last_date FROM bitcoin_prices', conn)['last_date'][0]
    latest_date_in_db = pd.to_datetime(latest_date_in_db).date() if latest_date_in_db else None
    
    if latest_date_in_db:
        delete_query = "DELETE FROM bitcoin_prices WHERE Date = (SELECT MAX(Date) FROM bitcoin_prices)"
        conn.cursor().execute(delete_query)

    new_data = data_df[data_df['Date'] >= latest_date_in_db] if latest_date_in_db else data_df
    print(new_data)
    if not new_data.empty:
        new_data.to_sql('bitcoin_prices', conn, if_exists='append', index=False)
        print(f"Added {len(new_data)} new rows to the database.")
    else:
        print("No new data to add. Replaced existing data with the last row.")

def load_model(csv_file):
    models = []
    with open(csv_file, 'rb') as f:
        resultof = pickle.load(f)
    for i in range(10):
        models.append(resultof[-10 + i][1])

    best_model = models.pop(6)
    models.insert(0, best_model) 
    return models


def load_model_non(csv_file):
    with open(csv_file, 'rb') as f:  # Load finalResult to variable
        resultof = pickle.load(f)
    resultof = [resultof[i:i + 6] for i in range(0, len(resultof), 6)]

    old_models = resultof[-10:]
    models = []

    for i in range(len(old_models)):
        model = Sequential()
        cLayer = old_models[i][1][0]
        cNeurons = old_models[i][1][3]
        cRecurDrop = old_models[i][1][2]
        cDroprate = old_models[i][1][1]
        weights = old_models[i][2]

        for i in range(cLayer):
            return_sequences = True if i < cLayer - 1 else False
            neurons = cNeurons[i] if i < len(cNeurons) else random.choice(cNeurons)
            recurrent_dropout = cRecurDrop[i] if i < len(cRecurDrop) else random.choice(cRecurDrop)
            dropout_rate = cDroprate[i] if i < len(cDroprate) else random.choice(cDroprate)

            if i == 0:
                model.add(LSTM(neurons,
                                return_sequences=return_sequences,
                                recurrent_dropout=recurrent_dropout,
                                input_shape=(30, 5)))
            else:
                model.add(LSTM(neurons,
                                return_sequences=return_sequences,
                                recurrent_dropout=recurrent_dropout))
            model.add(Dropout(dropout_rate))

        model.add(Dense(25))
        model.add(Dense(5))

        model.build()

        model.set_weights(weights)

        models.append(model)
    
    best_model = models.pop(7)
    models.insert(0, best_model) 

    return models

def load_models(txtfile):
    with open(txtfile, 'rb') as f:  # Load finalResult to variable
        old_models = pickle.load(f)

    models = []

    for i in range(len(old_models)):
        model = Sequential()
        cLayer = old_models[i][1][0]
        cNeurons = old_models[i][1][3]
        cRecurDrop = old_models[i][1][2]
        cDroprate = old_models[i][1][1]
        weights = old_models[i][2]

        for i in range(cLayer):
            return_sequences = True if i < cLayer - 1 else False
            neurons = cNeurons[i] if i < len(cNeurons) else random.choice(cNeurons)
            recurrent_dropout = cRecurDrop[i] if i < len(cRecurDrop) else random.choice(cRecurDrop)
            dropout_rate = cDroprate[i] if i < len(cDroprate) else random.choice(cDroprate)

            if i == 0:
                model.add(LSTM(neurons,
                                return_sequences=return_sequences,
                                recurrent_dropout=recurrent_dropout,
                                input_shape=(30, 5)))
            else:
                model.add(LSTM(neurons,
                                return_sequences=return_sequences,
                                recurrent_dropout=recurrent_dropout))
            model.add(Dropout(dropout_rate))

        model.add(Dense(25))
        model.add(Dense(5))

        model.build()

        model.set_weights(weights)

        models.append(model)
        
    best_model = models.pop(7)
    models.insert(0, best_model) 

    return models



def prepare_data(data, txtfile):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    x, _ = genetic.create_sequences(scaled, 30)
    models = load_models(txtfile)
    # try:
        
    # except:
    #     models = load_model_non(txtfile)

    return x, models, scaler

@st.cache_data
def predict(data, txtfile, steps_ahead=14):
    
    predictions_all = []
    params = []

    x, models, scaler = prepare_data(data, txtfile)
    for i in range(len(models)):
        params.append(genetic.getParams(models[i]))

    for model in models:
        predictions = []
        current_input = x[-2].reshape(1, 30, 5)  # Initial input for the model

        for _ in range(steps_ahead):
            prediction = model.predict(current_input)  # Get all 5 features
            predictions.append(prediction[0])  # Store all features
            current_input = np.concatenate((current_input[:, 1:, :], prediction.reshape(1, 1, 5)), axis=1)  # Update input with all features

        # Convert predictions to a numpy array for inverse transformation
        predictions_actual = scaler.inverse_transform(np.array(predictions))
        
        # If you want to extract just one feature, e.g., the first one, you can do so like this:
        predicted_prices = predictions_actual[:, 0]
        print("SALAWMO")
        
        predictions_all.append(predicted_prices)

    df_pred = pd.DataFrame(predictions_all).transpose()
    df_pred.columns = [f'M{i + 1}' for i in range(len(models))]
    print("FINISH")

    return df_pred, params


# Streamlit App Layout

##SHOW PARAMETER FOR EACH MODEL
##NEW MODELS WITH  BITFNIEX
##USE SLIDER INSTEAD OF DROPDOWN FOR TIMEFRAME
#PREDICIOTN AVERAGE (PROBABILITY) INCREASE/DECREASE
#CREDIT TO ME 

#IMPORTANT
##SOLIDIFY HOW CURRENT PRICE WORKS AND PREDICITION BASED ON THAT



# Calculate increase or decrease for next-day prediction
def calculate_price_change(current, predicted):
    difference = predicted - current
    percentage_change = (difference / current) * 100
    return difference, abs(difference), percentage_change

# Streamlit App Layout

# def main():
#     conn = init_db()

#     cursor = conn.cursor()

#     cursor.execute("DROP TABLE bitcoin_prices")
#     load_training_data()


def main():
    conn = init_db()
    new_data = fetch_bitfinex_data()
    update_database(conn, new_data)
    
    df = pd.read_sql('SELECT * FROM bitcoin_prices', conn)
    data = df.drop('Date', axis=1)

    df_pred, params = predict(data,'Result test2.txt',steps_ahead=14)
    
    
    #Predict Price at Today 00:00

    old_price = df['Price'].iloc[-3]
    yesterday_price = df['Price'].iloc[-2]
    predict_price = df_pred['M1'].iloc[0]
    current_price = df['Price'].iloc[-1]

    datelusa = df['Date'].iloc[-3]
    dateyesterday = df['Date'].iloc[-2]
    datetoday = df['Date'].iloc[-1]

    print(df)

    model_predictions = []
    for i in range(len(df_pred.columns)):
        model_predictions.append(df_pred.iloc[0,i])


    direction, price_change, percentage_change = calculate_price_change(yesterday_price, predict_price)

    # # Sidebar for recursive forecast days and model selection
    # forecast_days = st.sidebar.selectbox("Prediction Timeframe", options=[1, 3, 7, 14], index=2)
    # avg_forecast, forecast_all = recursive_forecast(data, days_ahead=14)
    # forecast_df = pd.DataFrame(avg_forecast[:forecast_days], columns=["Predicted Price"])
    # forecast_df.index = [f"Day {i + 1}" for i in range(forecast_days)]
    
    # Sidebar options
    st.sidebar.header("Select Prediction Timeframe")
    timeframe = st.sidebar.selectbox("Choose prediction timeframe (days)", options=[1, 3, 7, 14], index=2)
    selected_models = [col for col in df_pred.columns if st.sidebar.checkbox(col, value=True)]

    st.title("Bitcoin Price Prediction with LSTM Genetic Algorithm")
    st.markdown('''This app predicts Bitcoin prices using a genetic algorithm and pre-trained LSTM models.  
    Made by @dhuta_azikira''')
    st.divider()
    # Display prices with color-coded increase or decrease
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Day Before Yesterday's Price", value=f"${old_price:.2f}")
        st.markdown(f"{datelusa} 12AM EST")
    with col2:
        st.metric(label="Yesterday's Price", value=f"${yesterday_price:.2f}")
        st.markdown(f"{dateyesterday} 12AM EST")
    with col3:
        st.metric(label="Current Price", value=f"${current_price:.2f}", delta = f"{current_price-yesterday_price} ({(current_price-yesterday_price)/100}%)")
        st.markdown("Real-Time data, may subject to change")
    with col4:
        st.metric(label="Predicted Price", value=f"${predict_price:.2f}", 
                  delta=f"{direction:.2f} ({percentage_change:.2f}%)", delta_color="normal")
        st.markdown(f"Price prediction at {datetoday} 12AM EST")
    
    st.markdown('''
    **Prediction will update every 12 PM WIB**  
    **12 hours difference from WIB*
    ''')

    st.divider()
    # Historical chart with increased height
    st.subheader("Historical Bitcoin Prices (Last 365 Days)")
    st.line_chart(df.set_index("Date")["Price"][365:], height=500)

    # Forecast for the selected timeframe with dots
    st.subheader(f"{timeframe}-Day Price Forecast")
    if selected_models:
        st.subheader("Selected Model Predictions")
        
        # Filter based on timeframe and set a proper numeric index
        df_filtered = df_pred[selected_models].iloc[:timeframe]
        df_filtered.index = range(1, timeframe + 1)  # Ensures numeric index from 1 up to timeframe
        
        st.line_chart(df_filtered)

    st.subheader("Next-day Prediction for All Models")

    st.bar_chart(df_pred.iloc[0])


    st.subheader("Individual Model Predictions and Parameters")
    j = 0
    for i, prediction in enumerate(model_predictions, start=1):
        
        st.write(f"---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=f"Model {i} Prediction Today at 00:00", value=f"${prediction:.2f}")
        with col2:
            
            pLayer, pDroprate, pRecurDrop, pNeuron = params[j]
            st.write(f"**Model {i} Parameters**")
            st.write(f"LSTM Layers: {pLayer}")
            st.write(f"Neurons: {pNeuron}")
            st.write(f"Recurrent Dropout Rates: {pRecurDrop}")
            st.write(f"Dropout Rates: {pDroprate}")
        j+=1

    conn.close()

if __name__ == "__main__":
    main()