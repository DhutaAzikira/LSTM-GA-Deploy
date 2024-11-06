import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
import copy
import datetime
import tensorflow as tf
import keras

from keras import backend as K
from keras import callbacks
from keras.models import clone_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

def create_list(median_value, list_length=5, increment_size=1):

    start_value = median_value - (list_length // 2) * increment_size

    if start_value <= 0:
      start_value = 1

    if median_value < float(1):
      start_value = 0

    result_list = [start_value + i * increment_size for i in range(list_length)]

    return result_list


create_list(4)

def create_sequences(data, seq_length):

    x = []
    y = []

    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])

        y.append(data[i + seq_length, 0])

    return np.array(x), np.array(y)

def prepare_data(group, seq_length):

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(group)

    x, y = create_sequences(scaled_data, seq_length)

    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    return x_train, x_test, y_train, y_test, scaler

def feedSlideWindow(data, seq_length, batch_size, window_size):

    dataChain = []

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x, y = create_sequences(scaled_data, seq_length)

    n_groups = (len(x) - batch_size) // window_size

    print(f"You have :{len(x)-seq_length} total sequences.")
    print(f"You can make {n_groups} groups")

    for i in range(0, len(x), window_size):

      x_train = x[i : i + batch_size]
      y_train = y[i : i + batch_size]

      x_test = x[i + batch_size : i + batch_size + window_size]
      y_test = y[i + batch_size : i + batch_size + window_size]

      dataChain.append([x_train, x_test, y_train, y_test, scaler])


    return dataChain, n_groups

def group_data(data, group_size, historicLearning):

    n_groups = -(-len(data) // group_size)
    groups = []
    cumulativeGroup = pd.DataFrame()

    for i in range(0, len(data), group_size):
      group = data.iloc[i:i + group_size]
      if len(group) == group_size:
        if historicLearning == False:
          groups.append(group)
        else:
          cumulativeGroup = pd.concat([cumulativeGroup, group])
          groups.append(cumulativeGroup.copy())

      else:
        # Discard groups with insufficient data
        n_groups -= 1
        print(f"Discarding group starting at index {i} due to insufficient data.")

    print(f"Created {n_groups} groups")

    return groups, n_groups

def feedData(data, group_size, seq_length, historicLearning):

    groups, n_groups = group_data(data,group_size, historicLearning)
    dataChain = []

    for i in (range(len(groups))):

      x_train, x_test, y_train, y_test, scaler = prepare_data(groups[i],seq_length)

      dataChain.append([x_train, x_test, y_train, y_test, scaler])

    return dataChain, n_groups

##-Intialize Model

def seedModel(n_lstmLayer, n_dropout, n_recurrent_dropout, n_neurons):

  model = Sequential()
  for i in range(n_lstmLayer):
    if n_lstmLayer == 0:
      model.add(LSTM(n_neurons, return_sequences=True, recurrent_dropout=n_recurrent_dropout))
      model.add(Dropout(n_dropout))

    elif i < n_lstmLayer-1:
      model.add(LSTM(n_neurons, return_sequences=True,  recurrent_dropout=n_recurrent_dropout))
      model.add(Dropout(n_dropout))

    else:
      model.add(LSTM(n_neurons, return_sequences=False,  recurrent_dropout=n_recurrent_dropout))
      model.add(Dropout(n_dropout))

  model.add(Dense(25))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mean_squared_error')

  return model

def randomSeedModel():

  n_lstmLayer = create_list(3,3,1)
  n_Neurons = create_list(200,16,8)
  n_Dropouts = create_list(0,4,0.1)
  n_recurrentDropouts = create_list(0,3,0.1)


  model = Sequential()
  n_lstmLayer = random.choice(n_lstmLayer)

  for i in range(n_lstmLayer):
    if n_lstmLayer == 0:
      model.add(LSTM(random.choice(n_Neurons), return_sequences=True, recurrent_dropout=random.choice(n_recurrentDropouts)))
      model.add(Dropout(random.choice(n_Dropouts)))

    elif i < n_lstmLayer-1:
      model.add(LSTM(random.choice(n_Neurons), return_sequences=True,  recurrent_dropout=random.choice(n_recurrentDropouts)))
      model.add(Dropout(random.choice(n_Dropouts)))

    else:
      model.add(LSTM(random.choice(n_Neurons), return_sequences=False,  recurrent_dropout=random.choice(n_recurrentDropouts)))
      model.add(Dropout(random.choice(n_Dropouts)))

  model.add(Dense(25))
  model.add(Dense(1))

  model.compile(optimizer='adam', loss='mean_squared_error')

  return model

##-Mutation and Crossover

def getParams(model):

    lstm_params = []
    dropout_params = []
    pLayer = 0

    for layer in model.layers:
        if isinstance(layer, LSTM):
            pLayer += 1
            lstm_params.append({
                'neurons': layer.units,
                'recurrent_dropout': layer.recurrent_dropout,
            })
        elif isinstance(layer, Dropout):
            dropout_params.append({
                'dropout_rate': layer.rate
            })

    pNeuron = [layer['neurons'] for layer in lstm_params]
    pRecurDrop = [layer['recurrent_dropout'] for layer in lstm_params]
    pDroprate = [layer['dropout_rate'] for layer in dropout_params]

    # print(f"Model Total LSTM Layer : {pLayer} | Droprates : {pDroprate} | Recurrent_Droprate : {pRecurDrop} | Neurons : {pNeuron}")

    return pLayer, pDroprate, pRecurDrop, pNeuron

###--Mutation

def mutation(model):

    pLayer, pDroprate, pRecurDrop, pNeuron = getParams(model)

    cNeuron = []
    cDroprate = []
    cRecurDrop = []

    for i in range(len(pNeuron)):

      mutation_rate = random.uniform(0.05,0.1)*random.uniform(-1,1)

      #Neuron Mutation (100% chance to mutate every neuron in each layer)
      cNeuron.append(int(round(pNeuron[i]+pNeuron[i]*mutation_rate)))

      #Droprate Mutation (33% chance to mutate from 0)
      if pDroprate == []:
        for i in range(pLayer):
          if random.randint(0,2) == 1:
            cDroprate.append(0.1*abs(mutation_rate))
          else:
            cDroprate.append(0)

      else:
        cDroprate.append(pDroprate[i]+pDroprate[i]*mutation_rate )


      #RecurDrop Mutation (33% chance to mutate from 0)
      if sum(pRecurDrop) == 0:
        for i in range(pLayer):
          if random.randint(0,2) == 1:
            cRecurDrop.append(0.1*abs(mutation_rate))
          else:
            cRecurDrop.append(0)
      else:
        cRecurDrop.append(pRecurDrop[i]+pRecurDrop[i]*mutation_rate)


    #Layer Mutation (33% Chance to Mutate Layer)
    if pLayer >= 5:
      pLayer = 4

    _listLayer = create_list(pLayer,3,1)
    cLayer = random.choice(_listLayer)

    if cLayer > 5:
      return print("More than 5 layers is too much!")

    for i in range(len(cNeuron)):
      if cNeuron[i] < 80:
        cNeuron[i] = 80
      elif cNeuron[i] > 250:
        cNeuron[i] = 250

      if cDroprate[i] < 0:
        cDroprate[i] = 0
      elif cDroprate[i] > 0.5:
        cDroprate[i] = 0.5

      if cRecurDrop[i] < 0:
        cRecurDrop[i] = 0
      elif cRecurDrop[i] > 0.25:
        cRecurDrop[i] = 0.25



    return cLayer, cDroprate, cRecurDrop, cNeuron

def weightMutation(model):

  weightMutatedModel = tf.keras.models.clone_model(model)

  weightMutatedModel.set_weights(model.get_weights())

  for layer in weightMutatedModel.layers:
      weights = layer.get_weights()
      if weights:
          mutated_weights = [w + np.random.normal(0, 0.01, size=w.shape) for w in weights]
          layer.set_weights(mutated_weights)

  return weightMutatedModel

###--Crossover

def crossover(parent1, parent2):
    # Identify the primary and secondary parents based on layer count
    if len(parent1.layers) > len(parent2.layers):
        primary_parent, secondary_parent = parent1, parent2
    else:
        primary_parent, secondary_parent = parent2, parent1

    # Get parameters from both parents
    pLayer1, pDroprate1, pRecurDrop1, pNeuron1 = getParams(primary_parent)
    pLayer2, pDroprate2, pRecurDrop2, pNeuron2 = getParams(secondary_parent)

    cLayer = []
    cDroprate = []
    cRecurDrop = []
    cNeuron = []

    cDroprate = pDroprate2[:pLayer1] + pDroprate1[pLayer2:]
    cRecurDrop = pRecurDrop2[:pLayer1] + pRecurDrop1[pLayer2:]
    cNeuron = pNeuron2[:pLayer1] + pNeuron1[pLayer2:]

    # Choose the max number of layers between parents
    cLayer = max(pLayer1, pLayer2)

    return cLayer, cDroprate, cRecurDrop, cNeuron

def weightCrossover(parent1, parent2, alpha=0.5):

    weightCrossoverModel = tf.keras.models.clone_model(parent1)

    parent1_weights = parent1.get_weights()
    parent2_weights = parent2.get_weights()

    child_weights = []

    num_layers = len(parent1_weights)
    for i in range(num_layers):
        w1 = parent1_weights[i]
        w2 = parent2_weights[i]

        if random.choice([1,2]) == 2 and w1.shape == w2.shape:

            if w1.size == 1:
              child_weights.append(w1)

            else:
              split_point = random.randint(1, w1.size - 1)

              w1_flat = w1.flatten()
              w2_flat = w2.flatten()

              new_weights = np.concatenate((w1_flat[:split_point], w2_flat[split_point:])).reshape(w1.shape)

              child_weights.append(new_weights)
        else:

            child_weights.append(w1)

    weightCrossoverModel.set_weights(child_weights)

    return weightCrossoverModel


##-Build

def trackLossValLoss(history):

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def buildLSTM(cLayer, cDroprate, cRecurDrop, cNeurons, x_train, y_train, epochs, batch_size, pModel = None):

  model = Sequential()
  early_stopping = EarlyStopping(monitor='val_loss', patience=(10), restore_best_weights=True)

  for i in range(cLayer):
    return_sequences = True if i < cLayer - 1 else False
    neurons = cNeurons[i] if i < len(cNeurons) else random.choice(cNeurons)
    recurrent_dropout = cRecurDrop[i] if i < len(cRecurDrop) else random.choice(cRecurDrop)
    dropout_rate = cDroprate[i] if i < len(cDroprate) else random.choice(cDroprate)

    if i == 0:
      model.add(LSTM(neurons,
                    return_sequences=return_sequences,
                    recurrent_dropout=recurrent_dropout,
                    input_shape=(x_train.shape[1], x_train.shape[2])))
    else:
      model.add(LSTM(neurons,
                    return_sequences=return_sequences,
                    recurrent_dropout=recurrent_dropout))

    model.add(Dropout(dropout_rate))

  model.add(Dense(25))
  model.add(Dense(1))

  model.compile(optimizer='adam', loss='mean_squared_error')
  model.summary()

  if pModel is not None:
        for i, layer in enumerate(model.layers):
            if i < len(pModel.layers):
                try:
                    layer.set_weights(pModel.layers[i].get_weights())
                except Exception as e:
                    print(f"Could not transfer weights for layer {i}: {e}")
                    print("*this is usually fine because the first model does not have weights")


  history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping,SelectiveProgbarLogger(epoch_interval = 10,verbose = 1)], verbose=0)
  trackLossValLoss(history)

  mLayer, mDroprate, mRecurDrop, mNeuron = getParams(model)
  print(f"Model Total LSTM Layer : {mLayer} | Droprates : {mDroprate} | Recurrent_Droprate : {mRecurDrop} | Neurons : {mNeuron}")
  K.clear_session()


  return model

def populate(seedModel, n_population, dataChain, epochs, batch_size, firstGeneration = True):

    if n_population < 4 and n_population > 12:
      raise ValueError("Population must be between 4-12")

    x_train, x_test, y_train, y_test, scaler = dataChain

    models = []
    percentage = 0.25

    if firstGeneration == True:
      print("First Generation are seedModel and the rest randomSeedModel")
      pLayer, pDroprate, pRecurDrop, pNeuron = getParams(seedModel)
      model = buildLSTM(pLayer, pDroprate, pRecurDrop, pNeuron, x_train, y_train, epochs, batch_size)
      models.append(model)

      for i in range(n_population-1):

        model = randomSeedModel()
        pLayer, pDroprate, pRecurDrop, pNeuron = getParams(model)
        model = buildLSTM(pLayer, pDroprate, pRecurDrop, pNeuron, x_train, y_train, epochs, batch_size)
        models.append(model)

      firstGeneration = False
      return models

    else:

      _1bestmodel, _2bestmodel, _3bestmodel = seedModel
      _crossparams = []

      base_count = min(n_population, 10)
      extra_count = max(0, n_population - base_count)  #  #

      model_counts = [2, 2, 2, 1]

      if n_population > 10:
          model_counts[3] += extra_count


      for i in range(n_population):
          if i == 0:
              print("Exact Best Model")
              pLayer, pDroprate, pRecurDrop, pNeuron = getParams(_1bestmodel)
              model = buildLSTM(pLayer, pDroprate, pRecurDrop, pNeuron, x_train, y_train, epochs, batch_size)
              models.append(model)

              print(getParams(model))

          elif i < model_counts[0] + 1:
              print("Weight Mutation")
              mutated_model = weightMutation(_1bestmodel if i % 2 == 0 else _2bestmodel)
              models.append(mutated_model)
              _crossparams.append(mutated_model)
              mutated_model.summary()

              print(getParams(mutated_model))

          elif i < model_counts[0] + model_counts[1] + 1:
              print("Weight Crossover")
              crossover_model = weightCrossover(_1bestmodel if i % 2 == 0 else _2bestmodel, _crossparams[1] if i % 2 == 0 else _crossparams[0])
              models.append(crossover_model)
              crossover_model.summary()

              print(getParams(crossover_model))


          elif i < model_counts[0] + model_counts[1] + model_counts[2] + 1:
              print("Architecture Crossover")
              _crossover = crossover(_1bestmodel if i % 2 == 0 else _2bestmodel, _2bestmodel if i % 2 == 0 else _3bestmodel)
              cLayer, cDroprate, cRecurDrop, cNeuron = _crossover
              model = buildLSTM(cLayer, cDroprate, cRecurDrop, cNeuron, x_train, y_train, epochs, batch_size)
              models.append(model)

              print(getParams(model))

          else:
              print("Architecture Mutation")
              _mutation = mutation(_3bestmodel if i % 3 == 0 else (_1bestmodel if i % 2 == 0 else _2bestmodel))
              cLayer, cDroprate, cRecurDrop, cNeuron = _mutation
              model = buildLSTM(cLayer, cDroprate, cRecurDrop, cNeuron, x_train, y_train, epochs, batch_size)
              models.append(model)

              print(getParams(model))


      return models


def evaluate(models, x_test, y_test, scaler):

    evaluation_result = []

    for model in models:

        predictions = model.predict(x_test)

        predictions_actual = scaler.inverse_transform(
            np.concatenate((predictions, np.zeros((predictions.shape[0], 5 - 1))), axis=1)
        )[:, 0]
        y_test_actual = scaler.inverse_transform(
            np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 5 - 1))), axis=1)
        )[:, 0]

        mse = mean_squared_error(y_test_actual, predictions_actual)
        rmse = np.sqrt(mse)

        print("Metric Evaluation")
        print("RMSE :", rmse)
        print("MSE  :", mse)

        evaluation_result.append([model, rmse, predictions_actual, y_test_actual])

    evaluation_result.sort(key=lambda x: x[1])

    return evaluation_result[0:3]

def visualize(evaluation_result, group_index, group_size, data, slideWindow, historicLearning):

    for i in range(len(evaluation_result)):
      model, rmse, predictions_actual, y_test_actual = evaluation_result[i]

      if slideWindow == True:

        dataChain = data

        print(dataChain[2].shape)
        print(dataChain[3].shape)

        y_train = dataChain[2]
        y_test = dataChain[3]
        scaler = dataChain[4]

        train_actual = scaler.inverse_transform(
              np.concatenate((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], 5 - 1))), axis=1)
          )[:, 0]

        test_actual = scaler.inverse_transform(
              np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 5 - 1))), axis=1)
          )[:, 0]

        start_index = list(range(len(train_actual), len(train_actual) + len(predictions_actual)))

        plt.figure(figsize=(16, 8))
        plt.plot(train_actual, label='Train', color = 'blue')
        plt.plot(start_index,test_actual, label='Actual Test', color = 'blue', linestyle='dashed')
        plt.plot(start_index,y_test_actual, label='Test Inversed', color = 'orange')
        plt.plot(start_index,predictions_actual, label='Prediction', color = 'green')
        plt.ylabel('Bitcoin Price')
        plt.xlabel('Date')
        plt.title(f'Actual vs. Predicted Values for Generation {group_index + 1} (RMSE: {rmse:.4f})')
        plt.legend()
        plt.show()

      else:


        groups, n_groups = group_data(data, group_size, historicLearning)

        slice_index = int(len(groups[group_index]) * 0.8)

        plt.figure(figsize=(16, 8))

        plt.plot(groups[group_index]['Price'], label = 'Train', color = 'blue')
        plt.plot(groups[group_index].index[slice_index:], groups[group_index]['Price'][slice_index:], label = 'Actual Test', color='blue' , linestyle = 'dashed')
        plt.plot(groups[group_index].index[slice_index:][:len(y_test_actual)], y_test_actual, label = 'Test Inversed', color='orange')
        plt.plot(groups[group_index].index[slice_index:][:len(predictions_actual)], predictions_actual,label = 'Prediction' , color='green' )

        plt.legend()
        plt.title(f'Actual vs. Predicted Values for Generation {group_index + 1} (RMSE: {rmse:.4f})')
        plt.xlabel('Date')
        plt.ylabel('Bitcoin Price')
        plt.show()

##-Main Function

def genetic(seedModel, data, n_generation, n_population, group_size, seq_length, window_size, slideWindow = True, historicLearning = False, epochs = 50, batch_size = 32):

  if slideWindow == True:
    dataChain, n_groups = feedSlideWindow(data, seq_length, group_size, window_size)
  else:
    dataChain, n_groups = feedData(data, group_size, seq_length, historicLearning)

  finalResult = []
  firstGeneration = True

  for i in range(n_generation):
    if n_generation > n_groups:
      print("Not enough data")
      break
    print("Generation",{i+1})
    if firstGeneration == True:
      models = populate(seedModel, n_population,dataChain[i],epochs,batch_size,firstGeneration = True)
      firstGeneration = False
    else:
      models = populate(seedModel, n_population,dataChain[i],epochs,batch_size,firstGeneration = False)

    evaluation_result = evaluate(models, dataChain[i][1], dataChain[i][3], dataChain[i][4])
    if slideWindow == True:
      visualize(evaluation_result, i, group_size, dataChain[i], slideWindow, historicLearning)
    else:
      visualize(evaluation_result, i, group_size, data, slideWindow, historicLearning)

    seedModels = []
    for j in range(len(evaluation_result)):
      seedModels.append(evaluation_result[j][0])
      print(getParams(evaluation_result[j][0]))

    seedModel = seedModels
    for k in range(len(seedModel)):
      finalResult.append([i,evaluation_result[k][0],evaluation_result[k][1],evaluation_result[k][2],evaluation_result[k][3]])

  return finalResult

def evaluatePrediction(finalResult, data, group_size, seq_length, showAll = False):

    test = []
    prediction = []
    predictions_best = []
    test_best = []

    def percentage_error(actual, predicted):
        res = np.empty(actual.shape)
        for j in range(actual.shape[0]):
            if actual[j] != 0:
                res[j] = (actual[j] - predicted[j]) / actual[j]
            else:
                res[j] = predicted[j] / np.mean(actual)
        return res

    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100

    for i in range(len(finalResult)):
        prediction = np.concatenate((prediction,finalResult[i][3]))
        test = np.concatenate((test,finalResult[i][4]))

    for i in range(0, len(finalResult), 3):  # Step through every 3 items
        predictions_best.extend(finalResult[i][3])  # Extend, not concatenate
        test_best.extend(finalResult[i][4])


    print(len(test_best))
    print(len(predictions_best))
    print(len(test))
    print(len(prediction))

    mse = mean_squared_error(test_best, predictions_best)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_best, predictions_best)
    mape = mean_absolute_percentage_error(test_best, predictions_best)

    print("GenLSTM Metrics Evaluation")
    print(f"MSE :",{mse})
    print(f"RMSE :",{rmse})
    print(f"MAE :",{mae})
    print(f"MAPE :{mape}%")

    start_index = df.index[seq_length+group_size:seq_length+group_size+len(test_best)]

    plt.figure(figsize=(32, 18))
    if showAll == True:
      plt.plot(df['Price'], label = "Dataset")

    # plt.plot(start_index, df['Price'][len(start_index):len(start_index)+len(test_best)], label = 'Jembut', color ='green')
    plt.plot(df.index[:seq_length+group_size],df['Price'][:seq_length+group_size], label='training', color = 'C0')
    plt.axvline(x=start_index[0],label = 'Train', color="black")
    plt.plot(start_index, test_best, label = "Test", color = 'blue')
    plt.plot(start_index, predictions_best,  label = "Prediction", color = 'red')
    plt.xlabel('Days')
    plt.ylabel('Bitcoin Price')
    plt.title('GenLSTM Actual vs Prediction Graph')
    plt.legend()
    plt.show()

def saveFinal(finalResult):

  date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  try:
    with open(f'Result {date}.txt', 'wb') as f: #Save finalResult to Txt for later
        pickle.dump(finalResult,f)

  except:
    newFinalResult = []

    for i in range(len(finalResult)):
      for j in range(5):
        if j == 1:
          newFinalResult.append(getParams(finalResult[i][j]))
          newFinalResult.append(finalResult[i][j].get_weights())
        else:
          newFinalResult.append(finalResult[i][j])

    with open(f'Result NonKeras {date}.txt', 'wb') as f: #Save finalResult to Txt for later
        pickle.dump(newFinalResult,f)
