import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Task 1 - Loading in the data from provided url
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
Columns = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
Wilderness_Area = [f"Wilderness_Area_{i}" for i in range(1, 5)]
Soil_Type = [f"Soil_Type_{i}" for i in range(1, 41)]
Target = ["Cover_Type"]
Columns += Wilderness_Area
Columns += Soil_Type
Columns += Target
df =  pd.read_csv(url, names=Columns)

# Reducing data dimensionality - not quite feature engineering, more so because no heuristics could handle so much data
df["Wilderness_Area"] = df[Wilderness_Area].idxmax(axis=1)
df["Soil_Type"] = df[Soil_Type].idxmax(axis=1)
df = df.drop(Wilderness_Area, axis=1)
df = df.drop(Soil_Type, axis=1)
df["Wilderness_Area"] = df["Wilderness_Area"].str[16:]
df["Soil_Type"] = df["Soil_Type"].str[10:]
df["Wilderness_Area"] = pd.to_numeric(df["Wilderness_Area"])
df["Soil_Type"] = pd.to_numeric(df["Soil_Type"])

# Splitting data into train/test set 60%/40% 
X_train, X_test, y_train, y_test = train_test_split(df.drop(["Cover_Type"], axis=1), df["Cover_Type"], stratify=df["Cover_Type"], test_size=0.4)

# Task 2 - Simple heuristic approach - Genetic algorithm which finds the weights of an additive value model
class Genetic_algorithm:
    # Given X and y standardize the data and develop thresholds for class assignment -
    # the ranking provided here was acquired by doing an analysis of the sum of values for each class
    # this is a very naive method, but many of the more "convienient" methods worked 
    # only on a third of the data in my case (my personal computer isn't the greatest)
    def __init__(self, X_train, y_train):
        self.classes = 7
        self.n = 12
        self.dataset = (X_train-np.min(X_train, axis=0))/(np.max(X_train, axis=0)-np.min(X_train, axis=0))
        self.dataset = self.dataset.reset_index(drop=True)
        ranking = [4, 3, 6, 5, 2, 1, 7]
        dict_ranking = dict()
        for i in range(self.classes):
            dict_ranking[ranking[i]] = i
        self.target = pd.DataFrame(y_train, columns = ["Cover_Type"]).replace({"Cover_Type": dict_ranking})["Cover_Type"]
        self.weights = [1/self.n for i in range(12)]
        self.thresholds = [(i-1)/self.classes for i in range(1, self.classes+1)]

    # Generates random weights
    def random_solution(self):
        random_weights = random.sample(range(1, 13), 12)
        normalization_factor = sum(random_weights)
        random_weights = [i/normalization_factor for i in random_weights]
        return random_weights

    # Calculates the predicted assignments
    def distance(self, s):
        weighted_sum = pd.DataFrame(np.sum(self.dataset.astype(float).values * s, axis=1), columns = ["Weighted_sum"])
        weighted_sum["Predicted_class"] = 0
        for i in range(7):
            weighted_sum["Predicted_class"] = np.where(weighted_sum["Weighted_sum"] > self.thresholds[i], i, weighted_sum["Predicted_class"])
        return weighted_sum["Predicted_class"]

    # Provides an accuracy evaluation of the provided weights to the target value
    def evaluate(self, s):
        predicted = self.distance(s)
        correctly_classified = np.where(self.target.values == predicted.values, 1, 0).sum()
        return correctly_classified / self.dataset.shape[0], predicted
    
    # To mutate a list of weights I take a random "bit" of it and reverse it
    def mutate(self, s):
        sol = s[:]
        c1, c2 = random.sample(range(0, 12), 2)
        if c1 < c2:
            rev = sol[c1:c2+1]
            rev.reverse()
            sol[c1:c2+1] = rev
        else:
            rev = sol[c1:] + sol[:c2+1]
            rev.reverse()
            sol[:(c2+1)] = rev[-(c2+1):]
            sol[c1:] = rev[:-(c2+1)]
        return sol
        return s[:]
    
    # To crossover two sets of weights I do OX crossovering
    def crossover(self, s1, s2):
        sol1, sol2 = s1[:], s2[:]
        c1, c2 = random.sample(range(0, 12), 2)
        if c2 < c1:
            ctmp = c1
            c1 = c2
            c2 = ctmp
        sol1[c1:c2+1] = list(filter(lambda x: x in sol1[c1:c2+1], s2))
        sol2[c1:c2+1] = list(filter(lambda x: x in sol2[c1:c2+1], s1))
        return sol1, sol2

    # Returns a selected "parent" to be used to mutate/crossover
    def getTournamentSelection(self, N, st):
        index = []
        for i in range(N):
            index.append(i)
        np.random.shuffle(index)
        tournament = index[0:st]
        tournament.sort()
        parent = tournament[0]
        return parent

    # Returns a set of parent indices provided by tournament selection
    def getParentIndices(self, N, st):
        parents = []
        for i in range(2):
            parents.append(self.getTournamentSelection(N, st))
            if(i == 1):
                while(parents[0] == parents[1]):
                    parents[1] = self.getTournamentSelection(N, st)
        return parents

    #Function for training the "model" returning the best acquired results
    def train(self, evaluations = 1000, N = 100, st = 3, pm = 0.3, pc = 0.6):
        #best_evaluation = self.dataset.shape[1] * 7
        best_evaluation = 0
        best_r = []
        population = []
        noimprovement = 0
        best_classification = []
        for j in range(N):
            r = self.random_solution()
            evaluation = self.evaluate(r)
            population.append([evaluation[0], r])
        population.sort()
        for j in range(evaluations):
            matingPool = []
            for i in range(N):
                matingPool.append(self.getParentIndices(N, st))
            offspring = []
            x = 0
            for parent in matingPool:
                if np.random.random() < pc:
                    child1, child2 = self.crossover(population[parent[0]][1], population[parent[1]][1])
                    evaluation = self.evaluate(child1)
                    offspring.append([evaluation[0], child1])
                    evaluation = self.evaluate(child1)
                    offspring.append([evaluation[0], child2])
                    x += 2
            for i in range(x):
                if np.random.random() < pm:
                    child = self.mutate(offspring[i][1])
                    evaluation = self.evaluate(child)
                    offspring[i][0] = evaluation[0]
                    offspring[i][1] = child
            population += offspring
            population.sort(reverse = True)
            population = population[:N]
            if population[0][0] <= best_evaluation:
                noimprovement += 1
            else:
                noimprovement = 0
                evaluation = self.evaluate(population[0][1])
                best_classification = evaluation[1]
            best_evaluation = max(best_evaluation, population[0][0])
            if noimprovement >= 100: 
                break
        return best_evaluation, best_classification, population[0][1]

#Training the model and finiding the best weights
model_heuristic = Genetic_algorithm(X_train, y_train)
result, best_classification, best_weights = model_heuristic.train()

#Task 3 - Two baseline models - Decision Tree and K-Nearest Neighbors - training
model_clf = DecisionTreeClassifier()
model_clf.fit(X_train, y_train)

model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)

#Task 4 - Neural network with hyperparameter searching

#Since there is alreay a test set, only a validation one was needed
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, stratify=y_test, test_size=0.5)

#OHE all the targets
y_train_nn = pd.get_dummies(y_train).values
y_val_nn = pd.get_dummies(y_val).values
y_test_nn = pd.get_dummies(y_test).values

#function for making a nn, testing out different hyperparameters
def create_and_train_nn(hidden_layers, learning_rate, dropout_rate, epochs):
    nn_model = tf.keras.Sequential()
    nn_model.add(tf.keras.layers.Dense(128, input_shape=(12,), activation='relu'))
    for i in range(hidden_layers):
        nn_model.add(tf.keras.layers.Dense(128, activation='relu'))
        nn_model.add(tf.keras.layers.Dropout(dropout_rate))
    nn_model.add(tf.keras.layers.Dense(7, activation='softmax'))

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     loss=tf.keras.losses.CategoricalCrossentropy(),
                     metrics=['accuracy'])

    history = nn_model.fit(X_train, y_train_nn, epochs=epochs, validation_split=0.2)
    y_pred = nn_model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, history

# Hyperparameter space 
hyperparam_space = {
    'hidden_layers': [1, 2, 3],
    'learning_rate': [0.0001, 0.001, 0.01],
    'dropout_rate': [0.2, 0.4, 0.6],
    'epochs': [10, 20, 30]
}

# Number of trials for random search
num_trials = 10

best_hyperparams = None
best_accuracy = 0

# Random search to find the best set of hyperparameters
for i in range(num_trials):
    hyperparams = {
        'hidden_layers': np.random.choice(hyperparam_space['hidden_layers']),
        'learning_rate': np.random.choice(hyperparam_space['learning_rate']),
        'dropout_rate': np.random.choice(hyperparam_space['dropout_rate']),
        'epochs': np.random.choice(hyperparam_space['epochs'])
    }

    # training a neural network with the random hyperparameters
    accuracy, history = create_and_train_nn(**hyperparams)

    if accuracy > best_accuracy:
        best_hyperparams = hyperparams
        best_accuracy = accuracy
        best_history = history

print(f'Best Hyperparameters: {best_hyperparams}')

# Model with the best hyperparameters
model = keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(12,), activation='relu'))
for i in range(int(best_hyperparams['hidden_layers'])):
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(best_hyperparams['dropout_rate']))
model.add(keras.layers.Dense(7, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_hyperparams['learning_rate']),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train_nn, validation_data=(X_val, y_val_nn),
                    epochs=best_hyperparams['epochs'])

# Plot of the training curves for best hyperparameters
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(history.history['accuracy'], label='Training Accuracy')
ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Training Curves for best hyperparameters')
ax.legend(loc='lower right')
plt.show()

# Task 5 - Evaluation of all models

# Calculating test set accuracy
test = Genetic_algorithm(X_test, y_test)
heuristic_test_acc, y_predicted_heuristic = test.evaluate(best_weights)
nn_test_loss, nn_test_acc = model.evaluate(X_test, y_test_nn)
dt_test_acc = model_clf.score(X_test, y_test)
knn_test_acc = model_knn.score(X_test, y_test)

print(f'Heuristic Test Accuracy: {heuristic_test_acc:.4f}')
print(f'Neural Network Test Accuracy: {nn_test_acc:.4f}')
print(f'Decision Tree Test Accuracy: {dt_test_acc:.4f}')
print(f'K-Nearest Neighbors Test Accuracy: {knn_test_acc:.4f}')

# function for plotting the confusion matrix
def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    ax = plt.axes()
    sns.heatmap(cm, annot=True, ax = ax)
    ax.set_title(title)
    plt.show()

# Plot the confusion matrix for the heuristic
plot_confusion_matrix(y_test, y_predicted_heuristic, 'Heuristic Confusion Matrix')

# Plot the confusion matrix for the neural network
y_pred_nn = np.argmax(model.predict(X_test), axis=-1)
plot_confusion_matrix(y_test, y_pred_nn, 'Neural Network Confusion Matrix')

# Plot the confusion matrix for decision tree
y_pred_dt = model_clf.predict(X_test)
plot_confusion_matrix(y_test, y_pred_dt, 'Decision Tree Confusion Matrix')

# Plot the confusion matrix for K-nearest neighbors
y_pred_knn = model_knn.predict(X_test)
plot_confusion_matrix(y_test, y_pred_knn, 'K-Nearest Neighbors Confusion Matrix')
