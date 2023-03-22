import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import imageio
from tqdm import tqdm_notebook
from folium.plugins import MarkerCluster
import imageio
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import scipy
from itertools import product
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
from numpy.random import seed
from numpy.random import randint
from numpy import mean
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import collections
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D ,Dense, Flatten ,  MaxPool2D , Input

data= pd.read_csv('weather1.csv')
print(data.head())
data.info()
data2 = data.copy()
data2['RainToday'] = data2['RainToday'].replace(['No','Yes'],[0,1])
data2['RainTomorrow'] = data2['RainTomorrow'].replace(['No','Yes'],[0,1])
data2['RainToday']
data2.info()
data3 = data2.select_dtypes(include=np.number)
data3.head()
data3.describe()
#discrete random variable
def get_frequencies(values):
    frequencies = {}
    for v in values:
        if v in frequencies:
            frequencies[v] += 1
        else:
            frequencies[v] = 1
    return frequencies

def get_probabilities(sampledata, freqs):
    probabilities = []
    for k, v in freqs.items():
        probabilities.append(round(v / len(sampledata), 1))
    return probabilities
data["Cloud9am"]
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['axes.grid']=True
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8
colors = ['xkcd:pale orange', 'xkcd:sea blue', 'xkcd:pale red', 'xkcd:sage green', 'xkcd:terra cotta', 'xkcd:dull purple', 'xkcd:teal', 'xkcd: goldenrod', 'xkcd:cadet blue',
'xkcd:scarlet']
sample = data["Cloud9am"]
calculated_frequencies = get_frequencies(sample)
print(calculated_frequencies)
calculate_probabilities = get_probabilities(sample, calculated_frequencies)
print("prob", calculate_probabilities)
x_axis = list(set(sample))
plt.bar(x_axis, calculate_probabilities)
plt.legend(["Cloud thichness"])
plt.show()
plt.scatter(data['WindSpeed3pm' ]  ,data[ "WindSpeed9am"] )
plt.title("Wind Speeds 'at 3pm' to 'at 9am' ")
plt.xlabel("Wind Speeds at 3pm")
plt.ylabel("Wind Speeds at 9am")
plt.show()
plt.scatter(data['WindSpeed3pm' ]  ,data[ "Sunshine"] )
plt.title("Wind Speeds at 3pm to Sunshine")
plt.xlabel("Wind Speeds at 3pm")
plt.ylabel("Sunshine")
plt.show()
mylabels= ["Sunny day","No cloud","Very little cloud","-","Cloudy","Very cloudy","Heavy clouds","No-sun-day","-"]
myexplode = [0.15,0,0,0,0,0,0.2,0,0]
plt.pie(calculate_probabilities,labels =mylabels,explode = myexplode, shadow = True)
plt.title("Cloud thichness")
plt.show()
#Central limit theorem
# seed the random number generator, so that the experiment is #replicable
seed(1)
# generate a sample of men's weights
Rainfall = data['Rainfall']
# print(Rainfall)
print('The average probability of rain is {} %'.format(mean(Rainfall)))
plt.hist(Rainfall)
plt.title('Probability of Rain')
plt.show()
stat, p = shapiro(Rainfall)
print('Statistics={}, p={}'.format(stat, p))
alpha = 0.05
if p > alpha:
    print('Sample looks Normal (do not reject H0)')
else:
    print('Sample does not look Normal(rejectH0)')
# initializing 
Temp3pm = data['Temp3pm']
# getting data of the histogram
count, bins_count = np.histogram(Temp3pm, bins=20)
# finding the PDF of the histogram using count values
pdf = count / sum(count)
# using numpy np.cumsum to calculate the CDF
# We can also find using the PDF values by looping and adding
cdf = np.cumsum(pdf)
# plotting PDF and CDF
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.legend()
plt.show()
#NAIEVE BAYES
def bayesTheorem(pA, pB, pBA):
    return pA * pBA / pB
#define function for Bayes' theorem
def bayesTheorem(pA, pB, pBA):
    return pA * pBA / pB
#define probabilities
pRain = 0.2
pCloudy = 0.4
pCloudyRain = 0.85
#use function to calculate conditional probability
print(bayesTheorem(pRain, pCloudy, pCloudyRain))
data3.head()
outputs = data3["RainTomorrow"]
inputs = data3.drop(["RainTomorrow"], axis=1)
print(inputs.shape)
inputs.head()
print(outputs.shape)
outputs.head()
y= outputs.to_numpy()
X = inputs.to_numpy()
X[np.isnan(X)]
X[np.isnan(X)] = 0
X[np.isnan(X)]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.23, random_state=42)
print([i.shape for i in [ X_train, X_test, y_train, y_test]])
model_predict_rain = Sequential(
     [               
        tf.keras.Input(shape=(18,)),    #specify input size
        Dense(300,activation='relu'),
         Dense(200,activation='relu'),
        Dense(150,activation='relu'),
         Dense(10,activation='relu'),
        Dense(2,activation='softmax')
        
    ], name = "rain_prediction2" )
model_predict_rain.summary()
model_predict_rain.compile(
    
#     metrics=['accuracy'],
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    optimizer=tf.keras.optimizers.Adam(0.0001),
)
epochs = 200
model_predict_rain.fit(X_train, y_train ,  epochs=epochs,verbose=1, validation_split=0.2)
model_predict_rain.evaluate(X_test,y_test)
X[1].shape
test_value_number = 1
model_predict_rain.predict(X[1].reshape(1,X[test_value_number].shape[0],1))
y_preds = model_predict_rain.predict(X)
got_wrong= 0
for i in range(y.shape[0]):
    y_pred_value = np.where(y_preds[i]==np.max(y_preds[i]))[0][0]
    got_wrong += abs(y_pred_value-y[i])
    print(f'The Real Value is: {y[i]}\t The Predicted value is: {y_pred_value}')
print(f'got {got_wrong} wrong from {y.shape[0]} example')

