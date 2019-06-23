import pandas as pd
import numpy as np
import os
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, Input, concatenate, Lambda, Embedding, Layer
from keras import optimizers
import time
import keras.backend as K
def dice_coef(y_true, y_pred):
	y_sel = K.argmax(y_pred)

	return -K.sum(K.gather(y_true, y_pred), axis=-1)

def extract_train_test(reflen, outlen, start=1101, end=9999):
	"""
		get stock data from data
		data preprocess : fill "--" to nan
		then fillna to ffill
		data[x:x+reflen]        is input data
		data[x:x+reflen+outlen] is output ref
		divide close data to normalize
	"""
	x_data = np.zeros((reflen*4,0),float)
	y_data = []
	performance = []
	for i in range(start, end):
		if os.path.exists("./data/" + str(i) + ".csv"):	
			df = pd.read_csv("./data/" + str(i) + ".csv")
			o = ((df["open"].replace("--",np.nan).fillna(method="ffill").values.astype(float)))
			h = ((df["high"].replace("--",np.nan).fillna(method="ffill").values.astype(float)))
			l = ((df["low"].replace("--",np.nan).fillna(method="ffill").values.astype(float)))
			c = ((df["close"].replace("--",np.nan).fillna(method="ffill").values.astype(float)))
			#v = ((df["volume"].replace("--",np.nan).fillna(method="ffill").values.astype(float)))

			if o.shape[0] < reflen + outlen:
				continue

			for i in range(0, o.shape[0]-reflen-outlen):				
				if 0 in o[i:i+reflen+outlen] or 0 in h[i:i+reflen+outlen] or 0 in l[i:i+reflen+outlen] or 0 in c[i:i+reflen+outlen]:
					continue
				x_data = np.concatenate((x_data,np.append(np.append(np.append(o[i:i+reflen]/c[reflen+i],h[i:i+reflen]/c[reflen+i]),c[i:i+reflen]/c[reflen+i]),l[i:i+reflen]/c[reflen+i]).reshape(-1,1)),axis=1)
				profit = (c[reflen+outlen+i]-c[reflen+i])/c[reflen+i]
				if profit > 0.1:
					y_data.append([1,0,0,0,0])
				elif profit > 0.03:
					y_data.append([0,1,0,0,0])
				elif profit > -0.03:
					y_data.append([0,0,1,0,0])
				elif profit > -0.1:
					y_data.append([0,0,0,1,0])
				else:
					y_data.append([0,0,0,0,1])
				performance.append(profit)
			print(x_data.shape, len(y_data))
	x_data = np.swapaxes(x_data, 0, 1)
	y_data = np.asarray(y_data)
	performance = np.asarray(performance)
	#y_data = np.where(np.logical_and(-0.05 < y_data, y_data < 0.05), 0, y_data)
	#print(np.where(-0.05 < y_data < 0.05))
	#print(y_data[np.where(-0.05 < y_data < 0.05)])
	return x_data, y_data, performance
			

def create_model(reflen):

	input_data = Input(shape=(reflen*4,))
	l0 = Dense(200, activation='relu')(input_data)
	l1 = Dense(160, activation='relu')(l0)
	l2 = Dense(80, activation='relu')(l1)
	l3 = Dense(20, activation='relu')(l2)
	l4 = Dense(5, activation='softmax')(l3)
	adam = optimizers.Adam(lr=0.0001)
	model = Model(inputs=[input_data], output=l4)
	#model.compile(optimizer=adam, loss='mse')
	model.compile(optimizer=adam, loss='categorical_crossentropy')
	return model
#df = pd.read_csv("./data/1101.csv")
#print(df["high"][0:20].values + df["close"][0:20].values)
#a = (np.concatenate((df["high"].replace("--",np.nan).values.astype(float),df["close"].replace("--","-1").values.astype(float)),axis=0))
#print(df["close"].replace("--",np.nan).fillna("ffill").values)
#print(a.shape)
reflen = 50
outlen = 10
#xdata_train, ydata_train = loadtrain(reflen, outlen)
#xdata_test1, ydata_test1 = extract_train_test(reflen, outlen, 1101, 1102)
#xdata_test2, ydata_test2 = extract_train_test(reflen, outlen, 1301, 1302)
#xdata_test3, ydata_test3 = extract_train_test(reflen, outlen, 2330, 2331)
xdata, ydata, performance = extract_train_test(reflen, outlen, 2301, 2305)
xdata_train = xdata[:int(xdata.shape[0]/2)]
ydata_train = ydata[:int(ydata.shape[0]/2)]
xdata_test1 = xdata[int(xdata.shape[0]/2):]
ydata_test1 = ydata[int(ydata.shape[0]/2):]

mdl = create_model(reflen)
i = 0
ttlrpt = open("ttlrpt.csv", "w")
while True:
	mdl.fit(xdata_train, ydata_train, epochs=50, batch_size=128, shuffle=True)
	#if i%500 == 499:
	#	adam = optimizers.Adam(lr=0.00001)
	#	mdl.compile(optimizer=adam, loss='categorical_crossentropy')
	
	if i%10 == 9:
		
		prd = mdl.predict(xdata_test1)
		rpt = open("result " + str(i) + ".csv", "w")
		totalprofit1 = 0
		totalprofit2 = 0
		totalprofit3 = 0
		totalprofit4 = 0
		totalprofit5 = 0
		ttlrpt.write(str(i)+",")
		for j in range(prd.shape[0]):
			rpt.write(str(prd[j][0]) + "," + str(prd[j][1]) + "," + str(prd[j][2]) + "," + str(prd[j][3]) + "," + str(prd[j][4]) + "," + str(performance[j]) + "," + str(np.argmax(prd[j])) + ",\n")
			if np.argmax(prd[j]) == 0:
				totalprofit1 = totalprofit1 + performance[j] - 0.02
			if np.argmax(prd[j]) == 1:
				totalprofit2 = totalprofit2 + performance[j] - 0.02
			if np.argmax(prd[j]) == 2:
				totalprofit3 = totalprofit3 + performance[j] - 0.02
			if np.argmax(prd[j]) == 3:
				totalprofit4 = totalprofit4 - performance[j] - 0.02
			if np.argmax(prd[j]) == 4:
				totalprofit5 = totalprofit5 - performance[j] - 0.02
		ttlrpt.write(str(totalprofit1)+","+str(totalprofit2)+","+str(totalprofit3)+","+str(totalprofit4)+","+str(totalprofit5)+",\n")
		ttlrpt.flush()
		rpt.close()

		"""prd = mdl.predict(xdata_test2)
		rpt = open("result " + str(i) + ".csv", "a")
		rpt.write("\n")
		for j in range(prd.shape[0]):
			rpt.write(str(prd[j][0]) + "," + str(ydata_test2[j]) + ",\n")
		rpt.close()

		prd = mdl.predict(xdata_test3)
		rpt = open("result " + str(i) + ".csv", "a")
		rpt.write("\n")
		for j in range(prd.shape[0]):
			rpt.write(str(prd[j][0]) + "," + str(ydata_test3[j]) + ",\n")
		rpt.close()"""

	i = i + 1