import pandas as pd
import numpy as np
import os
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, Input, concatenate, Lambda, Embedding, Layer
from keras import optimizers
import time
import keras.backend as K
def dice_coef(y_true, y_pred):
	y_sel = K.one_hot(K.cast(y_pred,dtype='int32'),num_classes=3)
	return -K.sum(y_pred[K.cast(y_sel,dtype='int32')]-0.02, axis=-1)

def loadtrain(reflen, outlen):
	x_data = np.zeros((reflen*4,0),float)
	y_data = []
	for root, dirs, files in os.walk("./data"):
		for f in files:
			df = pd.read_csv(os.path.join(root,f))
			o = ((df["open"].replace("--",np.nan).fillna(method="ffill").values.astype(float)))
			h = ((df["high"].replace("--",np.nan).fillna(method="ffill").values.astype(float)))
			l = ((df["low"].replace("--",np.nan).fillna(method="ffill").values.astype(float)))
			c = ((df["close"].replace("--",np.nan).fillna(method="ffill").values.astype(float)))
			#v = ((df["volume"].replace("--",np.nan).fillna(method="ffill").values.astype(float)))
			if o.shape[0] < reflen + outlen:
				continue
			if 0 in o or 0 in h or 0 in l or 0 in c:
				continue
			
			for i in range(0, o.shape[0]-reflen-outlen):
				#print(x_data)
				x_data = np.concatenate((x_data, np.concatenate(o[i+reflen-3:i+reflen]/c[reflen+i],h[i+reflen-3:i+reflen]/c[reflen+i],c[i+reflen-3:i+reflen]/c[reflen+i],l[i+reflen-3:i+reflen]/c[reflen+i])),axis=1)
				#x_data = np.concatenate((x_data,np.concatenate(((o[i:reflen+i]-c[i:reflen+i])/c[reflen+i],(h[i:reflen+i]-c[i:reflen+i])/c[reflen+i],(l[i:reflen+i]-c[i:reflen+i])/c[reflen+i],c[i:reflen+i]/c[reflen+i]),axis = 0).reshape(-1,1)),axis= 1)
				if (c[reflen+outlen+i]-c[reflen+i])/c[reflen+i] > 0.02:
					y_data.append([1,0,0])
				elif (c[reflen+outlen+i]-c[reflen+i])/c[reflen+i] > -0.02:
					y_data.append([0,1,0])
				else:
					y_data.append([0,0,1])
				#y_data.append([(c[reflen+outlen+i]-c[reflen+i])/c[reflen+i],0.02,-(c[reflen+outlen+i]-c[reflen+i])/c[reflen+i]])
			
			print(x_data.shape, y_data.shape, "???????")
			if len(y_data) > 100000:
				x_data = np.swapaxes(x_data, 0, 1)
				y_data = np.asarray(y_data)
				#y_data = np.where(-0.05 < y_data < 0.05, 0, y_data)
				#print(np.where(-0.05 < y_data < 0.05))
				#print(y_data[np.where(-0.05 < y_data < 0.05)])
				return x_data, np.asarray(y_data)

	x_data = np.swapaxes(x_data, 0, 1)
	y_data = np.asarray(y_data)
	#y_data = np.where(-0.05 < y_data < 0.05, 0, y_data)
	#print(np.where(-0.05 < y_data < 0.05))
	#print(y_data[np.where(-0.05 < y_data < 0.05)])
	return x_data, np.asarray(y_data)

def extract_train_test(reflen, outlen, start=1101, end=9999):
	x_data = np.zeros((reflen+9,0),float)
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
			if 0 in o or 0 in h or 0 in l or 0 in c:
				continue
			
			for i in range(0, o.shape[0]-reflen-outlen):
				#print(x_data)

				x_data = np.concatenate((x_data,np.append(np.append(np.append(o[i+reflen-3:i+reflen]/c[reflen+i],h[i+reflen-3:i+reflen]/c[reflen+i]),c[i:i+reflen]/c[reflen+i]),l[i+reflen-3:i+reflen]/c[reflen+i]).reshape(-1,1)),axis=1)
				#x_data = np.concatenate((x_data,np.concatenate((o[i:reflen+i]/c[reflen+i],h[i:reflen+i]/c[reflen+i],l[i:reflen+i]/c[reflen+i],c[i:reflen+i]/c[reflen+i]),axis = 0).reshape(-1,1)),axis= 1)
				#y_data.append((c[reflen+outlen+i]-c[reflen+i])/c[reflen+i])
				if (c[reflen+outlen+i]-c[reflen+i])/c[reflen+i] > 0.04:
					y_data.append([1,0,0])
				elif (c[reflen+outlen+i]-c[reflen+i])/c[reflen+i] > -0.04:
					y_data.append([0,1,0])
				else:
					y_data.append([0,0,1])
				performance.append((c[reflen+outlen+i]-c[reflen+i])/c[reflen+i])
			print(x_data.shape, len(y_data))
			#if len(y_data) > 10000:
			#	x_data = np.swapaxes(x_data, 0, 1)
			#	return x_data, np.asarray(y_data)
	x_data = np.swapaxes(x_data, 0, 1)
	y_data = np.asarray(y_data)
	performance = np.asarray(performance)
	#y_data = np.where(np.logical_and(-0.05 < y_data, y_data < 0.05), 0, y_data)
	#print(np.where(-0.05 < y_data < 0.05))
	#print(y_data[np.where(-0.05 < y_data < 0.05)])
	return x_data, y_data, performance
			

def create_model(reflen):

	input_data = Input(shape=(reflen+9,))
	l0 = Dense(200, activation='relu')(input_data)
	l1 = Dense(160, activation='relu')(l0)
	l2 = Dense(80, activation='relu')(l1)
	l3 = Dense(20, activation='relu')(l2)
	l4 = Dense(3, activation='sigmoid')(l3)
	adam = optimizers.Adam(lr=0.0001)
	model = Model(inputs=[input_data], output=l4)
	model.compile(optimizer=adam, loss='categorical_crossentropy')
	return model
#df = pd.read_csv("./data/1101.csv")
#print(df["high"][0:20].values + df["close"][0:20].values)
#a = (np.concatenate((df["high"].replace("--",np.nan).values.astype(float),df["close"].replace("--","-1").values.astype(float)),axis=0))
#print(df["close"].replace("--",np.nan).fillna("ffill").values)
#print(a.shape)
reflen = 30
outlen = 10
#xdata_train, ydata_train = loadtrain(reflen, outlen)
#xdata_test1, ydata_test1 = extract_train_test(reflen, outlen, 1101, 1102)
#xdata_test2, ydata_test2 = extract_train_test(reflen, outlen, 1301, 1302)
#xdata_test3, ydata_test3 = extract_train_test(reflen, outlen, 2330, 2331)
xdata, ydata, performance = extract_train_test(reflen, outlen, 2301, 2302)
xdata_train = xdata[:int(xdata.shape[0]/2)]
ydata_train = ydata[:int(ydata.shape[0]/2)]
xdata_test1 = xdata[int(xdata.shape[0]/2):]
ydata_test1 = ydata[int(ydata.shape[0]/2):]

mdl = create_model(reflen)
i = 0
while True:
	mdl.fit(xdata_train, ydata_train, epochs=50, batch_size=64, shuffle=True)
	if i%500 == 499:
		adam = optimizers.Adam(lr=0.00001)
		mdl.compile(optimizer=adam, loss='categorical_crossentropy')
	
	if i%10 == 9:
		
		prd = mdl.predict(xdata_test1)
		rpt = open("result " + str(i) + ".csv", "w")
		for j in range(prd.shape[0]):
			rpt.write(str(prd[j][0]) + "," + str(prd[j][1]) + "," + str(prd[j][2]) + "," + str(performance[j]) + ",\n")
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