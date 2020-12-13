import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
data = pd.read_csv("ASII Historical Data.csv")
data2 = pd.read_csv("Jakarta Stock Exchange Composite Index Historical Data.csv")
import copy

window = 5
window2 = 5
window3 = 30

def datacleanup(name,dropVol = False):
	#csv to pandas
	data = pd.read_csv(name)
	data = (data.drop('Date',axis = 1))
	if dropVol:
		data = data.drop('Vol.', axis = 1)
	else:
		data['Vol.'] = data['Vol.'].replace({'K': '*1e3', 'M': '*1e6', 'B':'*1e9'}, regex=True).map(pd.eval).astype(int)
		data['Vol.'] = data['Vol.'].astype(float)/1000000
	
	data['Change %'] = data['Change %'].str.replace('%','').astype(float)
	data = data.drop('Change %',axis = 1)
	data['Price'] = data['Price'].str.replace(',','').astype(float)
	data['High'] = data['High'].str.replace(',','').astype(float)
	data['Low'] = data['Low'].astype(str).str.replace(',','').astype(float)
	data['Open'] = data['Open'].str.replace(',','').astype(float)
	data = data[::-1]
	datamin = (data.min()['Price'])
	alldatamin = data.min()
	actuallabel = data['Price']
	actuallabel = actuallabel[-365+window:]
	normalize = data.max()-data.min()
	label = data[:365]
	label = label['Price']
	data = (data-data.min())/(data.max()-data.min())
	train = data[:-365]
	test = data[-365:]
	initial = data
	return data,actuallabel,normalize,train,test,initial,datamin,alldatamin
	

def timetopredict(seq,model,day):
	model.eval()
	out = 0
	tempseq = copy.deepcopy(seq)
	for i in range(day):
		#print(tempseq)
		result = model(tempseq)
		temp = tempseq.tolist()
		temp[0].pop(0)
		temp[0].append(result.tolist()[0])
		tempseq = torch.tensor(temp)
		out = result
	return out
	
#LSTM class
class LSTM(nn.Module):
	def __init__(self, input_size, hidden_layer_size, output_size, batch_size):
		super().__init__()
		self.hidden_dim = hidden_layer_size

		self.batch_size = batch_size
		self.lstm = nn.LSTM(input_size, hidden_layer_size,batch_first= True)

		self.linear = nn.Linear(hidden_layer_size, output_size)

	def forward(self, input_seq):
		h0 = torch.zeros(1, input_seq.size(0), self.hidden_dim).requires_grad_()

		c0 = torch.zeros(1, input_seq.size(0), self.hidden_dim).requires_grad_()
		out, (hn, cn) = self.lstm(input_seq, (h0.detach(), c0.detach()))
		out = self.linear(out[:, -1, :]) 
		return out

#create sequence
def create_seq(input_data,label_data,window):
	seq = []
	L = len(input_data)
	for i in range(L-window):
		trainseq = input_data[i:i+window]
		trainlabel = label_data[i+window:i+window+1]
		seq.append((trainseq,trainlabel))
	return seq
	
data,actuallabel,normalize,train,test,initial,datamin,alldatamin = datacleanup("ASII Historical Data.csv")
data2,actuallabel2,normalize2,train2,test2,initial2,datamin2,alldatamin2 = datacleanup("Jakarta Stock Exchange Composite Index Historical Data.csv",dropVol = True)

model = None
#model = torch.load("Documents6")
if model is None:
	model = LSTM(5,100,1,10)
	loss = nn.MSELoss()
	optimizer = optim.SGD(model.parameters(), lr = 0.01)

	seq = create_seq(torch.FloatTensor(train.values),torch.FloatTensor(train['Price'].values),window)
	train_seq= DataLoader(seq, batch_size = 10)
	for epoch in range(200):
		totalMSE = 0
		percent = 0
		for data,label in train_seq:
			model.zero_grad()
			model.hidden = (torch.zeros(1, 1, model.hidden_dim),torch.zeros(1, 1, model.hidden_dim))
			result = model(data)
			#result = torch.reshape(result,(data.shape[0],1,5))
			egloss = loss(result,label)
			egloss.backward(retain_graph= True)
			optimizer.step()
			totalMSE += egloss
			percent+=1
		
		#print("epoch:" + str(epoch)+" loss:" +str(totalMSE))
		torch.save(model,"Documents6")



model2 = None
model2 = torch.load("Documents8")
if model2 is None:
	model2 = LSTM(4,100,1,10)
	loss = nn.MSELoss()
	optimizer = optim.SGD(model2.parameters(), lr = 0.01)

	seq2 = create_seq(torch.FloatTensor(train2.values),torch.FloatTensor(train2['Price'].values),window)
	train_seq2 = DataLoader(seq2, batch_size = 10)
	for epoch in range(100):
		totalMSE = 0
		percent = 0
		for data,label in train_seq2:
			model2.zero_grad()
			model2.hidden = (torch.zeros(1, 1, model2.hidden_dim),torch.zeros(1, 1, model2.hidden_dim))
			result = model2(data)
			#result = torch.reshape(result,(data.shape[0],1,5))
			egloss = loss(result,label)
			egloss.backward(retain_graph= True)
			optimizer.step()
			totalMSE += egloss
			percent+=1
		
		print("epoch:" + str(epoch)+" loss:" +str(totalMSE))
		torch.save(model2,"Documents8")

tens1 = []
seq = create_seq(torch.FloatTensor(train.values),torch.FloatTensor(train['Price'].values),window)
train_seq= DataLoader(seq, batch_size = 10)
for data,label in train_seq:
	with torch.no_grad():
		result = model(data)
		for i in result.tolist():
			tens1.append(i)
tens2 = []
train2 = train2[:len(train)]
seq2 = create_seq(torch.FloatTensor(train2.values),torch.FloatTensor(train2['Price'].values),window)
train_seq2 = DataLoader(seq2, batch_size = 10)
for data,label in train_seq2:
	with torch.no_grad():
		result = model2(data)
		for i in result.tolist():
			tens2.append(i)

train3 = torch.cat((torch.FloatTensor(tens1),torch.FloatTensor(tens2)),1)
print(train3)
model3 = None
model3 = torch.load("Documents9")
if model3 is None:
	model3 = LSTM(2,100,1,10)
	loss = nn.MSELoss()
	optimizer = optim.SGD(model3.parameters(), lr = 0.01)
	seq3 = create_seq(train3,torch.FloatTensor(train['Price'].values),1)
	train_seq3 = DataLoader(seq3, batch_size = 10)
	for epoch in range(400):
		totalMSE = 0
		percent = 0
		for data,label in train_seq3:
			model3.zero_grad()
			model3.hidden = (torch.zeros(1, 1, model3.hidden_dim),torch.zeros(1, 1, model3.hidden_dim))
			result = model3(data)
			#result = torch.reshape(result,(data.shape[0],1,5))
			egloss = loss(result,label)
			egloss.backward(retain_graph= True)
			optimizer.step()
			totalMSE += egloss
			percent+=1
		
		print("epoch:" + str(epoch)+" loss:" +str(totalMSE))
		torch.save(model3,"Documents9")
		
model2.eval()
testseq2 = create_seq(torch.FloatTensor(test2.values),torch.FloatTensor(test2['Price'].values),window)
test_seq2 = DataLoader(testseq2,batch_size =1)
actual2 = []
prediction2 = []
model.eval()
testseq = create_seq(torch.FloatTensor(test.values),torch.FloatTensor(test['Price'].values),window)
test_seq = DataLoader(testseq,batch_size =1)
actual = []
prediction = []
prediction2norm = []
predictionnorm = []
model3.eval()
prediction3 = []
loss1 = 0
loss3 = 0

loss = nn.MSELoss()
for data,label in test_seq2:
	with torch.no_grad():
		result = model2(data)
		prediction2.append(result.tolist()[0][0]*normalize2['Price']+datamin2)
		prediction2norm.append(result.tolist()[0])
for data,label in test_seq:
	with torch.no_grad():
		result = model(data)
		loss1+= abs(label-result.tolist()[0][0])
		prediction.append(result.tolist()[0][0]*normalize['Price']+datamin)
		predictionnorm.append(result.tolist()[0])
testseq3 = torch.cat((torch.FloatTensor(predictionnorm),torch.FloatTensor(prediction2norm)),1)
testseq3 = create_seq(testseq3,torch.FloatTensor(test['Price'].values),1)
test_seq3 = DataLoader(testseq3,batch_size = 1)
for data,label in test_seq3:
	with torch.no_grad():
		result = model3(data)
		prediction3.append(result.tolist()[0][0]*normalize['Price']+datamin)
		loss3+= abs(label-result.tolist()[0][0])
actual = actuallabel.tolist()
actual2 = actuallabel2.tolist()
print(loss1/len(test))
print(loss3/len(test))
print(loss1/len(test)*normalize['Price'])
print(loss3/len(test)*normalize['Price'])
		
x = 0
pred2 = []
for i,j in test_seq:
	#pred2.append(timetopredict(i,model,5).tolist()[0][0]*normalize['Price']+datamin)
	if(5+x >= len(actual)):
		break
	#print(actual[5+x])
	x+=1
	

plt.plot(actual,label = "actual")
plt.plot(prediction, label = "prediction")
plt.legend(loc="upper left")
arr = [[5550,5650,5700,5525,98.38],[5650,5525,5675,5500,89.57],[5500,5825,5850,5550,117.61],[5800,5725,5850,5725,40.93],[5725,5725,5825,5700,33.70]]
arr.reverse()
arr2 = [[5783.33,5773.56,5795.84,5745.88,25.14],[5759.92,5669.66,5759.92,5669.66,24.83],[5679.25,5735.14,5770.66,5666.76,32.54],[5701.03,5677.30,5710.38,5667.37,24.05],[5652.76,5583.33,5652.76,5583.33,18.99]]
array = pd.DataFrame(arr,columns=['Price','Open','High','Low','Vol.'])
array2 = pd.DataFrame(arr2,columns=['Price','Open','High','Low','Vol.'])
array2 = array2.drop('Vol.',axis = 1)
array = (array-alldatamin)/normalize
array2 = (array2-alldatamin2)/normalize2
tmr = torch.FloatTensor(array.values)
tmr = DataLoader([tmr],batch_size=1)
tmr2 = torch.FloatTensor(array2.values)
tmr2 = DataLoader([tmr2],batch_size=1)
tempmr = []
for i in tmr:
	tmr_pred = model(i)
	tempmr.append(tmr_pred.tolist()[0][0])
	print(tmr_pred.tolist()[0][0]*normalize['Price']+datamin)
for i in tmr2:
	tmr_pred = model2(i)
	tempmr.append(tmr_pred.tolist()[0][0])
	print(tmr_pred.tolist()[0][0]*normalize2['Price']+datamin2)
	
tmr3 = torch.FloatTensor([tempmr])
tmr3 = DataLoader([tmr3],batch_size=1)
for i in tmr3:
	print(model3(i).tolist()[0][0]*normalize['Price']+datamin)

plt.show()
plt.plot(actual2, label = "actual")
plt.plot(prediction2, label = "prediction")
plt.show()
plt.plot(actual, label = "actual")
plt.plot(prediction3, label = "prediction")
plt.show()