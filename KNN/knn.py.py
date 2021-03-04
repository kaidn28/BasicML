
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_label = datasets.load_iris()
#print(data_label.keys())

data = data_label.data
labels = data_label.target

X_train, X_test,y_train, y_test = train_test_split(data, labels, random_state = 42, test_size=0.33)

#print(X_train)
#print(y_test)

def distance(x1,x2):
  sum = 0
  for i in range(len(x1)):
    sum+= (x1[i]-x2[i])*(x1[i]-x2[i])
  return np.sqrt(sum)

def knn(k):  
  y_predict = []
  for test_point in X_test:
    # Tính khoảng cách từ điểm đang xét đến tất cả các điểm trong tập train, và nhãn của các điểm đó
    distances = [[distance(test_point, train_point), y_train[i]] for i,train_point in enumerate(X_train)]

    sorted_distances = sorted(distances, key= lambda x: x[0])
    #print(sorted_distances)

    #Lấy ra nhãn của 5 điểm gần nhất và voting    
    nearest_k_neighbors = sorted_distances[:k][1]
    number_of_0 = 0
    number_of_1 = 0
    number_of_2 = 0
    for j in nearest_k_neighbors:
      if j == 0:
        number_of_0+=1
      elif j == 1:
        number_of_1+=1
      else:
        number_of_2+=1
    most = np.argmax([number_of_0, number_of_1, number_of_2])
    y_predict.append(most)
  return y_predict

y_predict = knn(10)
#print(y_predict)

print(accuracy_score(y_predict, y_test))

