from src.model import create_model
from src.pretprocesing import load_train_data
import numpy as np


def test(x_test,y_test):
    model = create_model()
    model.load_weights("../model/weights2-100.h5")

    correct = 0
    result = model.predict(x_test, batch_size=1, verbose=0)
    for i  in range(len(result)):
        elem =  result[i].tolist()
        index = elem.index(max(elem))
        if index == int(y_test[i]):
            correct+=1
        else:
            print(i)
    print(correct/len(x_test))



if __name__ == '__main__':
    test_data, test_label = load_train_data("../data/test.csv", "../data/test/")
    test(test_data, test_label)
