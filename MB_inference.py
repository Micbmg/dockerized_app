import pickle
import pandas as pd
import numpy as np
from flask import Flask, request

# Opening the classifier from the pickle file
clf = pickle.load(open('churn_model.pkl', 'rb'))

# testing purposes function
def testing():
    test_csv = 'X_test.csv'
    pred_csv = 'preds.csv'
    X_test = pd.read_csv(test_csv)
    y_pred = np.loadtxt(pred_csv)
    # .all returns True if the two numpy arrays are equal
    yesorno = (clf.predict(X_test) == y_pred).all()
    assert(yesorno)
    return


# Generating flask app
app = Flask(__name__)

# We will need to know the features needed for our training model
FEATURES = ['is_male', 'num_inters', 'late_on_payment', 'age', 'years_in_contract']

#Creating our prediction route and function
@app.route('/prediction')
def prediction():
    # We will save all our inputs in inputs array
    sample = np.empty(len(FEATURES))
    for i, ft in enumerate(FEATURES):
        sample[i] = request.args.get(ft) #get the input for feature ft
    # We then predict using our loaded clf model
    y_pred = clf.predict(sample.reshape([1,len(FEATURES)])) # here y_pred is of size 1
    return str(y_pred[0])


def main():
    testing()
    app.run(host='0.0.0.0')

if __name__ == '__main__':
    main()
