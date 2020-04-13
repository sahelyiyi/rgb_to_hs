import pandas as pd
from sklearn import linear_model


class Regression():
    def __init__(self, X, y):
        self.X = pd.DataFrame(data=X)
        self.y = pd.DataFrame(data=y)
        self.lm = linear_model.LinearRegression()

        self.model = None

    def train(self):
        self.model = self.lm.fit(self.X, self.y)

    def test(self, X, y):
        predictions = self.model.predict(X)
