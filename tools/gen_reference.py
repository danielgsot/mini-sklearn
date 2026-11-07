"""Genera datos de referencia con scikit-learn real.

Uso:  pip install scikit-learn numpy
      python tools/gen_reference.py

Crea tests/data/linreg_reference.csv con X, y y las predicciones de sklearn.
Mas adelante tus tests en C++ leeran ese CSV y compararan.
"""
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X = rng.normal(50, 10, size=(200, 2))
y = 0.5 * X[:, 0] + 0.2 * X[:, 1] + 10 + rng.normal(0, 1, 200)

Xs = StandardScaler().fit_transform(X)
model = LinearRegression().fit(Xs, y)
pred = model.predict(Xs)

print("coef:", model.coef_, "intercept:", model.intercept_, "R2:", model.score(Xs, y))

os.makedirs("tests/data", exist_ok=True)
np.savetxt("tests/data/linreg_reference.csv",
           np.column_stack([X, y, pred]),
           delimiter=",", header="x0,x1,y,sklearn_pred", comments="")
