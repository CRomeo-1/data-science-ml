import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# predict disease progression after 1 year from 10 medical features
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ---- linear regression with gradient descent (my implementation) ----
def train_gd(X, y, lr=0.5, epochs=20000):
    w = np.zeros(X.shape[1])
    b = 0.0
    m = len(y)
    for _ in range(epochs):
        error = X @ w + b - y
        w -= lr * (X.T @ error) / m
        b -= lr * error.mean()
    return w, b


w, b = train_gd(X_train, y_train)
pred_gd = X_test @ w + b

# ---- same thing with scikit-learn to compare ----
model = LinearRegression().fit(X_train, y_train)
pred_sk = model.predict(X_test)

print(f"{'':<20}{'R2':>8}{'RMSE':>10}")
for name, pred in [("Gradient descent", pred_gd), ("scikit-learn", pred_sk)]:
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name:<20}{r2_score(y_test, pred):>8.3f}{rmse:>10.2f}")

plt.scatter(y_test, pred_gd, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="perfect prediction")
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("Linear regression - diabetes progression")
plt.legend()
plt.savefig("regression.png")
print("saved regression.png")
