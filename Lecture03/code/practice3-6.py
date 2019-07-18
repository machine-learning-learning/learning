import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


train_size = 20
test_size = 12
train_X = np.random.uniform(low=0, high=1.2, size=train_size)
test_X = np.random.uniform(low=0.1, high=1.3, size=test_size)
train_y = np.sin(train_X * 2 * np.pi) + np.random.normal(0, 0.2, train_size)
test_y = np.sin(test_X * 2 * np.pi) + np.random.normal(0, 0.2, test_size)
poly = PolynomialFeatures(6) # 次数は6
train_poly_X = poly.fit_transform(train_X.reshape(train_size, 1))
test_poly_X = poly.fit_transform(test_X.reshape(test_size, 1))
model = Ridge(alpha=1)   #ハイパーパラメータ
model.fit(train_poly_X, train_y)
train_pred_y = model.predict(train_poly_X)
test_pred_y = model.predict(test_poly_X)
print("平均学習誤差")
print(mean_squared_error(train_pred_y, train_y))
print("平均検証誤差")
print(mean_squared_error(test_pred_y, test_y))

plt.scatter(train_X, train_y, color="k")
plt.scatter(test_X, test_y, color="r")
plt.ylim([train_y.min() - 1, train_y.max() + 1])
#plt.plot(xx, yy, color="k")

xx = np.linspace(train_X.min(), train_X.max(), 300)
train_poly_XX = poly.fit_transform(xx.reshape(300, 1))
yy=model.predict(train_poly_XX)
#yy = np.array([model.predict(u) for u in train_poly_XX])
plt.plot(xx, yy, color="k")

plt.show()
