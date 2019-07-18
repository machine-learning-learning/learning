from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# リッジ回帰

plt.scatter(x, y, color="k")
plt.ylim([y.min() - 1, y.max() + 1])
xx = np.linspace(x.min(), x.max(), 300)
poly = PolynomialFeatures(6) #次元数
train_poly_X = poly.fit_transform(x.reshape(10, 1))
test_poly_X = poly.fit_transform(xx.reshape(300, 1))

model = Ridge(alpha=1.0)
model.fit(train_poly_X, y)
yy = model.predict(test_poly_X)
plt.plot(xx, yy, color="k")

num=0;
for wi in model.coef_:
    print("w{0} = {1:f}".format(num,wi))
    num+=1;


plt.show()

