model = polyreg.PolynomialRegression(6)
model.fit(x, y)

plt.scatter(x, y, color="k")
plt.ylim([y.min() - 1, y.max() + 1])
xx = np.linspace(x.min(), x.max(), 300)
yy = np.array([model.predict(u) for u in xx])
plt.plot(xx, yy, color="k")

num=0;
for wi in model.w_:
    print("w{0} = {1:f}".format(num,wi))
    num+=1;

plt.show()
