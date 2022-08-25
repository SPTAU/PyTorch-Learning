# Matplotlib 绘制 3D 图总结

引入模块

```py
import matplotlib.pyplot as plt
```

传入数据进行绘图

```py
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, Z)
ax.set_xlabel("xlabel")
ax.set_ylabel("ylabel")
ax.set_zlabel("zlabel")
plt.show()
```

其中 `meshgrid()` 函数是将列表 `x` 进行 Broadcast

将

```py
x = [3 3 3 3 3]
y = [4 4 4 4 4]
```

转换为

```py
x = [3 3 3 3 3
    3 3 3 3 3
    3 3 3 3 3
    3 3 3 3 3
    3 3 3 3 3]
y = [4 4 4 4 4
    4 4 4 4 4
    4 4 4 4 4
    4 4 4 4 4
    4 4 4 4 4]
```

`plot_surface()` 函数是创建一个曲面图

默认情况下，它的颜色是深浅不一的纯色，但它也支持通过提供 cmap 参数进行颜色映射。
