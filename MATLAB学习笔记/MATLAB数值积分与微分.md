## 多项式表达式积分与微分
#### 多项式表达
```matlab
p = [5 0 -2 0 1];
x = -1:0.05:20
polyval(p,x)
```
#### 多项式相乘
```matlab
conv(p1,p2)
```


#### 多项式微分
```matlab
p = [5 0 -2 0 1];
polyder(p)
polyval(polyder(p),7)
```

#### 多项式积分
```matlab
p = [5 0 -2 0 1];
polyint(p,3) %3为积分后的常数项
polyval(polyint(p,3),7)
```

## 数值的微分与积分
#### 数值微分
* `diff()`计算一个vector元素前后之间的差异
```matlab
x = [1 2 5 2 1]
diff(x)
```
* 使用`diff()`做微分
```matlab
x0 = pi/2;h=0.1
x = [x0 x0+h];
y = [sin(x0) sin(x0+h)];
m = diff(y)./diff(x) % element wise
```

#### 数值积分
* 梯形法
```matlab
x=0:0.5:1;
y=exp(-x.^2);
z=trapz(x,y)

```

* simpson公式
```matlab
g=inline('exp(-x.^2)');
z=quad(g,-1,1)
```