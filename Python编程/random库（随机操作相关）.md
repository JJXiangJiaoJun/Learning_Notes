[TOC]
# Random模块的功能
* Random库python中用于生成**随机数**的函数库。

# Random库中常用的API

常用函数 | 作用
---|---
`random.seed(a)` | 初始化给定随机数的种子
`random.random()` | 生成一个`[0.0,1.0)`之间的随机小数
`random.randint(a,b)`|生成一个`[a,b]`之间的整数
`random.randrange(m,n,[,k])`|生成一个`[m,n)`之间以k为步长的随机整数
`uniform(a,b)`|生成一个`[a,b]`之间的随机小数
`choice(seq)`|从序列seq中随机选择一个元素
`shuffle(seq)`|将序列seq中元素随机排列
`random.sample(seq,n)`|从序列中随机选取n个元素组成新的序列


