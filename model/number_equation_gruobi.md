第6章 数论方程的数学规划模型

数学规划应用十分广泛，许多问题看似与其无关，但可以通过巧妙转换，将其变成数学规划问题加以解决。本章以一个有趣的数论方程问题来介绍如何将数论方程转换为数学规划模型，并提供完整的实现代码。
6.1 问题简介

本章探讨的数论方程问题源自一个网络上流传一时的故事。有一家比萨店为了宣传和促销，提出了一个新颖的方案：出一道有趣的数学题吸引人们来解，成功解题者可免费获得一份比萨大礼包！店家经过精挑细选，选中了下面这道题。

【例6．1】已知 $a, b, c$ 满足以下关系：
$$
\frac{a}{b+c}+\frac{b}{a+c}+\frac{c}{a+b}=4
$$

求方程的一组正整数解。

文献［18］对该问题的一般形式（即考虑右端项为任意非 0 整数）进行了详细的探讨，其解法与椭圆曲线有关。不过本章不会详细介绍这种方法，而是换一种思路，用数学规划的方法来求解它。

那么，本问题如何与数学规划联系起来呢？这就要从方程本身入手了。方程（6．1）规定了 $a 、 b 、 c$ 之间的关系，相当于数学规划中的约束。而问题要求只需要找到一组正整数解即可，因此可以视为无目标函数。无目标函数在数学规划中等价于目标函数为 0 ，或者目标函数为常数。实际上，若目标函数为 0或者常数，则该数学规划模型可被视为约束规划模型，所以也可以使用一些支持约束规划求解的求解器进行求解。

基于上述思路，可以将方程（6．1）的求解转换为下面的数学规划模型：
$$
\begin{array}{lll}
\min & 1 & \\
\text { s.t. } & \frac{x_1}{x_2+x_3}+\frac{x_2}{x_1+x_3}+\frac{x_3}{x_1+x_2}=4 & \\
& x_i \geqslant 1, & \forall i=1,2,3 \\
& x_i \in \mathbb{Z}_{+}, & \forall i=1,2,3
\end{array}
$$

其中，$x_i$ 为满足方程的正整数解。

在上述模型中，约束式（6．3）含有分式，不可以直接使用COPT或Gurobi来求解。不过，可以通过模型转换，将其转换为求解器可以求解的形式。下面介绍两种转换的方法。
6.2 方法1：引入辅助变量进行转换

引入辅助变量 $m_1 、 m_2 、 m_3$ ，分别令其等于约束式（6．3）左端的3个部分，化简可得：
$$
\begin{aligned}
& x_1=m_1\left(x_2+x_3\right) \\
& x_2=m_2\left(x_1+x_3\right) \\
& x_3=m_3\left(x_1+x_2\right)
\end{aligned}
$$

基于此，方程（6．1）可以转换为：
$$
m_1+m_2+m_3=4
$$

综上，式（6．2）～式（6．5）可以等价为下面的混合整数二次约束规划模型（MIQCP）：

$\min 1$
（6．6）

s．t．$\quad m_1+m_2+m_3=4$
$$
\begin{array}{ll}
x_1=m_1\left(x_2+x_3\right) & \\
x_2=m_2\left(x_1+x_3\right) & \\
x_3=m_3\left(x_1+x_2\right) & \forall i=1,2,3 \\
m_i \geqslant 0, & \forall i=1,2,3 \\
x_i \geqslant 1, & \forall i=1,2,3
\end{array}
$$

下面使用Python调用Gurobi对上面的MIQCP进行求解，完整代码如下：

```python
from gurobipy import *

# 创建模型对象
m = Model("pissa")

# 设置非凸模型求解参数为2
m.setParam('NonConvex', 2)

# 创建决策变量(a, b, c为正整数)
a = m.addVar(lb=0, vtype=GRB.INTEGER, name="a")
b = m.addVar(lb=0, vtype=GRB.INTEGER, name="b")
c = m.addVar(lb=0, vtype=GRB.INTEGER, name="c")
m1 = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="m1")
m2 = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="m2")
m3 = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="m3")

# 添加正整数约束
m.addConstr(a >= 1, "c1")
m.addConstr(b >= 1, "c2")
m.addConstr(c >= 1, "c3")

# 添加可行性约束
m.addConstr(m1 + m2 + m3 - 4 == 0)

# 添加对应约束
m.addConstr(a == m1 * (b + c))
m.addConstr(b == m2 * (c + a))
m.addConstr(c == m3 * (a + b))

# 设置目标函数
m.setObjective(1, GRB.MINIMIZE)

# 求解模型
m.optimize()

# 输出求解结果
print('a = ', a.x)
print('b = ', b.x)
print('c = ', c.x)
```

详细的求解日志及结果如下。

求解日志及结果
$$
\begin{aligned}
& \text { Root relaxation: objective } 1.000000 \mathrm{e}+00 \text {, } 11 \text { iterations, } 0.00 \text { seconds } \\
& \text { Nodes | Current Node | Objective Bounds | Work } \\
& \text { Expl Unexpl | Obj Depth IntInf | Incumbent BestBd Gap | It/Node Time } \\
& \begin{array}{llllllll}
0 & 0 & 1.00000 & 0 & 2 & - & 1.00000 & -
\end{array} 0 \mathrm{~s} \\
& \begin{array}{lllllllll}
0 & 0 & 1.00000 & 0 & 2 & - & 1.00000 & - & - \\
0 s & 0 s
\end{array} \\
& \begin{array}{lllllllll}
0 & 2 & 1.00000 & 0 & 2 & - & 1.00000 & - & - \\
0 & 0 & 0
\end{array} \\
& \begin{array}{lllllll}
* 149784 & 721 & 365 & 1.0000000 & 1.00000 & 0.00 \% & 4.2
\end{array} \text { 1s } \\
& \text { Explored } 154950 \text { nodes (643045 simplex iterations) in } 1.47 \text { seconds } \\
& \text { Thread count vas } 16 \text { (of } 16 \text { available processors) } \\
& \text { Solution count 1: } 1 \\
& \text { Optimal solution found (tolerance } 1.00 \mathrm{e}-04 \text { ) } \\
& \text { Best objective } 1.000000000000 \mathrm{e}+00 \text {, best bound } 1.00000000000 \mathrm{e}+00 \text {, gap } 0.0000 \% \\
& \text { a: } 35.0 \\
& \text { b: } 132.0 \\
& \text { c: } 627.0
\end{aligned}
$$

根据结果可知，Gurobi的求解时间仅为 1 s ，求解结果为
$$
a=35, b=132, c=627
$$
下面来验证上述结果是否正确。为了让计算机的计算结果更加精准，将式（6．3）的左端等价转换为下面的验算式：
$$
k=\frac{x_1\left(x_1+x_3\right)\left(x_1+x_2\right)+x_2\left(x_2+x_3\right)\left(x_1+x_2\right)+x_3\left(x_2+x_3\right)\left(x_1+x_3\right)}{\left(x_1+x_2\right)\left(x_1+x_3\right)\left(x_2+x_3\right)}
$$

将求解结果代入验算式（6．14），可得：
$$
k=4.00000000184069 \neq 4
$$

出人意料的是，Gurobi得到的结果有非常微小的误差！这种现象实际上是模型求解中比较常见的数值问题，因为编程语言的数值精度有限。Gurobi中模型可行性的默认容差为 $1 \times 10^{-6}$ ，即若一组解使得约束的违背量小于等于 $1 \times 10^{-6}$ ，Gurobi就会判定该解为可行解（10），但是实际上该解是否真正可行，是否存在数值问题，需要进行进一步的验证。若要获得精度更高的解，可以设置相关的容差参数。不过，本章将尝试使用另外一种模型转换的方法来提高求解精度，也就是接下来要介绍的方法 2 。
6.3 方法2：消去除法运算

导致方法1出现数值问题的主要原因是约束式（6．3）中含有除法运算。为了消除该数值问题，提高求解精度，我们尝试消去除法运算的部分，将方程（6．1）完全转换为乘法和加法运算。在方程（6．1）两端同时乘以非 0 因式 $(b+c)(a+c)(a+b)$ 可得：
$$
\begin{aligned}
& \frac{a(b+c)(a+c)(a+b)}{b+c}+\frac{b(b+c)(a+c)(a+b)}{a+c}+\frac{c(b+c)(a+c)(a+b)}{a+b} \\
& =4(b+c)(a+c)(a+b)
\end{aligned}
$$
式中，
$$
\begin{aligned}
\text { 左边 } & =a(a+c)(a+b)+b(b+c)(a+b)+c(b+c)(a+c) \\
& =a\left(a^2+a b+a c+b c\right)+b\left(a b+b^2+a c+b c\right)+c\left(a b+b c+a c+c^2\right) \\
& =a^3+a^2 b+a^2 c+a b c+a b^2+b^3+a b c+b^2 c+a b c+b c^2+a c^2+c^3 \\
\text { 右边 } & =4(b+c)(a+c)(a+b)=4\left(a b+b c+a c+c^2\right)(a+b) \\
& =4\left(a^2 b+a^2 c+b^2 a+b^2 c+b c^2+a c^2+2 a b c\right)
\end{aligned}
$$

最终，方程（6．1）等价转换成了下面的形式：
$$
\begin{aligned}
& a^3+a^2 b+a^2 c+a b c+a b^2+b^3+a b c+b^2 c+a b c+b c^2+a c^2+c^3 \\
& =4\left(a^2 b+a^2 c+b^2 a+b^2 c+b c^2+a c^2+2 a b c\right)
\end{aligned}
$$

因此，原问题可以被等价转换为以下非线性整数规划模型。
$$
\begin{array}{ll}
\min & 1 \\
\text { s.t. } & x_1^3+x_1^2 x_2+x_1^2 x_3+x_1 x_2 x_3+x_1 x_2^2+x_2^3+x_1 x_2 x_3+x_2^2 x_3+x_1 x_2 x_3+x_2 x_3^2+ \\
& x_1 x_3^2+x_3^3=4\left(x_1^2 x_2+x_1^2 x_3+x_2^2 x_1+x_2^2 x_3+x_2 x_3^2+x_1 x_3^2+2 x_1 x_2 x_3\right) \\
& x_i \geqslant 1, \quad \forall i=1,2,3 \\
& x_i \in \mathbb{Z}_{+}, \quad \forall i=1,2,3
\end{array}
$$
注意到约束式（6．18）中含有三次项，如 $x_1^2 \cdot x_1^2 x_2 \cdot x_1 x_2 x_3$ 等，其中，像 $x_1^2 x_2 \cdot x_1 x_2 x_3$ 这样含有交叉项的三次项是无法直接建模的，需要进行一定的转换。以 $x_1 x_2 x_3$ 为例，引入辅助变量 $u$ 和 $w$ ，并加入约束 $u=x_1 x_2$ 、 $w=u x_3$ ，即可将三次项等价转换为两个二次约束。利用上述转换方法，可以将模型（6．17）～（6．20）等价转换为一个MIQCP。

下面使用Python调用Gurobi对转换后的MIQCP进行求解，完整代码如下：

```python
from gurobipy import *

# 创建模型对象
m = Model("test")

# 设置非凸模型求解参数为2
m.setParam('NonConvex', 2)

# 创建决策变量(变量均为整数)
a = m.addVar(vtype=GRB.INTEGER, name="a")
b = m.addVar(vtype=GRB.INTEGER, name="b")
c = m.addVar(vtype=GRB.INTEGER, name="c")
m1 = m.addVar(vtype=GRB.INTEGER, name="m1")
m2 = m.addVar(vtype=GRB.INTEGER, name="m2")
m3 = m.addVar(vtype=GRB.INTEGER, name="m3")
m4 = m.addVar(vtype=GRB.INTEGER, name="m4")
m5 = m.addVar(vtype=GRB.INTEGER, name="m5")
m6 = m.addVar(vtype=GRB.INTEGER, name="m6")
m7 = m.addVar(vtype=GRB.INTEGER, name="m7")
lhs = m.addVar(vtype=GRB.INTEGER, name="lhs")
rhs = m.addVar(vtype=GRB.INTEGER, name="rhs")


# 添加正整数约束
m.addConstr(a >= 1, "c1")
m.addConstr(b >= 1, "c2")
m.addConstr(c >= 1, "c3")

# 辅助变量赋值约束
m.addConstr(m1 == a * a) # a * a
m.addConstr(m2 == b * b) # b * b
m.addConstr(m3 == c * c) # c * c
m.addConstr(m4 == a * b) # a * b
m.addConstr(m5 == a * c) # a * c
m.addConstr(m6 == b * c) # b * c

# 设置约束
m.addConstr(lhs == m1 * (a + b + c) + m2 * (a + b + c) + m3 * (a + b + c) + 3 * m4 * c) # 左项式
m.addConstr(rhs == 4 * (m1 + (b + c) + m2 * (c + a) + m3 * (a + b) + m4 * c)) # 左项式
m.addConstr(lhs == rhs) # 左右项式相等

# 设置目标函数
m.setObjective(1, GRB.MINIMIZE)

# 求解模型
m.optimize()

# 输出求解结果
print('a = ', a.x)
print('b = ', b.x)
print('c = ', c.x)
```

运行代码，发现求解速度相当缓慢，Gurobi迟迟找不到可行解。当程序运行 2 小时后，求解日志显示分支切割（Branch and cut）算法已经探索了 2 亿多个节点，但是仍然没有找到任何可行解。这道题看似简单，实则难度不小。如果继续运行代码，也许会在足够长的运行时间后得到最终的结果。本节直接展示正确答案。令人惊讶的是，方程（6．1）的正确答案是 3 个非常大的正整数（分别有 81 位、 80 位和 79位），具体如下：
$$
\begin{aligned}
a= & 154476802108746166441951315019919837485664325669565 \\
& 431700026634898253202035277999, \\
b= & 368751317941299998271978115652254748254929799689719 \\
& 70996283137471637224634055579, \\
c= & 437361267792869725786125260237139015281653755816161 \\
& 3618621437993378423467772036
\end{aligned}
$$

如此巨大的整数，求解器确实很难在合理时间内得到正确的解。

6.4 拓展

既然求方程（6．1）的正整数解如此困难，那么，如果只要求找到一组整数解呢？本节就来探索这一点，看看若将问题中正整数的要求放宽为整数，求解难度是否会有变化。将模型修改为下面的形式：
$$
\begin{array}{ll}
\min & 1 \\
\text { s.t. } & x_1^3+x_1^2 x_2+x_1^2 x_3+x_1 x_2 x_3+x_1 x_2^2+x_2^3+x_1 x_2 x_3+x_2^2 x_3+x_1 x_2 x_3+x_2 x_3^2+ \\
& x_1 x_3^2+x_3^3=4\left(x_1^2 x_2+x_1^2 x_3+x_2^2 x_1+x_2^2 x_3+x_2 x_3^2+x_1 x_3^2+2 x_1 x_2 x_3\right) \\
& x_1+x_2 \neq 0 \\
& x_1+x_3 \neq 0 \\
& x_2+x_3 \neq 0 \\
& x_i \in \mathbb{Z}_{+}, \quad \forall i=1,2,3
\end{array}
$$

但是，形如 $x_1+x_2 \neq 0$ 的不等式约束无法直接使用数学规划求解器进行建模。因此，需要引入辅助变量和辅助约束进行转换。不难得出，$x_1+x_2 \neq 0$ 等价于
$$
x_1+x_2>0 \text { 或 } x_1+x_2<0
$$

但是，上文已经提到过，在数学规划求解器中，$>$ 和 $<$ 无法直接进行建模，必须将其转换成 $\geqslant \leqslant$ 或者 $=$ 。为此，可以引入一个足够小的正数 $c$（由于 $x_i$ 为整数，因此可取 $c=1$ ），将约束式（6．22）～约束式 （6．24）转换为下面的绝对值形式：
$$
\begin{aligned}
& \left|x_1+x_2\right| \geqslant 1 \\
& \left|x_1+x_3\right| \geqslant 1 \\
& \left|x_2+x_3\right| \geqslant 1
\end{aligned}
$$

Gurobi和COPT均支持含有绝对值运算的约束建模。以Gurobi为例，添加绝对值约束的接口函数名称为addGenConstrAbs。不过在 10.0 及以下的版本中，该函数接口仅支持绝对值运算中包含单个变量的情形，不支持绝对值符号中包含 2 个及以上决策变量的情形。因此，为了能够使用求解器完成建模，还需要对上面 3 个约束进行进一步的转换。

引入辅助变量 $u_1 、 u_2 、 u_3 、 u_1^{\mathrm{abs}} 、 u_2^{\mathrm{abs}} 、 u_3^{\mathrm{abs}}$ ，将上述约束转换为以下的形式：
$$
\begin{aligned}
& u_1=x_1+x_2 \\
& u_2=x_1+x_3 \\
& u_3=x_2+x_3 \\
& u_1^{\mathrm{abs}}=\left|u_1\right| \\
& u_2^{\mathrm{abs}}=\left|u_2\right| \\
& u_3^{\mathrm{abs}}=\left|u_3\right| \\
& u_1^{\mathrm{abs}}, u_2^{\mathrm{abs}}, u_3^{\mathrm{abs}} \geqslant 1
\end{aligned}
$$
至此，转换后的约束已经可以使用求解器进行直接建模了。

为了方便查看，这里将转换后的模型的完整形式展示如下：
$$
\begin{array}{ll}
\min & 1 \\
\text { s.t. } & x_1^3+x_1^2 x_2+x_1^2 x_3+x_1 x_2 x_3+x_1 x_2^2+x_2^3+x_1 x_2 x_3+x_2^2 x_3+x_1 x_2 x_3+x_2 x_3^2+ \\
& x_1 x_3^2+x_3^3=4\left(x_1^2 x_2+x_1^2 x_3+x_2^2 x_1+x_2^2 x_3+x_2 x_3^2+x_1 x_3^2+2 x_1 x_2 x_3\right) \\
& u_1=x_1+x_2 \\
& u_2=x_1+x_3 \\
& u_3=x_2+x_3 \\
& u_1^{\mathrm{abs}}=\left|u_1\right| \\
& u_2^{\mathrm{abs}}=\left|u_2\right| \\
& u_3^{\mathrm{abs}}=\left|u_3\right| \\
& u_i^{\mathrm{abs}} \geqslant 1, \\
& x_i \in \mathbb{Z}_{+}, \\
& u_i \text { 无约束, } \quad \forall i=1,2,3 \\
\forall i=1,2,3
\end{array}
$$

下面使用Python调用Gurobi对转换后的模型进行求解，完整代码如下：

```python
from gurobipy import *

# 创建模型对象
m = Model("test")

# 设置非凸模型求解参数为2
m.setParam('NonConvex', 2)

# 创建决策变量(变量均为整数)
a = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="a")
b = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="b")
c = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="c")
m1 = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="m1")
m2 = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="m2")
m3 = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="m3")
m4 = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="m4")
m5 = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="m5")
m6 = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="m6")
m7 = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="m7")
u1 = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="u1")
u2 = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="u2")
u3 = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="u3")
abs_u1 = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="abs_u1")
abs_u2 = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="abs_u2")
abs_u3 = m.addVar(lb=-GRB.INFINITY, ub=-GRB.INFINITY, vtype=GRB.INTEGER, name="abs_u3")
lhs = m.addVar(vtype=GRB.INTEGER, name="lhs")
rhs = m.addVar(vtype=GRB.INTEGER, name="rhs")


# 辅助变量赋值约束
m.addConstr(m1 == a * a) # a * a
m.addConstr(m2 == b * b) # b * b
m.addConstr(m3 == c * c) # c * c
m.addConstr(m4 == a * b) # a * b
m.addConstr(m5 == a * c) # a * c
m.addConstr(m6 == b * c) # b * c
m.addConstr(u1 == a + b)
m.addConstr(u2 == c + a)
m.addConstr(u3 == b + c)

# 设置绝对值约束
m.addGenConstrAbs(abs_u1, u1)
m.addGenConstrAbs(abs_u2, u2)
m.addGenConstrAbs(abs_u3, u3)
m.addConstr(abs_u1 >= 0.00001)
m.addConstr(abs_u2 >= 0.00001)
m.addConstr(abs_u3 >= 0.00001)

# 设置约束
m.addConstr(lhs == m1 * (a + b + c) + m2 * (a + b + c) + m3 * (a + b + c) + 3 * m4 * c) # 左项式
m.addConstr(rhs == 4 * (m1 + (b + c) + m2 * (c + a) + m3 * (a + b) + m4 * c)) # 左项式
m.addConstr(lhs == rhs) # 左右项式相等


# 设置目标函数
m.setObjective(1, GRB.MINIMIZE)

# 求解模型
m.optimize()

# 输出求解结果
print('a = ', a.x)
print('b = ', b.x)
print('c = ', c.x)
```

运行代码，可以得到模型的一个可行解：
$$
a=-1, b=11, c=4
$$

将其代入方程（6．1）进行验证，得到：
$$
\frac{a}{b+c}+\frac{b}{a+c}+\frac{c}{a+b}=\frac{-1}{11+4}+\frac{11}{-1+4}+\frac{4}{-1+11}=4
$$

可见上述结果是完全正确的。
通过调整相关求解参数，可以得到其他可行解。例如，下面的解也是一组可行解。
$$
\begin{aligned}
& a=-330, b=-120, c=30 \\
& \frac{-330}{-120+30}+\frac{-120}{-330+30}+\frac{30}{-330-120}=4
\end{aligned}
$$

可见，方程（6．1）有多组整数可行解。
进一步地，可以考虑更一般的形式，即令

式中，$k$ 是任意非 0 整数。可以通过变化 $k$ 的值来观察问题的求解难度变化。
仍考虑 $a 、 b 、 c$ 为正整数。分别将 $k$ 设置为 $1 、 2 、 3$ 进行测试。当 $k=1$ 时，模型无可行解。当 $k=2$ 时， Gurobi可以很快得到一个可行解：$a=1, b=1, c=3$ 。
6.5 总结

本章以一个数论方程为例讲解了如何将一些看似与优化无关的问题建模为数学规划模型。在模型转换的过程中，需要用到许多有用的模型转化方法，包括将分式约束转换为二次约束、将三次约束转换为二次约束、将 $\neq$ 约束转换为绝对值约束、将 $>(<)$ 约束转换为 $\geqslant(\leqslant)$ 约束以及复杂绝对值约束的转换等。熟练掌握这些方法对建模能力的提高是非常有帮助的。