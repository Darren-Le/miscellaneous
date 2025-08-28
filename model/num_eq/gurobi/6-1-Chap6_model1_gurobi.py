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