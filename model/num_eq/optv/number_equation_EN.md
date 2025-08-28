# Diophantine Equation Problem (Non-convex MIQCP)

We model and solve a Diophantine equation problem using the OptVerse solver.

## 1. Problem Description

This chapter examines a Diophantine equation problem:

Given $a, b, c$ satisfying the following relationship:
$$
\frac{a}{b+c}+\frac{b}{a+c}+\frac{c}{a+b}=4
$$

Find a set of positive integer solutions to the equation.

Using mathematical programming, we can transform the equation into the following optimization model:
$$
\begin{array}{lll}
\min & 1 & \\
\text { s.t. } & \frac{x_1}{x_2+x_3}+\frac{x_2}{x_1+x_3}+\frac{x_3}{x_1+x_2}=4 & \\
& x_i \geqslant 1, & \forall i=1,2,3 \\
& x_i \in \mathbb{Z}_{+}, & \forall i=1,2,3
\end{array}
$$

where $x_i$ represents the positive integer solutions to the equation.

Since the constraints contain fractions, we cannot directly use OptVerse to solve this model. Below we present three transformation methods.

## 2. Method 1: Transformation Using Auxiliary Variables

We introduce auxiliary variables $m_1, m_2, m_3$ and set them equal to the three components on the left side of the constraint. After simplification:
$$
\begin{aligned}
& x_1=m_1\left(x_2+x_3\right) \\
& x_2=m_2\left(x_1+x_3\right) \\
& x_3=m_3\left(x_1+x_2\right)
\end{aligned}
$$

Based on this, the original equation can be transformed to:
$$
m_1+m_2+m_3=4
$$

This yields the following equivalent Mixed Integer Quadratic Constrained Programming (MIQCP) model:

$$
\begin{array}{ll}
\min & 1\\
\text{s.t.}& m_1+m_2+m_3=4\\
& x_1=m_1\left(x_2+x_3\right)  \\
& x_2=m_2\left(x_1+x_3\right)  \\
& x_3=m_3\left(x_1+x_2\right)  \\
& m_i \geqslant 0,  \forall i=1,2,3 \\
& x_i \geqslant 1,  \forall i=1,2,3 \\
& x_i \in \mathbb{Z}_{+},  \forall i=1,2,3
\end{array}
$$

Here is the Python code using OptVerse to solve this MIQCP:

```python
from optvpy import *

# Create optimization environment and model
env = OPTVEnv()
model = OPTVModel(env)

# Create decision variables (a, b, c as positive integers)
a = model.AddVar(lb=1, vtype=OPTV_INTEGER, name="a")
b = model.AddVar(lb=1, vtype=OPTV_INTEGER, name="b")
c = model.AddVar(lb=1, vtype=OPTV_INTEGER, name="c")
m1 = model.AddVar(lb=0, vtype=OPTV_CONTINUOUS, name="m1")
m2 = model.AddVar(lb=0, vtype=OPTV_CONTINUOUS, name="m2")
m3 = model.AddVar(lb=0, vtype=OPTV_CONTINUOUS, name="m3")

# Add linear constraint
model.AddConstr(m1 + m2 + m3 - 4 == 0, name="sum_constraint")

# Add quadratic constraints
model.AddQConstr(a - m1 * (b + c), -OPTV_INF, 0.0, name="a_rhs")
model.AddQConstr(a - m1 * (b + c), 0.0, OPTV_INF, name="a_lhs")

model.AddQConstr(b - m2 * (c + a), -OPTV_INF, 0.0, name="b_rhs")
model.AddQConstr(b - m2 * (c + a), 0.0, OPTV_INF, name="b_lhs")

model.AddQConstr(c - m3 * (a + b), -OPTV_INF, 0.0, name="c_rhs")
model.AddQConstr(c - m3 * (a + b), 0.0, OPTV_INF, name="c_lhs")

# Set objective function
model.SetObjective(1, OPTVSense.MINIMIZE)

# Solve the model
model.Optimize()

# Output results
if model.STATUS == OPTV_OPTIMAL:
    print('a = ', a.X)
    print('b = ', b.X)
    print('c = ', c.X)
    print('Verification:')
    result = a.X/(b.X+c.X) + b.X/(a.X+c.X) + c.X/(a.X+b.X)
    print('a/(b+c) + b/(a+c) + c/(a+b) =', result)
else:
    print('No optimal solution found')
```

Executing this code yields a solution. Note that due to numerical precision issues, the results may contain small errors.

## 3. Method 2: Eliminating Division Operations

To eliminate numerical issues and improve solution accuracy, we attempt to remove division operations, converting the equation entirely to multiplication and addition. We multiply both sides of the original equation by the non-zero factor $(b+c)(a+c)(a+b)$:

$$
\begin{aligned}
& \frac{a(b+c)(a+c)(a+b)}{b+c}+\frac{b(b+c)(a+c)(a+b)}{a+c}+\frac{c(b+c)(a+c)(a+b)}{a+b} \\
& =4(b+c)(a+c)(a+b)
\end{aligned}
$$

After simplification, the equation is equivalent to:
$$
\begin{aligned}
& a^3+a^2 b+a^2 c+3 a b c+a b^2+b^3+b^2 c+b c^2+a c^2+c^3 \\
& =4\left(a^2 b+a^2 c+a b^2+b^2 c+b c^2+a c^2+2 a b c\right)
\end{aligned}
$$

Therefore, the original problem can be transformed into the following nonlinear integer programming model:

$$
\begin{array}{ll}
\min & 1 \\
\text { s.t. } & x_1^3+x_1^2 x_2+x_1^2 x_3+3 x_1 x_2 x_3+x_1 x_2^2+x_2^3+x_2^2 x_3+x_2 x_3^2+ \\
& x_1 x_3^2+x_3^3=4\left(x_1^2 x_2+x_1^2 x_3+x_1 x_2^2+x_2^2 x_3+x_2 x_3^2+x_1 x_3^2+2 x_1 x_2 x_3\right) \\
& x_i \geqslant 1, \quad \forall i=1,2,3 \\
& x_i \in \mathbb{Z}_{+}, \quad \forall i=1,2,3
\end{array}
$$

Note that the constraint contains cubic terms, requiring auxiliary variables for transformation. For example, for the term $x_1 x_2 x_3$, we introduce auxiliary variables $u$ and $w$ with constraints $u=x_1 x_2$ and $w=u x_3$, which converts the cubic term into two quadratic constraints.

Here is the Python code using OptVerse to solve the transformed MIQCP:

```python
from optvpy import *

# Create optimization environment and model
env = OPTVEnv()
model = OPTVModel(env)

# Create decision variables (all integer variables)
a = model.AddVar(lb=1, vtype=OPTV_INTEGER, name="a")
b = model.AddVar(lb=1, vtype=OPTV_INTEGER, name="b") 
c = model.AddVar(lb=1, vtype=OPTV_INTEGER, name="c")
# Auxiliary variables for quadratic and cubic terms
m1 = model.AddVar(vtype=OPTV_INTEGER, name="m1") 
m2 = model.AddVar(vtype=OPTV_INTEGER, name="m2") 
m3 = model.AddVar(vtype=OPTV_INTEGER, name="m3") 
m4 = model.AddVar(vtype=OPTV_INTEGER, name="m4") 
lhs = model.AddVar(vtype=OPTV_INTEGER, name="lhs")
rhs = model.AddVar(vtype=OPTV_INTEGER, name="rhs")

# Auxiliary variable constraints
model.AddQConstr(m1 - a * a, -OPTV_INF, 0.0, name="a_squared_rhs")
model.AddQConstr(m1 - a * a, 0.0, OPTV_INF, name="a_squared_lhs")

model.AddQConstr(m2 - b * b, -OPTV_INF, 0.0, name="b_squared_rhs")
model.AddQConstr(m2 - b * b, 0.0, OPTV_INF, name="b_squared_lhs")

model.AddQConstr(m3 - c * c, -OPTV_INF, 0.0, name="c_squared_rhs")
model.AddQConstr(m3 - c * c, 0.0, OPTV_INF, name="c_squared_lhs")

model.AddQConstr(m4 - a * b, -OPTV_INF, 0.0, name="ab_rhs")
model.AddQConstr(m4 - a * b, 0.0, OPTV_INF, name="ab_lhs")

# Left-hand side
model.AddQConstr(lhs - (m1 * (a + b + c) + m2 * (a + b + c) + m3 * (a + b + c) + 3 * m4 * c), -OPTV_INF, 0.0, name="lhs_rhs") 
model.AddQConstr(lhs - (m1 * (a + b + c) + m2 * (a + b + c) + m3 * (a + b + c) + 3 * m4 * c), 0.0, OPTV_INF, name="lhs_lhs")

# Right-hand side
model.AddQConstr(rhs - 4 * (m1 * (b + c) + m2 * (c + a) + m3 * (a + b) + 2 * m4 * c), -OPTV_INF, 0.0, name="rhs_rhs") 
model.AddQConstr(rhs - 4 * (m1 * (b + c) + m2 * (c + a) + m3 * (a + b) + 2 * m4 * c), 0.0, OPTV_INF, name="rhs_lhs") 

# Equality constraint
model.AddConstr(lhs - rhs == 0, name="main_constraint")

# Set objective function
model.SetObjective(1, OPTVSense.MINIMIZE)

# Solve the model
model.Optimize()

# Output results
if model.STATUS == OPTV_OPTIMAL:
    print('a = ', a.X)
    print('b = ', b.X)  
    print('c = ', c.X)
    print('Verification:')
    result = a.X/(b.X+c.X) + b.X/(a.X+c.X) + c.X/(a.X+b.X)
    print('a/(b+c) + b/(a+c) + c/(a+b) =', result)
else:
    print('No optimal solution found')
```

Executing this code reveals that the solution process is extremely slow. Despite appearing straightforward, this problem is quite challenging. The correct answer consists of three very large positive integers (with 81, 80, and 79 digits respectively):

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

With such enormous integers, it is indeed difficult for the solver to find the correct solution within reasonable time.

## 4. Method 3: Extending to General Integer Solutions

Given the difficulty in finding positive integer solutions, we consider relaxing the requirement to allow any integers and examine whether this affects the solving difficulty. The model becomes:

$$
\begin{array}{ll}
\min & 1 \\
\text { s.t. } & x_1^3+x_1^2 x_2+x_1^2 x_3+3 x_1 x_2 x_3+x_1 x_2^2+x_2^3+x_2^2 x_3+x_2 x_3^2+ \\
& x_1 x_3^2+x_3^3=4\left(x_1^2 x_2+x_1^2 x_3+x_1 x_2^2+x_2^2 x_3+x_2 x_3^2+x_1 x_3^2+2 x_1 x_2 x_3\right) \\
& x_1+x_2 \neq 0 \\
& x_1+x_3 \neq 0 \\
& x_2+x_3 \neq 0 \\
& x_i \in \mathbb{Z}, \quad \forall i=1,2,3
\end{array}
$$

However, inequality constraints such as $x_1+x_2 \neq 0$ cannot be directly modeled in mathematical programming solvers. The constraint $x_1+x_2 \neq 0$ is equivalent to:
$$
x_1+x_2>0 \text { or } x_1+x_2<0
$$

We can introduce a small positive number $\epsilon$ (since $x_i$ are integers, we can use $\epsilon=1$) and transform the constraints into absolute value form:
$$
\begin{aligned}
& \left|x_1+x_2\right| \geqslant 1 \\
& \left|x_1+x_3\right| \geqslant 1 \\
& \left|x_2+x_3\right| \geqslant 1
\end{aligned}
$$

OptVerse supports constraints with absolute value operations. To model this, we need to introduce auxiliary variables.

By introducing auxiliary variables $u_1, u_2, u_3$, the constraints become:
$$
\begin{aligned}
& u_1=x_1+x_2 \\
& u_2=x_1+x_3 \\
& u_3=x_2+x_3 \\
& \left|u_1\right|, \left|u_2\right|, \left|u_3\right| \geqslant 1
\end{aligned}
$$

Here is the Python code using OptVerse to solve the transformed model:

```python
from optvpy import *

# Create optimization environment and model
env = OPTVEnv()
model = OPTVModel(env)

# Create decision variables (can be negative integers)
a = model.AddVar(lb=-OPTV_INF, ub=OPTV_INF, vtype=OPTV_INTEGER, name="a")
b = model.AddVar(lb=-OPTV_INF, ub=OPTV_INF, vtype=OPTV_INTEGER, name="b")
c = model.AddVar(lb=-OPTV_INF, ub=OPTV_INF, vtype=OPTV_INTEGER, name="c")
# Auxiliary variables
m1 = model.AddVar(lb=-OPTV_INF, ub=OPTV_INF, vtype=OPTV_INTEGER, name="m1")
m2 = model.AddVar(lb=-OPTV_INF, ub=OPTV_INF, vtype=OPTV_INTEGER, name="m2")
m3 = model.AddVar(lb=-OPTV_INF, ub=OPTV_INF, vtype=OPTV_INTEGER, name="m3")
m4 = model.AddVar(lb=-OPTV_INF, ub=OPTV_INF, vtype=OPTV_INTEGER, name="m4")

# Auxiliary variables for absolute value constraints
u1 = model.AddVar(lb=-OPTV_INF, ub=OPTV_INF, vtype=OPTV_INTEGER, name="u1")
u2 = model.AddVar(lb=-OPTV_INF, ub=OPTV_INF, vtype=OPTV_INTEGER, name="u2")
u3 = model.AddVar(lb=-OPTV_INF, ub=OPTV_INF, vtype=OPTV_INTEGER, name="u3")

abs_u1 = model.AddVar(lb=-OPTV_INF, ub=OPTV_INF, vtype=OPTV_INTEGER, name="abs_u1")
abs_u2 = model.AddVar(lb=-OPTV_INF, ub=OPTV_INF, vtype=OPTV_INTEGER, name="abs_u2")
abs_u3 = model.AddVar(lb=-OPTV_INF, ub=OPTV_INF, vtype=OPTV_INTEGER, name="abs_u3")
lhs = model.AddVar(vtype=OPTV_INTEGER, name="lhs")
rhs = model.AddVar(vtype=OPTV_INTEGER, name="rhs")

# Auxiliary variable constraints
model.AddQConstr(m1 - a * a, -OPTV_INF, 0.0, name="a_squared_rhs")
model.AddQConstr(m1 - a * a, 0.0, OPTV_INF, name="a_squared_lhs")

model.AddQConstr(m2 - b * b, -OPTV_INF, 0.0, name="b_squared_rhs")
model.AddQConstr(m2 - b * b, 0.0, OPTV_INF, name="b_squared_lhs")

model.AddQConstr(m3 - c * c, -OPTV_INF, 0.0, name="c_squared_rhs")
model.AddQConstr(m3 - c * c, 0.0, OPTV_INF, name="c_squared_lhs")

model.AddQConstr(m4 - a * b, -OPTV_INF, 0.0, name="ab_rhs")
model.AddQConstr(m4 - a * b, 0.0, OPTV_INF, name="ab_lhs")

# Define u variables
model.AddConstr(u1 - a - b == 0, name="u1_def")
model.AddConstr(u2 - a - c == 0, name="u2_def")
model.AddConstr(u3 - b - c == 0, name="u3_def")

# Absolute value constraints: |ui| >= 1
model.AddGenConstrAbs(abs_u1, u1)
model.AddGenConstrAbs(abs_u2, u2)
model.AddGenConstrAbs(abs_u3, u3)
model.AddConstr(abs_u1 >= 0.00001)
model.AddConstr(abs_u2 >= 0.00001)
model.AddConstr(abs_u3 >= 0.00001)

# Set constraints
# Left-hand side
model.AddQConstr(lhs - (m1 * (a + b + c) + m2 * (a + b + c) + m3 * (a + b + c) + 3 * m4 * c), -OPTV_INF, 0.0, name="lhs_rhs") 
model.AddQConstr(lhs - (m1 * (a + b + c) + m2 * (a + b + c) + m3 * (a + b + c) + 3 * m4 * c), 0.0, OPTV_INF, name="lhs_lhs")
# Right-hand side
model.AddQConstr(rhs - 4 * (m1 * (b + c) + m2 * (c + a) + m3 * (a + b) + 2 * m4 * c), -OPTV_INF, 0.0, name="rhs_rhs") 
model.AddQConstr(rhs - 4 * (m1 * (b + c) + m2 * (c + a) + m3 * (a + b) + 2 * m4 * c), 0.0, OPTV_INF, name="rhs_lhs") 

model.AddConstr(lhs == rhs) # Equality constraint

# Set objective function
model.SetObjective(1, OPTVSense.MINIMIZE)

# Solve the model
model.Optimize()

# Output results
if model.STATUS == OPTV_OPTIMAL:
    print('Found an integer solution:')
    print('a = ', int(a.X))
    print('b = ', int(b.X))
    print('c = ', int(c.X))
    
    # Verify solution
    a_val, b_val, c_val = int(a.X), int(b.X), int(c.X)
    if (b_val + c_val) != 0 and (a_val + c_val) != 0 and (a_val + b_val) != 0:
        result = a_val/(b_val+c_val) + b_val/(a_val+c_val) + c_val/(a_val+b_val)
        print('Verification: a/(b+c) + b/(a+c) + c/(a+b) =', result)
else:
    print('No optimal solution found, solving status:', model.STATUS)
```

Executing this code yields a feasible solution, for example:
$$
a=-1, b=11, c=4
$$

Verification by substitution:
$$
\frac{a}{b+c}+\frac{b}{a+c}+\frac{c}{a+b}=\frac{-1}{11+4}+\frac{11}{-1+4}+\frac{4}{-1+11}=4
$$

This confirms that the solution is correct.

By adjusting the solving strategy, we can obtain other feasible solutions, such as:
$$
\begin{aligned}
& a=-330, b=-120, c=30 \\
& \frac{-330}{-120+30}+\frac{-120}{-330+30}+\frac{30}{-330-120}=4
\end{aligned}
$$

Clearly, this equation has multiple integer feasible solutions.

Furthermore, we can consider a more general form:
$$
\frac{a}{b+c}+\frac{b}{a+c}+\frac{c}{a+b}=k
$$

where $k$ is any non-zero integer. We can observe how problem difficulty varies with different values of $k$. When $k=1$, the model has no feasible solution. When $k=2$, OptVerse quickly finds a feasible solution: $a=1, b=1, c=3$.

## 5. Conclusion

This example demonstrates how to model seemingly unrelated problems as mathematical programming models using a Diophantine equation as an illustration. The modeling transformation process involves several useful techniques, including converting fractional constraints to quadratic constraints, expressing cubic constraints as quadratic constraints, and transforming inequality ($\neq$) constraints to absolute value constraints.

We presented three different modeling approaches:
1. **Method 1**: Introducing auxiliary variables to transform fractions into quadratic constraints
2. **Method 2**: Using algebraic operations to eliminate division, expressing cubic terms with quadratic terms
3. **Method 3**: Relaxing to general integer solutions and handling absolute value constraints