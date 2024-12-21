import numpy as np

def f(x):
    """目标函数"""
    return x[0]**2 + 4 * x[1]**2

def grad_f(x):
    """函数的梯度"""
    return np.array([2*x[0], 8*x[1]])

def hessian_f(x):
    """函数的海森矩阵"""
    return np.array([[2, 0],
                     [0, 8]])

def steepest_descent(x0, tol=1e-6, max_iter=1000):
    """最速下降法实现"""
    x = x0.copy()
    for i in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"最速下降法在第 {i} 次迭代中收敛。")
            return x
        # 正确的步长计算
        numerator = np.dot(grad, grad)
        H = hessian_f(x)
        denominator = np.dot(grad, H @ grad)
        alpha = numerator / denominator
        x = x - alpha * grad
    print("最速下降法在最大迭代次数内未收敛。")
    return x

def newton_method(x0, tol=1e-6, max_iter=1000):
    """牛顿法实现"""
    x = x0.copy()
    H = hessian_f(x)
    H_inv = np.linalg.inv(H)
    for i in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"牛顿法在第 {i} 次迭代中收敛。")
            return x
        x = x - H_inv @ grad
    print("牛顿法在最大迭代次数内未收敛。")
    return x

def bfgs(x0, tol=1e-6, max_iter=1000):
    """BFGS 算法实现"""
    n = len(x0)
    H = np.eye(n)  # 初始近似逆海森矩阵
    x = x0.copy()
    for i in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"BFGS 算法在第 {i} 次迭代中收敛。")
            return x
        p = -H @ grad
        # 对于二次函数，精确线搜索步长
        alpha = np.dot(grad, grad) / np.dot(p, hessian_f(x) @ p)
        s = alpha * p
        x_new = x + s
        grad_new = grad_f(x_new)
        y = grad_new - grad
        if np.dot(y, s) > 1e-10:
            rho = 1.0 / np.dot(y, s)
            I = np.eye(n)
            Hy = H @ y
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x = x_new
    print("BFGS 算法在最大迭代次数内未收敛。")
    return x

# 初始点
x0 = np.array([1.0, 1.0])

# 最速下降法
xmin_sd = steepest_descent(x0)
print("最速下降法求得的极小点:", xmin_sd)

# 牛顿法
xmin_newton = newton_method(x0)
print("牛顿法求得的极小点:", xmin_newton)

# BFGS 算法
xmin_bfgs = bfgs(x0)
print("BFGS 算法求得的极小点:", xmin_bfgs)
