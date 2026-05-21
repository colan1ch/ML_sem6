import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Определяем целевую функцию
def objective(x):
    # Предотвращаем логарифм от нуля или отрицательных чисел
    x1 = max(x[0], 1e-9)
    return (10 * np.log(x1) - 2)**2 + 10 * (x[1]**2 + 2)**2 - 10

# 2. Определяем ограничения
def constraint1(x):
    return 2 - (x[0]**2 + x[1]**2)  # g1(x) <= 0 -> 2 - x1^2 - x2^2 >= 0

# Граничные условия (x1 > 0, x2 >= 0)
bounds = ((1e-9, None), (0, None))
cons = {'type': 'ineq', 'fun': constraint1}

# 3. Численное решение
x0 = [1.0, 0.5]  # Начальная точка
res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)

print("--- РЕЗУЛЬТАТ ЧИСЛЕННОЙ ОПТИМИЗАЦИИ ---")
print(f"Стационарная точка x^e: [{res.x[0]:.4f}, {res.x[1]:.4f}]")
print(f"Минимальное значение f(x^e): {res.fun:.4f}\n")

# 4. Построение графиков
x1_vals = np.linspace(0.01, 2.0, 400)
x2_vals = np.linspace(-0.5, 2.0, 400)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Вычисляем целевую функцию на сетке
Z = (10 * np.log(X1) - 2)**2 + 10 * (X2**2 + 2)**2 - 10

plt.figure(figsize=(9, 7))

# Линии уровня
contours = plt.contour(X1, X2, Z, levels=[25, 30, 35, 50, 100, 200, 400], colors='teal', alpha=0.6)
plt.clabel(contours, inline=True, fontsize=9, fmt='%1.0f')

# Отображение допустимой области (круг x1^2 + x2^2 <= 2 при x1>0, x2>=0)
theta = np.linspace(0, np.pi/2, 200)
x1_circle = np.sqrt(2) * np.cos(theta)
x2_circle = np.sqrt(2) * np.sin(theta)

plt.fill_between(x1_circle, x2_circle, color='green', alpha=0.15, label='Допустимое множество $X$')
plt.plot(x1_circle, x2_circle, 'g--', linewidth=2, label='$x_1^2 + x_2^2 = 2$')

# Оси координат (границы x1=0 и x2=0)
plt.axhline(0, color='black', linewidth=1.2)
plt.axvline(0, color='black', linewidth=1.2)

# Точка минимума
plt.plot(res.x[0], res.x[1], 'ro', markersize=8, label=f'Минимум $x^e$ ({res.x[0]:.3f}, {res.x[1]:.3f})')

plt.title('Линии уровня функции и допустимое множество (Вариант 1.10)')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(0, 1.8)
plt.ylim(-0.2, 1.6)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right')

# Сохраняем график для вставки в LaTeX
plt.savefig('plot_opt.png', dpi=300)
plt.show()