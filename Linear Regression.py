
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('study_time_vs_score.csv')




def loss_function( m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].Study_Time
        y = points.iloc[i].Score
        total_error += (y-m*x-b)**2

    return total_error/len(points)

def gradient_descent_function(m_now, b_now, points, l):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].Study_Time
        y = points.iloc[i].Score

        m_gradient += (-2/n)*x*(y-m_now*x-b_now)
        b_gradient += (-2/n)*(y-m_now*x-b_now)

    m = m_now - l*m_gradient
    b = b_now - l*b_gradient
    return m, b

m = 0
b = 0
l = 0.01
epochs = 2000

for i in range(epochs):
    if i%100 == 0:
        print(i)
    m, b = gradient_descent_function( m, b, data, l)

print(m)
print(b)

plt.scatter(data.Study_Time, data.Score, color='blue')
plt.plot(list(range(0, 12)), [m*x+b for x in range(0,12)], color='red')
plt.xlabel('Study_Time')
plt.ylabel('Score')
plt.show()

