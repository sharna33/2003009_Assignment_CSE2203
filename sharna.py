import math
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def differenceTable(population):
    # make a 2d array
    # take difference
    r = []
    r.append(population)
    for i in range (1, len(population)):
        r_temp =[]
        for j in range (0, len(population)-i):
            delta = r[i-1][j+1] - r[i-1][j]
            r_temp.append(delta)
        r.append(r_temp)
        # print(r)
    return r

def newtonInterpolation(diffTable, years, populations, x):
    p = (x - years[0]) / (years[1] - years[0])
    result = populations[0]     # f(x) = y0
    backP = p
    for i in range(1, len(diffTable)):
        result += (p * diffTable[i][0]) / math.factorial(i)
        p *= (backP - i)
    return result

def newtonRaphson(f_x, f_dash_x, x0, target_pop, e = 0.0001):
    x1 = x0 - (f_x(x0, target_pop)/f_dash_x(x0))
    while abs(x1-x0) >= e:
        x0 = x1
        if (f_dash_x(x0) == 0):
            print("Derivative is 0.")
            return None
        x1 = x0 - (f_x(x0, target_pop)/f_dash_x(x0))
    return round(x1, 4)


def f_x(x, target_pop):
    # population(t) - target_pop = 0
    return -131165*x**3/48 + 264301127*x**2/16 - 1597629941995*x/48 + 357657700062273/16 - target_pop

def f_dash_x(x):
    return -131165*x**2/16 + 264301127*x/8 - 1597629941995/48

def curveEqn(x, m, c):
    return m*x + c

def curveFit(x, y):
    return curve_fit(curveEqn, x, y)[0]



url = 'https://raw.githubusercontent.com/sharna33/2003009_Assignment_CSE2203/main/2003009.csv'
data = pd.read_csv(url)


# years = [2009, 2011, 2013, 2015]
# population = [149079155, 152511195, 156207391, 160036578]

years = data[data.columns[0]].tolist()
population = data[data.columns[1]].tolist()

print("Given Data: ")
for year, pop in zip(years, population):
    print(f"Population in {year}: {pop}")

diffTable = differenceTable(population)

last_year = years[len(years)-1]
inputYear = np.arange(last_year+1, last_year+11)
newTonInter = [newtonInterpolation(diffTable, years, population, year) for year in inputYear]
print("\nNewton Interpolation: ")
for year, pop in zip(inputYear, newTonInter):
    print(f"Population in {year}: {pop}")

newtonRaph = [newtonRaphson(f_x, f_dash_x, last_year, pop) for pop in newTonInter]
print("\nNewton Raphson: ")
for year, pop in zip(newtonRaph, newTonInter):
    print(f"In {year} ({round(year)}) Year Population: {pop}")

m, c = curveFit(years, population)
curveFitted = curveEqn(inputYear, m, c)
print("\nCurve Fitting: ")
for year, pop in zip(inputYear, curveFitted):
    print(f"Population in {year}: {pop}")

plt.plot(years, population, 'co-', label = "Given Data")
# plt.plot(inputYear, newTonInter, 'ro-', label = "Newton Interpolation")
# plt.plot(newtonRaph, newTonInter, 'go-', label = "Newton Raphson")
plt.plot(inputYear, curveFitted, 'bo-', label = "Curve Fitting")
plt.legend()
plt.show()
