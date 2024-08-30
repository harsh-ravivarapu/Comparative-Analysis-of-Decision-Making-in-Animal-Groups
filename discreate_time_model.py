import numpy as np
import matplotlib.pyplot as plt
# Parameters
N = 200 # Total number of individuals
K1, K2 = 2, 10 # Parameters for weighting
r = 0.5 # Sensing range threshold
theta1, theta2 = np.random.uniform(0, 2*np.pi),
np.random.uniform(0, 2*np.pi)
N1, N2 = N // 3, N // 3 # Number of individuals in subgroups
1 and 2
N3 = N - N1 - N2 # Number of uninformed individuals
# Initialize directions and interaction weights
theta = np.random.uniform(0, 2*np.pi, N)
ajl = np.random.uniform(0, 1, (N, N)) # Interaction weights
def update_theta(theta, ajl, N1, N2, K1, theta1, theta2):
dtheta = np.zeros(N)
for j in range(N):
sum_term = sum(ajl[j, :] * np.sin(theta -
theta[j])) / N
if j < N1:
dtheta[j] = np.sin(theta1 - theta[j]) + K1 *
sum_term
elif j < N1 + N2:
dtheta[j] = np.sin(theta2 - theta[j]) + K1 *
sum_term
else:
dtheta[j] = K1 * sum_term
return dtheta
def update_ajl(ajl, theta, K2, r):
dajl = np.zeros_like(ajl)
for j in range(N):
for l in range(N):
rho_jl = abs(np.cos(0.5 * (theta[j] - theta[l])))
dajl[j, l] = K2 * (1 - ajl[j, l]) * ajl[j, l] *
(rho_jl - r)
return dajl
# Simulation settings
dt = 0.01
steps = 1000
# Simulation loop
for t in range(steps):
dtheta = update_theta(theta, ajl, N1, N2, K1, theta1,
theta2)
dajl = update_ajl(ajl, theta, K2, r)
theta += dtheta * dt
ajl += dajl * dt
# Plotting the directions
plt.figure(figsize=(10, 6))
plt.scatter(np.cos(theta), np.sin(theta), alpha=0.6)
plt.title('Directions of Individuals in the Continuous-Time
Model',fontsize=15)
plt.xlabel('Cosine Component',fontsize=15)
plt.ylabel('Sine Component',fontsize=15)
plt.show()