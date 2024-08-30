import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 500  # Total number of individuals
a = 1.0  # Minimum desired distance for collision avoidance
r = 5.0  # Local interaction range for alignment
p = 0.2  # Proportion of informed individuals
q = 0.5  # Weighting term for balancing preferred direction and social interaction
speed = 1.0  # Speed of each individual

# Initialize positions and directions
positions = np.zeros((N, 2))  # Start from (0,0)
directions = np.array([[1, 1] for _ in range(N)])  # Initial direction along 45-degree line
directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
speeds = np.full(N, speed)
informed = np.random.rand(N) < p

# Function to normalize a vector
def normalize(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(norm != 0, v / norm, v)

# Movement and decision-making logic
def update_directions(positions, directions, informed, a, r, q, g1, g2, group1_indices, group2_indices):
    new_directions = np.zeros_like(directions)

    for i in range(N):
        # Collision Avoidance
        avoidance = np.sum([
            (positions[j] - positions[i]) / np.linalg.norm(positions[j] - positions[i])
            if np.linalg.norm(positions[j] - positions[i]) != 0 else np.zeros(2)
            for j in range(N) if i != j and np.linalg.norm(positions[j] - positions[i]) < a
        ], axis=0)

        # Alignment with Neighbors
        alignment = np.sum([
            directions[j] / np.linalg.norm(directions[j])
            if np.linalg.norm(directions[j]) != 0 else np.zeros(2)
            for j in range(N) if i != j and np.linalg.norm(positions[j] - positions[i]) < r
        ], axis=0)

        # Combine avoidance and alignment
        d_i = avoidance + alignment
        d_i_hat = normalize(d_i.reshape(1, -1))[0]

        # Influence of Informed Individuals
        if i in group1_indices:
            combined_direction = d_i_hat + q * g1
            new_directions[i] = normalize(combined_direction.reshape(1, -1))[0]
        elif i in group2_indices:
            combined_direction = d_i_hat + q * g2
            new_directions[i] = normalize(combined_direction.reshape(1, -1))[0]
        else:
            new_directions[i] = d_i_hat

    return new_directions

# Move initially in the 45-degree direction for 4 steps
for _ in range(4):
    positions += directions * speeds[:, None]

# Split the informed individuals into two groups
half_informed = int(np.sum(informed) / 2)
group1_indices = np.where(informed)[0][:half_informed]
group2_indices = np.where(informed)[0][half_informed:]

# Example preferred directions for informed groups
g1 = np.array([1, 0])  # Direction vector for the first informed group
g2 = np.array([0, 1])  # Direction vector for the second informed group

# Initialize a list to store the history of positions
positions_history = [positions.copy()]

# Main simulation loop with position history recording
for step in range(200):
    directions = update_directions(positions, directions, informed, a, r, q, g1, g2, group1_indices, group2_indices)
    positions += directions * speeds[:, None]
    positions_history.append(positions.copy())

# Function to plot positions with trajectories
def plot_positions_with_trajectories(positions_history, group1_indices, group2_indices, step, scale_factor=1.0):
    plt.figure(figsize=(10, 10))  # Increase plot size

    for i in range(N):
        trajectory = np.array([pos[i] for pos in positions_history[:step+1]]) * scale_factor
        if i in group1_indices:
            plt.plot(trajectory[:, 0], trajectory[:, 1], c='red', alpha=0.6, linewidth=0.5)  # Trajectory for subgroup 1
        elif i in group2_indices:
            plt.plot(trajectory[:, 0], trajectory[:, 1], c='blue', alpha=0.6, linewidth=0.5)  # Trajectory for subgroup 2
        else:
            plt.plot(trajectory[:, 0], trajectory[:, 1], c='green', alpha=0.6, linewidth=0.5)  # Trajectory for rest of the flock

    plt.xlim(0, 100 * scale_factor)
    plt.ylim(0, 100 * scale_factor)
    plt.title(f'Flock Positions and Trajectories at Step {step}',fontsize=15)
    plt.xlabel('X Position',fontsize=15)
    plt.ylabel('Y Position',fontsize=15)
    plt.show()

# Use the updated plotting function in the simulation loop
scale_factor = 1.5  # Adjust as needed for better visualization
for step in range(200):
    if step % 20 == 0:
        plot_positions_with_trajectories(positions_history, group1_indices, group2_indices, step, scale_factor)
