import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations

# ========== STEP 1: Select coordinates ==========
def select_coords(img_path, num_points=25):
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title(f"Click {num_points} points (cities) on the map")
    coords = plt.ginput(num_points, timeout=0)
    plt.close()
    return coords

# ========== STEP 2: Distance matrix ==========
def distance_matrix(coords):
    N = len(coords)
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist[i, j] = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
    return dist

# ========== STEP 3: K-Medoids clustering ==========
def k_medoids_clustering(coords, K=5, max_iter=100):
    N = len(coords)
    D = distance_matrix(coords)
    medoids = random.sample(range(N), K)
    
    for _ in range(max_iter):
        # Assign each city to nearest medoid
        clusters = {m: [] for m in medoids}
        for i in range(N):
            nearest = min(medoids, key=lambda m: D[i, m])
            clusters[nearest].append(i)
        
        # Update medoids
        new_medoids = []
        for m in medoids:
            cluster_points = clusters[m]
            costs = [sum(D[i, j] for j in cluster_points) for i in cluster_points]
            new_m = cluster_points[np.argmin(costs)]
            new_medoids.append(new_m)
        
        if set(new_medoids) == set(medoids):
            break
        medoids = new_medoids
    
    return medoids, clusters

# ========== STEP 4: Choose configuration with max inter-cluster distance ==========
def best_cluster_configuration(coords, K=5, trials=20):
    D = distance_matrix(coords)
    best_config = None
    best_score = -1
    
    for _ in range(trials):
        medoids, clusters = k_medoids_clustering(coords, K)
        inter_dist = np.mean([D[a, b] for a, b in combinations(medoids, 2)])
        intra_dist = np.mean([D[i, j] for m in medoids for i in clusters[m] for j in clusters[m]])
        score = inter_dist / (intra_dist + 1e-6)
        
        if score > best_score:
            best_score = score
            best_config = (medoids, clusters)
    
    return best_config

# ========== STEP 5: Plot results ==========
def plot_clusters(coords, medoids, clusters, img_path=None):
    if img_path:
        img = plt.imread(img_path)
        plt.imshow(img)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(medoids)))
    for color, m in zip(colors, medoids):
        points = np.array([coords[i] for i in clusters[m]])
        plt.scatter(points[:, 0], points[:, 1], color=color, label=f'Power Grid City {m}')
        plt.scatter(coords[m][0], coords[m][1], color='black', marker='x', s=100)
    
    plt.legend()
    plt.title("5 City Clusters — Max Intercluster, Min Intracluster Distance")
    plt.show()

# ========== STEP 6: Run everything ==========
if __name__ == "__main__":
    # img_path = "india-map.jpg"  # <-- replace with your map image path
    img_path =r"B:\PERSONAL DOCS\BUSSINESS MODEL\CASE STUDY\india-map.jpg"
    num_cities = 25
    K = 5
    
    coords = select_coords(img_path, num_cities)
    medoids, clusters = best_cluster_configuration(coords, K)
    plot_clusters(coords, medoids, clusters, img_path)
