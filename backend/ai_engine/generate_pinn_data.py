import numpy as np
import os
import sys

# Mount the models directory to import legacy Gaussian Teacher
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.plume_physics import gaussian_concentration

def generate_synthetic_data(num_samples=50000):
    """
    Constructs Ground Truth advection boundaries parsing variables:
    x: 10m to 5000m downwind
    y: -500m to 500m lateral
    u (wind): 1.0 to 10.0 m/s
    Q: Static ~447,000g/s (Chlorine baseline metric)
    """
    np.random.seed(42)
    x = np.random.uniform(10, 5000, num_samples)
    y = np.random.uniform(-500, 500, num_samples)
    u = np.random.uniform(1.0, 10.0, num_samples)
    Q = np.full(num_samples, 447000.0)

    print(f"Generating Ground Truth structural boundary states over {num_samples} coordinates...")
    C = np.array([gaussian_concentration(x[i], y[i], u[i], Q[i], 'D') for i in range(num_samples)])

    # Extrapolating D (Dispersion tensor variable) mapped functionally for the PINN PyTorch residual
    # D approx (sy^2 * u) / 2x via pure Gaussian expansion
    sy = 0.128 * ((x/1000)**0.90) * 1000
    D = (sy**2 * u) / (2 * x)

    # Output dataset aligns to 5 dimensions per record
    dataset = np.stack((x, y, u, D, C), axis=1)
    
    output_dir = os.path.dirname(__file__)
    file_path = os.path.join(output_dir, 'synthetic_plume_data.npy')
    np.save(file_path, dataset)
    print(f"Dataset securely compiled and saved to {file_path}")

if __name__ == '__main__':
    generate_synthetic_data()
