import torch
import torch.nn as torch_nn
import numpy as np
import os

class PlumeSurrogatePINN(torch_nn.Module):
    def __init__(self):
        super(PlumeSurrogatePINN, self).__init__()
        # Input Layer: x, y, u, D mapped across 5 deep geometric perceptrons
        # Output: Normalized Geographic Concentration (C)
        self.network = torch_nn.Sequential(
            torch_nn.Linear(4, 64),
            torch_nn.Tanh(),
            torch_nn.Linear(64, 64),
            torch_nn.Tanh(),
            torch_nn.Linear(64, 64),
            torch_nn.Tanh(),
            torch_nn.Linear(64, 1)
        )

    def forward(self, x_in):
        return self.network(x_in)

def physics_loss(output, x, y, u, v, D):
    """
    Computes analytical differentiation tensors recursively mapping the 
    Ground Truth outputs against standard Advection-Diffusion Equations.
    """
    dC_dx = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    dC_dy = torch.autograd.grad(output, y, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    
    d2C_dx2 = torch.autograd.grad(dC_dx, x, grad_outputs=torch.ones_like(dC_dx), create_graph=True)[0]
    d2C_dy2 = torch.autograd.grad(dC_dy, y, grad_outputs=torch.ones_like(dC_dy), create_graph=True)[0]

    # ADE Residual (v is effectively 0 for x-aligned wind physics in symmetric flow)
    residual = (u * dC_dx + v * dC_dy) - D * (d2C_dx2 + d2C_dy2)
    return torch.mean(residual**2)

def train_pinn(epochs=100):
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'synthetic_plume_data.npy')
        dataset = np.load(data_path)
    except FileNotFoundError:
        print("Dataset missing! Re-run generate_pinn_data.py to compile the Tensors first.")
        return

    # Extract boundaries
    X_train = torch.tensor(dataset[:, :4], dtype=torch.float32, requires_grad=True)
    C_truth = torch.tensor(dataset[:, 4:5], dtype=torch.float32)

    model = PlumeSurrogatePINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_loss_fn = torch_nn.MSELoss()

    print(f"Propagating PINN surrogate across {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Partition parameters for PyTorch Autograd Graph mapping
        x = X_train[:, 0:1]
        y = X_train[:, 1:2]
        u = X_train[:, 2:3]
        D = X_train[:, 3:4]
        v = torch.zeros_like(u) # Cross-wind lateral shift parameter constraint

        inputs = torch.cat([x, y, u, D], dim=1)
        C_pred = model(inputs)

        # 1. Standard Dataset Loss Constraints
        loss_data = mse_loss_fn(C_pred, C_truth)

        # 2. Physics-Informed Residual Boundaries (The PINN factor)
        loss_phys = physics_loss(C_pred, x, y, u, v, D)

        lambda_phys = 1e-4
        total_loss = loss_data + (lambda_phys * loss_phys)

        total_loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Total Loss: {total_loss.item():.6f} | Physics Residual: {loss_phys.item():.6f}")

    # Compile the model matrix safely into ONNX cross-platform boundaries
    export_path = os.path.join(os.path.dirname(__file__), '..', 'plume_surrogate.onnx')
    
    dummy_input = torch.randn(1, 4, requires_grad=True)
    torch.onnx.export(
        model, dummy_input, export_path,
        export_params=True, opset_version=10, 
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Successfully compiled the surrogate model: Output path -> {export_path}")

if __name__ == '__main__':
    train_pinn(epochs=100)
