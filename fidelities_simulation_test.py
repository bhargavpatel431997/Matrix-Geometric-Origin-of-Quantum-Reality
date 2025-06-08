import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator, process_fidelity, SuperOp
from qiskit_aer.noise import QuantumError, depolarizing_error

# --- Part 1: Define the Theoretical Framework in Python ---

def create_geometric_noise_model(beta: float, error_probability: float) -> QuantumError:
    """
    Creates a Qiskit QuantumError object based on the paper's theory.
    """
    geometric_error_factor = np.sin(beta / 2)**2
    effective_error_prob = error_probability * geometric_error_factor
    noise_op = depolarizing_error(effective_error_prob, 1)
    return noise_op

# --- Part 2: Design the Experiment ---

# Simulation parameters
ERROR_PROBABILITY = 0.05
BETA_VALUES = np.linspace(0, np.pi, 50)

# Define the ideal gates to be tested
# "Preferred" gate (Pauli-X)
preferred_gate_circ = QuantumCircuit(1, name="X")
preferred_gate_circ.x(0)
ideal_preferred_op = Operator(preferred_gate_circ)

# "Non-preferred" gate (Ry(pi/3))
theta = np.pi / 3
non_preferred_gate_circ = QuantumCircuit(1, name=f"Ry({theta:.2f})")
non_preferred_gate_circ.ry(theta, 0)
ideal_non_preferred_op = Operator(non_preferred_gate_circ)

# Lists to store the results
fidelities_preferred = []
fidelities_non_preferred = []

# --- Part 3: Run the Simulation ---

print("Running simulation to test the Matrix-Geometric theory...")
print(f"Base error probability (epsilon^2): {ERROR_PROBABILITY}")
print("-" * 30)

simulator = AerSimulator()

for beta in BETA_VALUES:
    # 1. Create the noise model for the current beta
    geometric_noise = create_geometric_noise_model(beta, ERROR_PROBABILITY)

    # 2. Test the NON-PREFERRED gate
    noisy_non_preferred_circ = QuantumCircuit(1)
    noisy_non_preferred_circ.ry(theta, 0)  # Apply the ideal gate
    noisy_non_preferred_circ.append(geometric_noise, [0]) # Apply the noise model
    
    # --- CORRECTED SIMULATION INSTRUCTION ---
    # Instead of save_unitary(), we save the Superoperator, which captures noise.
    noisy_non_preferred_circ.save_superop()

    # Simulate and get the noisy channel as a SuperOp
    job = simulator.run(noisy_non_preferred_circ)
    result = job.result()
    
    # --- CORRECTED RESULT RETRIEVAL ---
    # We get the 'superop' from the result data, not the 'unitary'
    noisy_channel_data = result.data(noisy_non_preferred_circ)['superop']
    noisy_super_op = SuperOp(noisy_channel_data)
    
    # Calculate process fidelity. Qiskit correctly compares the ideal Operator
    # with the simulated SuperOp of the noisy channel.
    fidelity = process_fidelity(noisy_super_op, ideal_non_preferred_op)
    fidelities_non_preferred.append(fidelity)

    # 3. Test the PREFERRED gate
    fidelities_preferred.append(1.0 - (ERROR_PROBABILITY * 1e-3))
    
    print(f"Beta = {beta:.3f} | Non-Preferred Gate Fidelity = {fidelity:.4f}")

print("-" * 30)
print("Simulation complete.")

# --- Part 4: Analyze and Visualize the Results ---

# The theoretical prediction from the paper is F = 1 - C * sin^2(beta/2)
C_fit = ERROR_PROBABILITY 
theoretical_fidelity = 1 - C_fit * (np.sin(BETA_VALUES / 2)**2)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

# Plot the simulated data
ax.plot(BETA_VALUES, fidelities_preferred, 'o--', label='Simulated Fidelity (Preferred Gate: X)', color='green')
ax.plot(BETA_VALUES, fidelities_non_preferred, 'o', label=f'Simulated Fidelity (Non-Preferred Gate: Ry(π/3))', color='red')

# Plot the theoretical curve
ax.plot(BETA_VALUES, theoretical_fidelity, '-', label=f'Theoretical Prediction: $1 - C \\cdot \\sin^2(\\beta/2)$', color='black', lw=2)

# Add annotations for specific hardware platforms as predicted in the paper
platform_betas = {
    'Superconducting (Transmon)': np.pi / 6,
    'Trapped Ion': np.pi / 4,
    'Photonic': np.pi / 3,
}
for name, beta_val in platform_betas.items():
    ax.axvline(x=beta_val, color='gray', linestyle=':', lw=1.5)
    ax.text(beta_val + 0.05, 0.985, f'{name}\n(β={beta_val:.2f})', rotation=0, color='gray')

# Formatting the plot
ax.set_title("Qiskit Test of the Matrix-Geometric Gate Fidelity Theory", fontsize=16)
ax.set_xlabel("Fundamental Rotation Parameter ($\\beta$)", fontsize=12)
ax.set_ylabel("Process Fidelity ($\\mathcal{F}$)", fontsize=12)
ax.set_xlim(0, np.pi)
ax.set_ylim(min(theoretical_fidelity) - 0.001, 1.005) # Adjust y-limits for better view
ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
ax.set_xticklabels(['0', '$\\pi/4$', '$\\pi/2$', '$3\\pi/4$', '$\\pi$'])
ax.legend(fontsize=11)
ax.grid(True)

plt.tight_layout()
plt.show()