import os
import numpy as np
import Ofpp

# ==========================================
# 1. Physical Constants & Setup
# ==========================================
L = 1.0          # Characteristic length (m)
delta_T = 19.6   # Temperature difference (K)

# Calculate k from thermoPhysicalProperties
Cp = 1004.4
mu = 1.831e-05
Pr = 0.705
k = (Cp * mu) / Pr  # Thermal conductivity W/(m K)

# Grid setup 
base_dir = "/home/shilaj/shilaj_data/repitframework/order_of_accuracy/smaller_timesteps"
grids = {
    "coarse": os.path.join(base_dir, "coarse_grid/natural_convection_case1_3D"),
    "mid": os.path.join(base_dir, "mid_grid/natural_convection_case1_3D"),
    "fine": os.path.join(base_dir, "fine_grid/natural_convection_case1_3D")
}

# Grid Refinement Ratio (Assumes you used 34, 51, 76 as discussed)
r = 1.5  

# Time constraints
t_start = 10.0
t_end = 20.0

# ==========================================
# 2. Helper Functions
# ==========================================
def get_valid_time_directories(case_dir):
    """Filters OpenFOAM time directories between t_start and t_end."""
    valid_dirs = []
    if not os.path.exists(case_dir):
        raise FileNotFoundError(f"Directory not found: {case_dir}")
        
    for d in os.listdir(case_dir):
        full_path = os.path.join(case_dir, d)
        if os.path.isdir(full_path):
            try:
                t_val = float(d)
                if t_start <= t_val <= t_end:
                    valid_dirs.append((t_val, full_path))
            except ValueError:
                # Ignore non-time directories like 'constant' or 'system'
                pass
                
    # Sort by time value
    valid_dirs.sort(key=lambda x: x[0])
    return [path for _, path in valid_dirs]

def calculate_time_averaged_nu(case_dir):
    """Extracts wallHeatFlux, calculates Nu for each timestep, and averages them."""
    time_dirs = get_valid_time_directories(case_dir)
    
    if not time_dirs:
        raise ValueError(f"No valid time directories found in {case_dir} between {t_start}s and {t_end}s.")
        
    nu_time_series = []
    
    for t_dir in time_dirs:
        flux_file = os.path.join(t_dir, "wallHeatFlux")
        
        if not os.path.exists(flux_file):
            continue 
            
        # Parse the boundary field using Ofpp
        boundary_data = Ofpp.parse_boundary_field(flux_file)
        
        # Ofpp parses dictionary keys as byte strings in Python 3
        hot_wall_key = b'hot' if b'hot' in boundary_data else 'hot'
        
        if hot_wall_key not in boundary_data:
            raise KeyError(f"Could not find 'hot' boundary in {flux_file}")
            
        patch_data = boundary_data[hot_wall_key]
        
        # Extract the array of face values
        value_key = b'value' if b'value' in patch_data else 'value'
        flux_values = np.array(patch_data[value_key])
        
        # Calculate spatial average heat flux for this timestep
        # We take the absolute value in case surface normals make the flux negative
        q_avg = np.mean(np.abs(flux_values))
        
        # Calculate Nusselt number for this timestep
        nu = (q_avg * L) / (k * delta_T)
        nu_time_series.append(nu)
        
    if not nu_time_series:
        raise ValueError(f"wallHeatFlux files were missing in the time directories of {case_dir}.")
        
    # Return the time-averaged Nusselt number
    return np.mean(nu_time_series)

# ==========================================
# 3. Execution & GCI Math
# ==========================================
try:
    print("Extracting data and calculating Time-Averaged Nusselt Numbers...")
    phi_3 = calculate_time_averaged_nu(grids["coarse"])
    print(f"Coarse Grid Nu (phi_3): {phi_3:.5f}")
    
    phi_2 = calculate_time_averaged_nu(grids["mid"])
    print(f"Mid Grid Nu (phi_2):    {phi_2:.5f}")
    
    phi_1 = calculate_time_averaged_nu(grids["fine"])
    print(f"Fine Grid Nu (phi_1):   {phi_1:.5f}")

    # Step 1: Calculate differences
    eps_32 = phi_3 - phi_2
    eps_21 = phi_2 - phi_1

    # Using the simplified formula for constant r
    p = (1.0 / np.log(r)) * np.abs(np.log(np.abs(eps_32 / eps_21)))
    # p = (1.0 / np.log(r)) * np.abs(np.log(np.abs(eps_32 / eps_21))) #TODO: convert back to original formula if needed.


    # Step 3: Extrapolated value
    phi_ext12 = ( (r**p * phi_1) - phi_2 ) / (r**p - 1.0)
    phi_ext23 = ( (r**p * phi_2) - phi_3 ) / (r**p - 1.0)

    # Step 4: Approximate relative error
    e_a12 = np.abs((phi_1 - phi_2) / phi_1)
    e_a23 = np.abs((phi_2 - phi_3) / phi_2)

    # Step 5: Grid Convergence Index (GCI) with Factor of Safety = 1.25
    Fs = 1.25
    gci_12 = (Fs * e_a12) / (r**p - 1.0) * 100.0
    gci_23 = (Fs * e_a23) / (r**p - 1.0) * 100.0

    print("\n" + "="*40)
    print("GRID CONVERGENCE INDEX (GCI) RESULTS")
    print("="*40)
    print(f"Apparent Order of Convergence (p): {p:.4f}")
    print(f"Extrapolated Asymptotic Nu (1-2):  {phi_ext12:.5f}")
    print(f"Extrapolated Asymptotic Nu (2-3):  {phi_ext23:.5f}")
    print(f"Approximate Relative Error (e_a):  {e_a12 * 100.0:.3f}%")
    print(f"GCI (Fine Grid):                   {gci_12:.3f}%")
    print(f"GCI (Mid Grid):                    {gci_23:.3f}%")
    print("="*40)
    
except Exception as e:
    print(f"An error occurred: {e}")