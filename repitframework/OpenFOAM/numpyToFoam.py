import numpy as np
from pathlib import Path
import subprocess
import re

def parse_numpy(data: np.ndarray) -> str:
    """
    Convert a NumPy array to a string representation suitable for OpenFOAM field files.

    Parameters:
    - data (np.ndarray): The NumPy array to convert.

    Returns:
    - str: The string representation of the data.

    Raises:
    - ValueError: If the data shape is not supported.
    """
    if data.ndim == 1:
        # 1D array of scalars
        return  '\n'.join(map(str, data))
    elif data.ndim == 2:
        if data.shape[1] == 1:
            # 2D array with a single column, treat as scalar field
            return '\n'.join(map(str, data[:, 0]))
        else:
            # 2D array with multiple columns, treat as vector field
            lines = ['(' + ' '.join(map(str, row)) + ')' for row in data]
            return '\n'.join(lines)
    else:
        raise ValueError("Data shape not supported. Aborting conversion from numpy to OpenFOAM.")


def numpyToFoam( variables:list, 
                latestCFD_time, 
                latestML_time,
                solver_dir:Path,
                assets_path:Path) -> bool:
    """
    This function takes a numpy file and writes it to an OpenFOAM file.
    Example: 
    If we have a numpy file U_3.npy, we can write it to the OpenFOAM file U at time t=3.

    Args:
    variables: The OpenFOAM variables list.
    latestCFD_time: The final time step for which OpenFOAM file is already present.
    latestML_time:  The final time step for which ML simulation is present. 
    solver_dir: The directory of the solver insider "Solvers" directory e.g: "Solvers/natural_convection"
    assets_path: The path to the assets directory where the numpy files are stored: e.g: "Assets/natural_convection"


    **************** NOTE ****************
    The latestCFD_time should be the time step for which the OpenFOAM file is already present. 
    Because, we need to copy format to the present time step for which we are trying to run the 
    simulation.
    **************************************

    Returns:
    True if the function executes successfully.
    """
    latestCFD_time_dir = Path(str(solver_dir) + f"/{latestCFD_time}") # time directory for current time
    latestML_time_dir  = Path(str(solver_dir) + f"/{latestML_time}") # time directory for next time (latestCFD_time + 1)
    subprocess.run(["cp", "-r", latestCFD_time_dir, latestML_time_dir])    # copy the contents of latest CFD simulation time to the latest ML simulation time.

    for variable in variables:
        openfoam_var_path = Path(str(latestML_time_dir) + f"/{variable}") # openfoam variable path: where we write the numpy data    
        numpy_file_path =   Path(assets_path, f"{variable}_{float(latestML_time)}.npy") # numpy file path: where the numpy data is stored
        # numpy file processing:
        data = np.load(numpy_file_path)
        data_str = "(\n" + parse_numpy(data) + "\n)\n;" # convert numpy data to OpenFOAM format

        with open(openfoam_var_path, "r") as file:
            foam_data_temp = file.read()
            foam_data = re.sub(r'(location\s*)"([^"]*)"',rf'\1"{latestML_time}"',foam_data_temp) # update the location to the next time step
            foam_data = re.sub(r'\([\s\S]*?\)\n;', f'{data_str}',foam_data,count=1) # Update the data in the OpenFOAM file; count=1 to replace only the first occurrence.

        with open(openfoam_var_path, "w") as file:
            file.write(foam_data)
    return True    

    
if __name__ == "__main__":
    from utils import run_solver, update_time_foamDictionary
    from repitframework import config

    latestCFD_time:int = 2
    latestML_time:int = 5
    time_step:int = 1
    openfoam_config = config.OpenfoamConfig()
    solver_dir:Path = openfoam_config.solver_dir
    assets_path: Path = openfoam_config.assets_dir

    numpyToFoam(openfoam_config.data_vars, latestCFD_time,latestML_time, solver_dir, assets_path)

    # check_time_update:bool = update_time_foamDictionary(solver_dir=solver_dir,
    #                                                 present_time=latestML_time,
    #                                                 end_time=latestML_time + 2*time_step,
    #                                                 time_step=time_step)
    
    # if check_time_update:
    #     run_solver()