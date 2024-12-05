from pathlib import Path
import subprocess
import re

import numpy as np

from repitframework.config import OpenfoamConfig

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
		if data.shape[1] == 1: # 1D array
			return '\n'.join(map(str, data[:, 0]))
		elif data.shape[-1] == 2: # 2D array
			# For the vector field, OpenFOAM requires the data to be in the form of (x y z) for each row.
			# So, if we have a 2D array of shape (n, 2), we need to add a column of zeros to make it (n, 3).
			zero_columns = np.zeros((data.shape[0], 1))
			data = np.concatenate((data, zero_columns), axis=1)
			lines = ['(' + ' '.join(map(str, row)) + ')' for row in data]
			return '\n'.join(lines)
		else: # 3D array
			lines = ['(' + ' '.join(map(str, row)) + ')' for row in data]
			return '\n'.join(lines)
	else:
		raise ValueError("Data shape not supported. Aborting conversion from numpy to OpenFOAM.")


def numpyToFoam(openfoam_config:OpenfoamConfig, 
				latestML_time:float,
				latestCFD_time:int|float=None,
				variables:list=None,
				solver_dir:Path=None,
				assets_path:Path=None,
				is_ground_truth:bool=False) -> bool:
	"""
	This function takes a numpy file and writes it to an OpenFOAM file.

	Example
	-------
	If we have a numpy file U_3.npy, we can write it to the OpenFOAM file U at time t=3.

	Args
	----
	openfoam_config: The OpenFOAM configuration object.
	latestML_time:  The final time step for which ML simulation is present. 
	latestCFD_time: The final time step for which OpenFOAM file is already present.
	variables: The OpenFOAM variables list.
	solver_dir: The directory of the solver insider "Solvers" directory e.g: "Solvers/natural_convection"
	assets_path: The path to the assets directory where the numpy files are stored: e.g: "Assets/natural_convection"
	is_ground_truth: If True, it will load the ground truth data. If False, it will load the predicted data.
					 Because, for the predicted cases we will have var_timestamp_predicted.npy files.

	NOTE
	----
	The latestCFD_time should be the time step for which the OpenFOAM file is already present. 
	Because, we need to copy format to the present time step for which we are trying to run the 
	simulation.

	Returns
	-------
	True if the function executes successfully.

	Remember
	--------
	latestML_time should always be float value. Because, while saving any value to numpy, we save it as float.
	see: repitframework/OpenFOAM/utils.py: parse_numpy
	"""
	solver_dir = Path(solver_dir) if solver_dir else openfoam_config.solver_dir
	assets_path = Path(assets_path) if assets_path else openfoam_config.assets_path
	variables = variables if variables else openfoam_config.extend_variables()

	if not latestCFD_time:
		command_to_list_time_directories = ["foamListTimes", "-case", solver_dir]
		command_result = subprocess.run(command_to_list_time_directories,capture_output=True ,text=True, check=True)
		time_list = command_result.stdout.split("\n")
		time_list = [round(time,openfoam_config.round_to) for time in time_list if time.strip()]
		latestCFD_time = max(time_list)

	latestCFD_time_dir = Path.joinpath(solver_dir,f"{latestCFD_time}") # time directory for current time

	# Because, in openfoam if the time directory is float at the terminal values like; 11.0, 12.0, 13.0 are converted to 11, 12, 13.
	ml_dir_time_name = str(latestML_time).split(".")[-1]
	ml_dir_time_name = int(latestML_time) if int(ml_dir_time_name) == 0 else latestML_time
	latestML_time_dir  = Path.joinpath(solver_dir, f"{ml_dir_time_name}") # time directory for next time (latestCFD_time + 1)

	subprocess.run(["cp", "-r", latestCFD_time_dir, latestML_time_dir], check=True)    # copy the contents of latest CFD simulation time to the latest ML simulation time.
	for variable in variables:
		numpy_file_name = f"{variable}_{latestML_time}.npy" if is_ground_truth else f"{variable}_{latestML_time}_predicted.npy"
		openfoam_var_path = Path.joinpath(latestML_time_dir, f"{variable}") # openfoam variable path: where we write the numpy data    
		numpy_file_path =   Path.joinpath(assets_path, numpy_file_name) # numpy file path: where the numpy data is stored
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
	openfoam_config = OpenfoamConfig()
	numpyToFoam(openfoam_config, latestCFD_time=9, latestML_time=10.0, is_ground_truth=True)