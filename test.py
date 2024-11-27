from repitframework import OpenFOAM
from repitframework.Metrics import pwd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


metrics_dir = "/home/openfoam/repitframework/repitframework/Metrics/natural_convection"
def visualize_output(assets_dir:Path, timestamp, 
					data_vars:list=["U", "T"], animate="all"):
	'''
	This function is used to visualize the output of the simulation.

	Args:
	assets_dir: Path
		The path to the directory where the output numpy files are stored. 
		1. files should be in the format: U_{timestamp}.npy, T_{timestamp}.npy, etc.
		2. It must be a pathlib.Path object.
		3. The last directory of the path should be case name for the framework to work properly. e.g. 
			/home/openfoam/repitframework/repitframework/Assets/natural_convection
	timestamp: int/float
		The time at which the output is to be visualized.
	data_vars: list
		The list of variables to be visualized. Default is ["U", "T"]
	animate: bool
		1. "all". It will create the animation compiling all the results from start_time to end_time.
		2. list: It will create the animation for the given list of timestamps.
	'''
	U = np.load(assets_dir / f"U_{timestamp}.npy")
	T = np.load(assets_dir / f"T_{timestamp}.npy")
	
	u_x = U[:,0].reshape(200,200,order="F")
	u_y = U[:,1].reshape(200,200,order="F")
	T = T.reshape(200,200,order="F")
	
	# flip the rows from top to bottom
	u_x = np.flipud(u_x)
	u_y = np.flipud(u_y)
	total_u = np.sqrt(u_x**2 + u_y**2)
	T = np.flipud(T)
	
	fig, ax = plt.subplots(1, 2, figsize=(10, 5))
	# u_x = ax[0].imshow(u_x, cmap="coolwarm")
	# fig.colorbar(u_x, ax=ax[0])
	# ax[0].set_title("Velocity X")
	# u_y = ax[1].imshow(u_y, cmap="coolwarm")
	# fig.colorbar(u_y, ax=ax[1])
	# ax[1].set_title("Velocity Y")
	total_u = ax[0].imshow(total_u, cmap="coolwarm")
	fig.colorbar(total_u, ax=ax[0])
	ax[0].set_title("Velocity Magnitude")
	T = ax[1].imshow(T, cmap="coolwarm")
	fig.colorbar(T, ax=ax[1])
	ax[1].set_title("Temperature")
	fig.tight_layout()
	fig.suptitle("At time={}s".format(timestamp))
	plt.savefig(metrics_dir / f"output_{timestamp}.png")
	
if __name__ == "__main__":
    # # start the simulation
    # start_time = 0
    # end_time = 6
    # solver_dir = "/home/openfoam/repitframework/repitframework/Solvers/natural_convection"
    # assets_dir = "/home/openfoam/repitframework/repitframework/logs"
    # # update the controlDict file
    # is_time_updated = OpenFOAM.utils.update_time_foamDictionary(
    #     solver_dir=solver_dir,
    #     present_time=start_time,
    #     end_time=end_time,
    #     time_step=1,
    # )
    # assets_path = OpenFOAM.utils.run_solver(solver_dir=solver_dir,
    #                                         assets_dir=assets_dir)
    # # save the results to assets dire
    # is_converted_to_nmpy = OpenFOAM.utils.parse_to_numpy(solver_dir=solver_dir,
    #                                                     assets_dir=assets_path,del_dirs=False)

    # visualize the output
    assets_dir = Path("/home/openfoam/repitframework/repitframework/logs/natural_convection")
    for i in range(1,7):
        visualize_output(assets_dir=assets_dir, timestamp=i)
