from repitframework.OpenFOAM.utils import run_solver, update_time_foamDictionary, parse_to_numpy
from repitframework.OpenFOAM.numpyToFoam import numpyToFoam
from repitframework.config import OpenfoamConfig

if __name__ == "__main__":
    openfoam_config = OpenfoamConfig()

    update_time_foamDictionary(solver_dir=openfoam_config.solver_dir, present_time=0,
                               end_time=2, time_step=1)
    
    # # To run the solver
    numpy_save_path = run_solver(solver_dir=openfoam_config.solver_dir)

    # Convert the OpenFOAM output to numpy
    print("Converting OpenFOAM output to numpy...")
    is_converted_to_numpy:bool = parse_to_numpy(solver_dir=openfoam_config.solver_dir, 
                                  assets_dir=numpy_save_path,del_dirs=True)
    
    print(
    '''
    Assume: \n
    1. We run the OpenFOAM for 0-2 time steps.\n 
    2. We predict from 3-5 time steps using ML model.\n
    3. Now, we have to convert numpy file for 5th time step to OpenFOAM format. 
        to run the simulation again for 6-7th time steps.\n 
    ''')

    latestCFD_time:int = 2
    latestML_time:int = 5
    solver_dir = openfoam_config.solver_dir
    assets_dir = openfoam_config.assets_dir   
    
    is_converted_from_numpy:bool = numpyToFoam(variables=openfoam_config.data_vars,
                                                latestCFD_time=latestCFD_time,
                                                latestML_time=latestML_time,
                                                solver_dir=solver_dir,
                                                assets_path=assets_dir)
    
    # Update the time in controlDict file
    update_time_foamDictionary(solver_dir=solver_dir, present_time=latestML_time,
                               end_time=7, time_step=1)
    
    # Run the solver again
    run_solver(solver_dir=solver_dir)