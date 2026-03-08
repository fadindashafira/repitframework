from pathlib import Path
import subprocess
from datetime import datetime
from typing import List, Union
import timeit

import Ofpp
import numpy as np

# Assuming OpenfoamConfig is imported from your framework
# from repitframework.config import OpenfoamConfig

class OpenfoamUtils:
    def __init__(self, openfoam_config, solver_dir: Path = None, assets_dir: Path = None):
        self.config = openfoam_config
        
        # Enforce Path objects and default to config if not provided
        self.solver_dir = Path(solver_dir) if solver_dir else Path(self.config.solver_dir)
        self.assets_dir = Path(assets_dir) if assets_dir else Path(self.config.assets_dir)
        
        self.assets_path = self._setup_assets_path()
        self.mesh_type = self._get_mesh_type()
        self.solver_type = self._read_solver_type()
        
        # Look for num_processors in config, default to 1 (serial)
        self.num_processors = getattr(self.config, 'num_processors', 1)

    def _setup_assets_path(self) -> Path:
        """Creates and returns the case-specific assets directory."""
        assets_path = self.assets_dir / self.solver_dir.name
        assets_path.mkdir(parents=True, exist_ok=True)
        return assets_path
    
    def _get_mesh_type(self) -> str:
        """Determines mesh type by inspecting the system directory."""
        if getattr(self.config, 'mesh_type', None):
            return self.config.mesh_type
        
        system_dir = self.solver_dir / "system"
        if (system_dir / "blockMeshDict").exists():
            return "blockMesh"
        if (system_dir / "snappyHexMeshDict").exists():
            return "snappyHexMesh"
            
        raise ValueError("Mesh type not recognized. Please set it manually in OpenfoamConfig.")
    
    def _read_solver_type(self) -> str:
        """Reads the application type from system/controlDict."""
        if getattr(self.config, 'solver_type', None):
            return self.config.solver_type

        command = [
            "foamDictionary", "-case", str(self.solver_dir),
            "-entry", "application", "-value", "system/controlDict"
        ]
        try:
            output = self.run_subprocess(command, capture_output=True, text=True)
            return output.strip()
        except subprocess.CalledProcessError as e:
            self.config.logger.exception(f"Error reading solver type: {e}")
            return ""

    def _run_mesh_utility(self):
        """Creates the mesh if it doesn't already exist."""
        polyMesh_dir = self.solver_dir / "constant" / "polyMesh"
        if polyMesh_dir.exists():
            return "Mesh already exists! Skipping generation."
            
        self.config.logger.debug(f"Creating mesh using {self.mesh_type}...")
        return self.run_subprocess([self.mesh_type, "-case", str(self.solver_dir)], capture_output=True, text=True)

    def decompose_case(self):
        """Decomposes the mesh and fields for parallel processing."""
        self.config.logger.debug(f"Decomposing case for {self.num_processors} processors...")
        # -force is used to overwrite any existing processor directories
        return self.run_subprocess(["decomposePar", "-force", "-case", str(self.solver_dir)], capture_output=True, text=True)

    def reconstruct_case(self):
        """Reconstructs the parallel results back into single time directories."""
        self.config.logger.debug("Reconstructing parallel results...")
        return self.run_subprocess(["reconstructPar", "-case", str(self.solver_dir)], capture_output=True, text=True)

    @staticmethod
    def generate_intervals(start_time: float, end_time: float, time_step: float, round_to: int = 2) -> list:
        """Generates consistent time intervals, avoiding numpy float drift."""
        time_list = []
        current_time = start_time
        while current_time <= end_time:
            time_list.append(round(current_time, round_to))
            current_time = round(current_time + time_step, round_to)
        return time_list

    @staticmethod
    def run_subprocess(command_list: list, capture_output: bool = True, text: bool = True) -> str:
        """Wrapper for subprocess calls."""
        result = subprocess.run(command_list, capture_output=capture_output, text=text, check=True)
        return result.stdout

    @staticmethod
    def parse_to_numpy(
        config,
        start_time: float,
        end_time: float,
        solver_dir: Path = None,
        save_path: Path = None,
        variables: list = None,
        write_interval: float = 0.01,
        del_dirs: bool = False
    ) -> Path:
        """Parses OpenFOAM internal fields to numpy arrays using Ofpp."""
        solver_dir = Path(solver_dir) if solver_dir else Path(config.solver_dir)
        
        if not save_path:
            save_path = Path(str(solver_dir).replace("Solvers", "Assets"))
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        variables = variables or config.get_variables()
        time_list = OpenfoamUtils.generate_intervals(start_time, end_time, write_interval, config.round_to)

        # Clean integer casting for folder names (e.g., '10' instead of '10.0')
        time_folders = [str(int(t)) if float(t).is_integer() else str(t) for t in time_list]

        for time_folder in time_folders:
            time_dir = solver_dir / time_folder
            if not time_dir.exists():
                config.logger.warning(f"Time directory missing: {time_dir}")
                continue
                
            for var in variables:
                target_file = time_dir / var
                if not target_file.exists():
                    config.logger.warning(f"Variable file missing: {target_file}")
                    continue
                    
                data = Ofpp.parse_internal_field(str(target_file))
                config.logger.debug(f"Parsed {var}_{time_folder} --> {data.shape}")
                
                np_file = save_path / f"{var}_{float(time_folder)}.npy"
                np.save(np_file, data)

        if del_dirs and len(time_folders) > 1:
            # Delete all but the last time directory
            times_to_remove = ",".join(time_folders[:-1])
            OpenfoamUtils.run_subprocess([
                "foamListTimes", "-case", str(solver_dir), "-rm", "-time", times_to_remove
            ], capture_output=True, text=True)
            config.logger.debug(f"Deleted time directories: {times_to_remove}")
            
        return save_path

    @staticmethod
    def update_control_dict(config, solver_dir: Path, start_time: float, end_time: float, write_interval: float) -> bool:
        """Updates controlDict time settings using foamDictionary."""
        time_string = f"startTime={start_time},endTime={end_time},writeInterval={write_interval}"
        command = [
            "foamDictionary", "-case", str(solver_dir),
            "-set", time_string, "system/controlDict"
        ]
        
        output = OpenfoamUtils.run_subprocess(command, capture_output=True, text=True)
        config.logger.debug(f"Time updated successfully: {output}")
        return True
    
    @staticmethod
    def update_subdomains(config, solver_dir: Path, num_processors: int) -> bool:
        """Updates decomposeParDict with the number of subdomains for parallel runs."""
        command = [
            "foamDictionary", "-case", str(solver_dir),
            "-set", f"numberOfSubdomains={num_processors}", "system/decomposeParDict"
        ]
        
        output = OpenfoamUtils.run_subprocess(command, capture_output=True, text=True)
        config.logger.debug(f"Subdomains updated successfully: {output}")
        return True

    def run_solver(
        self, 
        start_time: float = None,
        end_time: float = None,
        write_interval: float = None,
        save_to_numpy: bool = True,
        del_dirs: bool = False
    ) -> float:
        """Orchestrates mesh generation, decomposition (if parallel), running, and reconstruction."""
        # Setup variables
        write_interval = write_interval or self.config.write_interval
        start_time = round(start_time or self.config.start_time, self.config.round_to)
        end_time = round(end_time or self.config.end_time, self.config.round_to)

        # 1. Update Time
        self.update_control_dict(self.config, self.solver_dir, start_time, end_time, write_interval)
        
        # 2. Mesh Generation
        mesh_result = self._run_mesh_utility()
        self.config.logger.debug(f"Mesh Output: {mesh_result}")

        # 3. Setup Command (Serial vs Parallel)
        if self.num_processors > 1:
            updated_subdomains = self.update_subdomains(self.config, self.solver_dir, self.num_processors)
            print(f"Updated decomposeParDict with {self.num_processors} subdomains: {updated_subdomains}")
            decompose_case_result = self.decompose_case()
            self.config.logger.debug(f"DecomposePar Output: {decompose_case_result}")
            command = ["mpirun", "-np", str(self.num_processors), self.solver_type, "-parallel", "-case", str(self.solver_dir)]
            print(f"Running OpenFOAM {self.solver_type} in PARALLEL on {self.num_processors} cores...")
        else:
            command = [self.solver_type, "-case", str(self.solver_dir)]
            print(f"Running OpenFOAM {self.solver_type} in SERIAL...")

        # 4. Run Solver
        solver_start_time = timeit.default_timer()
        solver_result = self.run_subprocess(command, capture_output=True, text=True)
        elapsed_time = timeit.default_timer() - solver_start_time
        self.config.logger.debug(f"Solver Output: {solver_result}")

        # 5. Reconstruct (if Parallel)
        if self.num_processors > 1:
            self.reconstruct_case()

        # 6. Parse to Numpy
        if save_to_numpy:
            self.parse_to_numpy(
                config=self.config,
                start_time=start_time,
                end_time=end_time,
                solver_dir=self.solver_dir,
                write_interval=write_interval,
                del_dirs=del_dirs
            )

        return elapsed_time


if __name__ == "__main__":
    # Example Usage
    # NOTE: Ensure `openfoam_config.num_processors` is set appropriately in your dataclass
    # openfoam_config = OpenfoamConfig(num_processors=4) 
    
    # openfoam_utils = OpenfoamUtils(
    #     openfoam_config,
    #     solver_dir=Path("/home/shilaj/shilaj_data/repitframework/repitframework/Solvers/natural_convection_case1"),
    #     assets_dir=Path("/home/shilaj/shilaj_data/repitframework/repitframework/Assets")
    # )
    
    # start_time = timeit.default_timer()
    # elapsed_cfd_time = openfoam_utils.run_solver(
    #     start_time=10.0, 
    #     end_time=110.0, 
    #     write_interval=0.01,
    #     save_to_numpy=True, 
    #     del_dirs=False
    # )
    # print(f"Simulation completed in: {elapsed_cfd_time:.2f} seconds")
    pass