from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging


import torch.cuda as cuda
import torch

@dataclass
class BaseConfig:
	# Directories:
	root_dir:Path = Path(__file__).parent.resolve()
	dataloader_dir:Path = Path(root_dir, "DataLoader")
	logs_dir:Path = Path(root_dir, "logs")
	metrics_dir:Path = Path(root_dir, "Metrics")
	modelselector_dir:Path = Path(root_dir, "Models")
	openfoam_dir:Path = Path(root_dir, "OpenFOAM")
	solver_dir:Path = Path(root_dir, "Solvers","natural_convection")

	assets_dir:Path = Path(root_dir, "Assets")
	assets_path:Path = Path.joinpath(assets_dir, solver_dir.name)
	assets_path.mkdir(parents=True, exist_ok=True)

	model_dir:Path = Path(root_dir, f"ModelDump/{solver_dir.name}")
	model_dir.mkdir(parents=True, exist_ok=True)

	# Logging Level
	logger_level = logging.DEBUG

	# Data parameters: Remember, if you calculating the residual, first vector should be U and last scalar should be T. 
	data_vars:dict[str] = field(default_factory=lambda: {"vectors":["U"],"scalars":["T"]})
	data_dim:int = 2 # It is to denote either data is 1D or 2D or 3D.
	grid_x: int = 200 # Number of grid points in x-direction
	grid_y: int = 200 # Number of grid points in y-direction
	grid_z: int = 1 # Number of grid points in z-direction
	grid_step: int|float = 0.005 # Grid step size
	write_interval: int|float = 0.01 # Time step size
	round_to: int = len(str(write_interval).split(".")[-1]) # Number of decimal places to round to

	def get_variables(self, vars_dict:dict=None):
		"""
		Because we have a dictionary that separates variables into scalars and vectors,
		we need to combine them into a single list. We need to further differentiate vectors 
		based on the number of dimensions (1D or 2D or 3D).

		Args:
			vars_dict (dict): The dictionary containing the variables separated into vectors and scalars.

		Returns:
			list: The list of variables: ["U_x", "U_y", "T"]
		"""
		self.data_vars = self.data_vars if vars_dict is None else vars_dict
		vars_list = []
		for key, value in self.data_vars.items():
			if key == "vectors":
				for var in value:
					if self.data_dim == 1:
						vars_list.append(f"{var}_x")
					elif self.data_dim == 2:
						vars_list.extend([f"{var}_x", f"{var}_y"])
					elif self.data_dim == 3:
						vars_list.extend([f"{var}_x", f"{var}_y", f"{var}_z"])
			elif key == "scalars":
				vars_list.extend(value)
			else: 
				raise ValueError(f"Invalid key {key} in vars_dict. Must be either 'vectors' or 'scalars'.")
		return vars_list
	
	def extend_variables(self, vars_dict:dict=None):
		'''
		There are cases where we don't exactly need dimension separated variables, just the variables themselves just like 
		how they are represented in OpenFOAM. Hence, this method is to return the variables as they are, as a list.

		Args:
			vars_dict (dict): The dictionary containing the variables separated into vectors and scalars.
		
		Returns:
			list: The list of variables: ["U", "T"]
		'''
		vars_dict = self.data_vars if vars_dict is None else vars_dict
		return [var for _, value in vars_dict.items() for var in value]
	
	def setup_logger(self, name:str,log_file: Path) -> logging.Logger:
		"""
		Sets up and returns a logger instance.

		Args:
			log_file (Path): The file path where logs will be saved.
			level (int): Logging level (e.g., logging.DEBUG).

		Returns:
			logging.Logger: Configured logger.
		"""
		level = self.logger_level
		logger = logging.getLogger(name)

		today_date = datetime.now().strftime("%Y-%m-%d")
		today_date_dir = Path(self.logs_dir, today_date)
		today_date_dir.mkdir(parents=True, exist_ok=True)
		log_file = Path(today_date_dir, log_file)
		# Prevent adding multiple handlers if the logger is already configured
		if not logger.handlers:
			logger.setLevel(level)
			formatter = logging.Formatter('%(asctime)s:%(pathname)s:%(levelname)s:%(message)s', datefmt='%H:%M:%S')

			file_handler = logging.FileHandler(log_file)
			file_handler.setFormatter(formatter)

			logger.addHandler(file_handler)

		return logger

	def __post_init__(self):
		log_file: Path = Path("BaseConfig.log")
		self.logger = self.setup_logger("BaseLogger",log_file)

class OpenfoamConfig(BaseConfig):
	def __init__(self):
		super(OpenfoamConfig, self).__init__()
		self.case_name:str = None if self.solver_dir is None else self.solver_dir.name
		self.log_file = Path("OpenFOAM.log")
		self.mesh_type:str = "blockMesh"
		self.solver_type:str = "buoyantFoam"
		self.logger = self.setup_logger("OpenFOAMLogger",self.log_file)
		self.start_time = 0
		self.end_time = 2
		self.start_time = round(self.start_time, self.round_to)
		self.end_time = round(self.end_time, self.round_to)

class TrainingConfig(BaseConfig):
	def __init__(self):
		super().__init__()
		self.batch_size: int = 10000
		self.epochs: int = 1
		self.learning_rate: float = 0.001
		self.residual_threshold: float = 0.001
		self.device: str = "cuda" if cuda.is_available() else "cpu"
		self.optimizer = torch.optim.Adam
		self.loss = torch.nn.MSELoss()
		self.activation = torch.nn.ReLU

		self.training_start_time = 10.0
		self.training_end_time = 10.03
		self.prediction_start_time = 10.03
		self.prediction_end_time = 20.0

		self.log_file: Path = Path("Training.log")
		self.logger = self.setup_logger("TrainingLogger",self.log_file)

if __name__ == "__main__":
	openfoam_config = OpenfoamConfig()
	base_config = BaseConfig()

	# Testing logger:
	base_config.logger.info("Testing BaseConfig logger")
	openfoam_config.logger.info("Testing OpenFOAM logger")

	print(openfoam_config.data_vars, openfoam_config.extend_variables(),openfoam_config.assets_path)