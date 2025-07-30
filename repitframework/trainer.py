"""RePIT Framework: Hybrid CFD-ML Simulation Framework with Transfer Learning:
1. Model with best validation loss is saved as "best_model_{model_type}.pth".
2. metrics have special naming convention (in norm_denorm_metrics.json):
   - "label_mean", "label_std", "input_mean", "input_std", "true_residual_mass"
3. To load optimizer and scheduler from checkpoint, override "__init__" method and set `load_optimizer=True` and `load_scheduler=True`.
4. To change the loss logic, architectural nuances in loss calculation, override `process_one_batch` method.
5. A method is provided to give the residual mass calculation function, which can be used in the `predict` method.
"""

from __future__ import annotations
from pathlib import Path
import json
from typing import Tuple, List,Union, Dict
import timeit

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset

torch.set_default_dtype(torch.float64)
torch.manual_seed(1004)
torch.cuda.manual_seed_all(1004)
np.random.seed(1004)

from repitframework.Dataset import (BaseDataset, 
									FVMNDataset, 
									parse_numpy, 
									hard_constraint_bc, 
									add_feature, 
									match_input_dim, 
									normalize,
									denormalize)
from repitframework.DataLoader import train_val_split
from repitframework.Models import FVMNetwork
from repitframework.config import TrainingConfig, NaturalConvectionConfig, OpenfoamConfig
from repitframework.OpenFOAM import OpenfoamUtils, numpyToFoam
from repitframework.plot_utils import save_loss

from repitframework.model_selector import ModelSelector, OptimizerSelector, SchedulerSelector
from repitframework.utils import freeze_layers, load_from_state_dict, save_to_state_dict, optimize_required_grads_only
from repitframework.Metrics import residual_mass


class BaseHybridTrainer:
	def __init__(self, training_config:NaturalConvectionConfig,
				 saved_model_name:str=None,
				 load_optimizer:bool=False,
				 load_scheduler:bool=False):
		self.training_config = training_config
		self.device = training_config.device
		self.model = ModelSelector(
			training_config.model_type,
			training_config.model_kwargs
		)
		self.model.to(self.device)
		if saved_model_name:
			self.model, self.optimizer, self.scheduler = self._from_checkpoint(
				saved_model_name,
				load_optimizer=load_optimizer,
				load_scheduler=load_scheduler)
			self.training_config.epochs = 0  # Set epochs to 0 if loading from checkpoint
		else:
			self.optimizer = self.optimizer_selection()
			self.scheduler = self.scheduler_selection(self.optimizer)
		
		self.best_validation_loss = float('inf')
		self.variables = self.training_config.extend_variables()                                      


	def _from_checkpoint(self, 
					  saved_model_name:str,
					  load_optimizer:bool=False,
					  load_scheduler:bool=False) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
		
		optim = self.optimizer_selection()
		scheduler = None
		if load_optimizer and load_scheduler:
			scheduler = self.scheduler_selection(optim)
			return load_from_state_dict(
				self.model,
				self.training_config.model_dump_dir,
				saved_model_name,
				self.optimizer,
				self.scheduler
			)
		elif load_optimizer:
			model, optim, *_ = load_from_state_dict(
				self.model,
				self.training_config.model_dump_dir,
				saved_model_name,
				self.optimizer,
				scheduler=scheduler
			)
			return model, optim, scheduler

		model, *_ = load_from_state_dict(
			self.model,
			self.training_config.model_dump_dir,
			saved_model_name
		)
		
		return model, optim, scheduler
	
	def optimizer_selection(self,) -> torch.optim.Optimizer:
		"""
		Get the optimizer used for training.
		Returns:
			torch.optim.Optimizer: The optimizer instance.
		"""
		return OptimizerSelector(
			self.training_config.optimizer_type,
			self.model.parameters(),
			self.training_config.optim_kwargs
		)
	
	def scheduler_selection(self, optim_instance:torch.optim.Optimizer) -> Union[torch.optim.lr_scheduler._LRScheduler, None]:
		"""
		Get the learning rate scheduler used for training.
		Returns:
			torch.optim.lr_scheduler._LRScheduler: The scheduler instance.
		"""
		if self.training_config.scheduler_type is None:
			return None
		
		return SchedulerSelector(
			scheduler_name=self.training_config.scheduler_type,
			optimizer=optim_instance
		)
	
	def _process_module_dict(self, predictions:torch.nn.ModuleDict, labels:torch.Tensor) -> float:
		
		loss = 0.0
		for var, pred in predictions.items():
			var_index = self.variables.index(var)
			label = labels[:, var_index:var_index + 1]  # Assuming labels are always present on dimension 1.
			loss += self.training_config.loss(pred, label)
		return loss

	def process_one_batch(self, inputs:torch.Tensor, labels:torch.Tensor) -> float:
		"""
		Train the model for one batch of data.
		Args:
			inputs (torch.Tensor): Input data.
			labels (torch.Tensor): Ground truth labels.
		Returns:
			Tuple[float, float]: Average loss and average residual for the batch.
		"""
		inputs, labels = inputs.to(self.device), labels.to(self.device)

		predictions = self.model(inputs)

		# Multiple models can be used, so we need to handle ModuleDict.
		if hasattr(self.model, "networks") and isinstance(self.model.networks, torch.nn.ModuleDict):
			loss = self._process_module_dict(predictions, labels)
		else:
			loss = self.training_config.loss(predictions, labels)

		if self.model.training: # returns True if model is in training mode.
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		return loss.item()

		
	def process_one_epoch(self, dataloader:DataLoader) -> float:
		loss = 0.0
		num_batches = len(dataloader)

		for batch in dataloader:
			inputs, labels = batch
			loss += self.process_one_batch(inputs, labels)
		loss /= num_batches

		return loss

	def fit(self, 
		   train_loader:DataLoader, 
		   val_loader:DataLoader,
		   epochs:int=None,
		   freeze_layers:bool=True,):
		
		if freeze_layers:
			self.model, self.optimizer = optimize_required_grads_only(
				self.model,
				self.training_config
			)
		if epochs is None:
			epochs = self.training_config.epochs
		
		print("Training started...")
		for epoch in range(epochs):
			training_loss = self.train(train_loader)
			validation_loss = self.validate(val_loader)

			print(f"\rEpoch: {epoch}/{epochs}| Training Loss: {training_loss} | Validation Loss: {validation_loss}", end='', flush=True)
			# At the end of the epoch, after all batches are processed:
			if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
				self.scheduler.step(validation_loss) # Requires a metric
			elif self.scheduler:
				self.scheduler.step() # Other schedulers just step without a metric
			self.training_config.log_metrics(key="Epoch", value=epoch+1, metrics_type="training")
			self.training_config.log_metrics(key="Training Loss", value=training_loss, metrics_type="training")
			self.training_config.log_metrics(key="Validation Loss", value=validation_loss, metrics_type="training")
			

	def train(self, train_loader:DataLoader):
		self.model.train()
		train_loss = self.process_one_epoch(dataloader=train_loader)
		self.training_config.logger.debug(f"Training Loss: {train_loss:.4f}")
		return train_loss
	

	def validate(self, val_loader:DataLoader):
		self.model.eval()
		with torch.no_grad():
			val_loss = self.process_one_epoch(dataloader=val_loader)
		self.training_config.logger.debug(f"Validation Loss: {val_loss:.4f}")

		if val_loss < self.best_validation_loss:
			self.best_validation_loss = val_loss
			model_name = f"best_model_{self.training_config.model_type}.pth"
			save_to_state_dict(
				self.model,
				self.training_config.model_dump_dir,
				model_name,
				self.optimizer,
				self.scheduler
			)
			self.training_config.logger.info(f"Model saved with validation loss: {val_loss:.4f}")

		return val_loss
	

class BaseHybridPredictor:
	def __init__(self, training_config:NaturalConvectionConfig,
			  model:torch.nn.Module):
		self.training_config = training_config
		self.model = model.to(self.training_config.device)
		self.variables = self.training_config.extend_variables()
		self.ux_index = self.variables.index("U_x")
		self.uy_index = self.variables.index("U_y")

		assert self.ux_index, "U_x and U_y must be in the variables list. Otherwise, residue calculation will not work.Hence, no swithching point."
	
	def _get_normalization_metrics(self, dataset_dir:Union[str, Path]) -> Dict[str:np.ndarray]:
		"While creatting the dataset instance, the normalization metrics are saved in a JSON file (if do_normalize is TRUE)."
		metrics_path = Path(dataset_dir) / "norm_denorm_metrics.json"
		with open(metrics_path, "r") as f:
			metrics = json.load(f)
		
		return metrics
	
	def get_ground_truth_data(
			self, 
			time_step:int|float
		) -> List[np.ndarray]:
		'''
		For the first prediction, we need ground truth data to give input to the model.
		Args
		---- 
		time_step: float: 
			The time step for which we are predicting. e.g., 5.03

		Returns
		-------
		>>> Shape: [num_variables, grid_y, grid_x]

		Functionality
		-------------
		1. Get data for the time step from ground truth data.
		2. Parse the numpy data for the variables.
		3. Separate the dimensions of the data if present.
		'''
		data_path = self.training_config.assets_dir
		variables = self.training_config.get_variables()
		full_data_path = [data_path / f"{var}_{time_step}.npy" for var in variables]
		numpy_data = [
			parse_numpy(
				dataset_file=path,
				grid_x=self.training_config.grid_x,
				grid_y=self.training_config.grid_y,
				grid_z=self.training_config.grid_z,
				data_dim=self.training_config.data_dim
			)
			for path in full_data_path
		]
		temp = list()
		for data in numpy_data:
			if len(data.shape) > 2:
				for i in range(self.training_config.data_dim):
					temp.append(data[:,:, i])
			else:
				temp.append(data)
		return np.stack(temp, axis=0)  # Shape: [num_variables, grid_y, grid_x]
	
	def save_prediction_results(self,
								pred_data:np.ndarray,
								time_step:Union[int,float]) -> np.ndarray:
		
		'''
		Save the predicted data to the assets directory.
		Args
		----
		pred_data: np.ndarray:
			The predicted data from the model.
		time_step: Union[int, float]:
			The time step for which the data is predicted.

		Returns
		-------
		The predicted data stacking the variables in the order of self.variables in the dimension 0.
		>>> [num_variables, grid_y, grid_x]
		'''
		# Regardless of the shape, throughout the framework, we always put features in dimension 1.
		pred_data = [pred_data[:, i].reshape(self.training_config.grid_y, self.training_config.grid_x) for i in range(pred_data.shape[1])]

		for notation, var in self.training_config.data_vars.items():
			if notation == "vectors":
				for i, vector in enumerate(var):
					var_Xindex = self.variables.index(vector + "_x")
					var_Yindex = self.variables.index(vector + "_y")
					Xdata:np.ndarray = pred_data[var_Xindex]
					Ydata:np.ndarray = pred_data[var_Yindex]
					if vector == "U":
						self.ux_matrix = Xdata
						self.uy_matrix = Ydata
					vector_data = np.concatenate([Xdata.reshape(-1,1), Ydata.reshape(-1,1)], axis=1)
					np.save(self.training_config.assets_dir / f"{vector}_{time_step}_predicted.npy", vector_data)
			elif notation == "scalars":
				for i, v in enumerate(var):
					var_index = self.variables.index(v)
					np.save(self.training_config.assets_dir / f"{v}_{time_step}_predicted.npy", pred_data[var_index].reshape(-1))
			else:
				raise ValueError(f"Invalid notation {notation} in data_vars. Must be either 'vectors' or 'scalars'.")

		return np.stack(pred_data, axis=0)  # Shape: [num_variables, grid_y, grid_x]
	

	def apply_boundary_conditions(self, pred_data:np.ndarray) -> np.ndarray:
		'''
		Apply the boundary conditions to the data.

		Args
		----
		data: torch.Tensor:
			The data for which the boundary conditions are to be applied.
		time_step: int|float:
			The time step for which the boundary conditions are to be applied.
		data_path: Path:
			The path where the data is stored.
		'''
		if self.training_config.do_feature_selection:
			pred_data = np.concatenate([add_feature(data) for data in hard_constraint_bc(
				pred_data,
				extended_vars_list=self.variables,
				left_wall_temperature=self.training_config.left_wall_temperature,
				right_wall_temperature=self.training_config.right_wall_temperature
				)], axis=0)
		temp = match_input_dim(
			output_dims=self.training_config.output_dims,
			inputs= [pred_data]
		)
		return temp
	
	def prepare_input_for_prediction(self, time_step:int|float,  
									 prediction_input:np.ndarray=None) -> np.ndarray:
		'''
		If feature selection is enabled, boundary values need to be enforced.

		Args
		----
		time_step: int|float:
			If we are predicting for t then time_step = t-dt.
		data: torch.Tensor: 
			The output from the model after denormalizing and adding with the input [batch_size, num_features]
		data_path: Path: 
			if we predict for time step 5.03 then we need the original data for the time step 5.03 to get the boundary data.

		Functionality
		-------------
		1. Because, we are using the same training_config.get_variables() to get the variables.
		   We leverage this to get the index of U_x, U_y, T.
		2. If it is not the first prediction, we are setting U_x and T values in that iteration as previous values 
		   and as the process progresses, we update the previous values with the predicted values.
		3. We save the predicted values here. In the prediction loop, we get the output for time(running_time) + dt.
		   So, it makes sense that we can update the running time, and while preparing input for the next prediction, 
		   we can add boundary values to the prev. predicted values and that would represent the predicted values for 
		   currently running_time in prediction loop.

		Reasoning
		---------
		But why did we assign the present/previous ux_matrix, uy_matrix, t_matrix in this function? 
		Because, we would have input and output data both in the predict method. Wouldn't it make sense to assign the values there?

		Sadly NO.
		Because, the input for the network is feature extracted. Example shape: [40000,15]
		And the output from the network is boundary excluded data. Example shape: [39204,3]
		Hence, we must do the post-processing before calculating the residue. So, for me, 
		it made a lot of sense to assign the values here. If you have a better idea, please let me know.
		'''
		if prediction_input is None:
			# If it is the first prediction, we need to get the ground truth data.
			ground_truth = self.get_ground_truth_data(time_step)
			self.relative_residual_mass = residual_mass(ground_truth[self.ux_index], ground_truth[self.uy_index])/self.true_residual_mass

			return self.apply_boundary_conditions(ground_truth)
		
		pred_data = self.save_prediction_results(prediction_input, time_step)

		# Save residual mass for the time step
		predicted_residual_mass = residual_mass(self.ux_matrix, self.uy_matrix, order="C")
		self.relative_residual_mass = predicted_residual_mass / self.true_residual_mass

		self.training_config.logger.debug(f"Relative Residual Mass: {self.relative_residual_mass:.4f}")
		self.training_config.log_metrics(key="Running Time", value=time_step, metrics_type="prediction")
		self.training_config.log_metrics(key="Relative Residual Mass", value=self.relative_residual_mass, metrics_type="prediction")

		return self.apply_boundary_conditions(pred_data)
	
	def prediction_loop(self, running_time:float,
					 prediction_input:np.ndarray,
					 metrics:dict) -> np.ndarray:
		
		prediction_input:np.ndarray = self.prepare_input_for_prediction(
					time_step=running_time,
					prediction_input=prediction_input
				)
		
		if self.training_config.do_normalize:
			network_input, *_ = normalize(prediction_input,mean=np.array(metrics["input_mean"]),std=np.array(metrics["input_std"]))
		else:
			network_input = prediction_input
		# Convert to torch tensor and move to device
		network_input:torch.Tensor = torch.from_numpy(network_input).double().to(self.training_config.device)
		predicted_output:torch.Tensor = self.model(network_input)

		# Denormalize the output
		if isinstance(predicted_output, Dict):
			predicted_output = torch.cat([output.cpu() for output in predicted_output.values()], dim=1)

		if self.training_config.do_normalize:
			network_output = denormalize(
				predicted_output.numpy(),
				np.array(metrics["label_mean"]),
				np.array(metrics["label_std"])
			)
		else:
			network_output = predicted_output.numpy()

		if self.training_config.do_feature_selection:
			# Always on dimension 1, we have the features.
			auto_regressed_input = prediction_input[:,::5] + network_output
		else:
			auto_regressed_input = prediction_input + network_output

		return auto_regressed_input

	def predict(self,prediction_start_time:float)-> float:
		
		self.model.eval()
		# Initialize prediction start parameters:
		running_time = prediction_start_time
		prediction_input = None
		self.relative_residual_mass = self.training_config.residual_threshold # This will be updated during the prediction loop.

		metrics = self._get_normalization_metrics(self.training_config.assets_dir)
		self.true_residual_mass = metrics["true_residual_mass"]

		with torch.inference_mode():
			while (self.relative_residual_mass <= self.training_config.residual_threshold) and (running_time <= self.training_config.prediction_end_time):
				prediction_input = self.prediction_loop(
					running_time=running_time,
					prediction_input=prediction_input,
					metrics=metrics
				)
				# Update the running time
				running_time = round(running_time + self.training_config.write_interval, self.training_config.round_to)

			# Because prepare_input_for_prediction function calculates the residual values.
			# Hence, even if the residue value exceeds the threshold, the running time will be updated.
			# So, we need to step down the running time by the write interval outside the loop.
		return round(running_time - self.training_config.write_interval, self.training_config.round_to)
	
	def predict_batched(self, model: torch.nn.Module, inputs):
		model.eval()
		# If model returns a dict, create per-key storage
		all_outputs = None
		with torch.no_grad():
			for i in range(0, inputs.shape[0], self.training_config.batch_size):
				batch = inputs[i:i+self.training_config.batch_size].to(self.training_config.device)
				outputs = model(batch)
				if isinstance(outputs, dict):
					if all_outputs is None:
						all_outputs = {k: [] for k in outputs}
					for k, v in outputs.items():
						all_outputs[k].append(v.cpu())
				else:
					if all_outputs is None:
						all_outputs = []
					all_outputs.append(outputs.cpu())
		# Concatenate appropriately
		if isinstance(all_outputs, dict):
			return {k: torch.cat(v, dim=0) for k, v in all_outputs.items()}
		else:
			return torch.cat(all_outputs)



def hybrid_train_predict(training_config:NaturalConvectionConfig,
						 openfoam_config:OpenfoamConfig,
						 saved_model_name:str=None,
						 transfer_learning_epochs:int=2,
						 relative_residual_threshold:int=5) -> None:
	"""
	Function to train and predict using the BaseHybridTrainer and BaseHybridPredictor classes.
	"""
	training_start_time = training_config.training_start_time
	training_end_time = training_config.training_end_time
	running_time = training_start_time
	trainer = BaseHybridTrainer(training_config, saved_model_name=saved_model_name)
	# trainer = BaseHybridTrainer(training_config=training_config)
	print("BaseHybridTrainer initialized successfully.")

	openfoam_utils = OpenfoamUtils(openfoam_config)
	framework_start_time = timeit.default_timer()
	first_training = True
	# Storing times 
	cfd_times = 0.0
	ml_times = 0.0
	update_times = 0.0

	switch_count = 0
	ml_timesteps = 0
	cfd_timesteps = 0
	training_config.logger.info(f"Framework started at {timeit.default_timer()}")
	while running_time < training_config.prediction_end_time:
		# Run CFD first:
		cfd_start_time = timeit.default_timer()
		openfoam_utils.run_solver(
			start_time=running_time, 
			end_time=training_end_time,
			save_to_numpy=True
		)
		cfd_end_time = timeit.default_timer()
		if training_end_time >= trainer.training_config.prediction_end_time: break
		# Create dataset instance
		dataset = FVMNDataset(
			start_time=training_start_time,
			end_time=training_end_time,
			time_step=0.01,
			dataset_dir=training_config.assets_dir,
			first_training=first_training,
			vars_list=training_config.get_variables(),
			extended_vars_list=training_config.extend_variables(),
			do_normalize=training_config.do_normalize,
			left_wall_temperature=training_config.left_wall_temperature,
			right_wall_temperature=training_config.right_wall_temperature,
			do_feature_selection= training_config.do_feature_selection
		)
		train_loader, val_loader = train_val_split(
			dataset, 
			batch_size=trainer.training_config.batch_size,
			train_size=2/3
		)

		# Train the model
		update_start_time = timeit.default_timer()
		trainer.fit(
			train_loader, 
			val_loader,
			freeze_layers= not first_training
		)
		update_end_time = timeit.default_timer()

		trainer.best_validation_loss = float("inf") # Reset the best validation accuracy for transfer learning
		# Before prediction, load the best model: because we are using the same instance of self.model for prediction, hence last trained parameters will be used.
		# trainer.model, trainer.optimizer = load_from_state_dict(
		# 	model=trainer.model,
		# 	model_save_path=trainer.training_config.model_dir,
		# 	model_name="best_model.pth",
		# 	optimizer=trainer.optimizer
		# )
		trainer.model, *_ = trainer._from_checkpoint(f"best_model_{training_config.model_type}.pth")

		if trainer.training_config.epochs == 5000: 
			save_to_state_dict(
				trainer.model,
				trainer.training_config.model_dump_dir,
				f"init_model_{trainer.training_config.model_type}.pth",
				trainer.optimizer,
				trainer.scheduler
			)
			save_loss(
				training_config=training_config,
				save_initial_losses=True
			)
		
		print("\nStarting prediction from: ", 
			round(training_end_time+trainer.training_config.write_interval,2)
		)
		# Store times 
		if running_time > training_config.training_start_time:
			cfd_times += cfd_end_time - cfd_start_time
			update_times += update_end_time - update_start_time
	
		ml_start_time = timeit.default_timer()

		predictor = BaseHybridPredictor(
			training_config=training_config,
			model=trainer.model
		)
		running_time = predictor.predict(prediction_start_time=training_end_time)

		ml_end_time = timeit.default_timer()
		ml_times += ml_end_time - ml_start_time

		# ML timesteps per cross-computation
		ml_timesteps += round((running_time - training_end_time)/trainer.training_config.write_interval)
		print("ML timesteps: ", round((running_time - training_end_time)/trainer.training_config.write_interval))
		print("Switch count: ", switch_count)
		switch_count += 1
		print(f"Prediction ended at:{running_time}\n")

		# Convert predicted numpy to foam
		numpyToFoam_string = numpyToFoam(
			openfoam_config=openfoam_config, 
			latestML_time=float(running_time), 
			latestCFD_time=training_end_time,
		)

		openfoam_config.logger.info(f"Converted numpy to foam: {numpyToFoam_string}")

		# Transfer learning
		# trainer.training_config.epochs, cfd_runs = dynamic_parameters(switch_count)
		trainer.training_config.epochs = transfer_learning_epochs
		cfd_runs = 10
		# if switch_count == 2: break
		cfd_timesteps += cfd_runs
		training_end_time = round(running_time + cfd_runs*trainer.training_config.write_interval,
								  2)
		# Just using last three time steps for transfer learning: 
		training_start_time = round(training_end_time - 3*trainer.training_config.write_interval, 2)
		first_training = False


	framework_end_time = timeit.default_timer()
	training_config.logger.info(f"\n\nFramework ended at {framework_end_time}")
	training_config.logger.info(f"Transfer learning epochs: {trainer.training_config.epochs}")
	training_config.logger.info(f"Relative Residual Mass: {trainer.training_config.residual_threshold}\n")
	training_config.logger.info(f"Total CFD Time: {cfd_times}")
	training_config.logger.info(f"Total ML Time: {ml_times}")
	training_config.logger.info(f"Total Update Time: {update_times}")
	training_config.logger.info(f"Total Framework Time: {framework_end_time-framework_start_time}")
	training_config.logger.info(f"Total ML timesteps: {ml_timesteps}")
	training_config.logger.info(f"Total CFD runs: {cfd_timesteps}")

	if_CFD_alone = (ml_timesteps + cfd_timesteps)*(cfd_times/cfd_timesteps)
	training_config.logger.info("###############################################")
	training_config.logger.info(f"CFD alone time: {if_CFD_alone}")
	training_config.logger.info(f"CFD+ML+update times: {cfd_times+ml_times+update_times}")
	training_config.logger.info(f"Acceleration: {if_CFD_alone/(ml_times + cfd_times + update_times)}")
	training_config.logger.info(f"ML timesteps per cross-computation: {ml_timesteps/switch_count}")
	training_config.logger.info(f"t_ML: {ml_times/ml_timesteps}")
	training_config.logger.info(f"t_CFD: {cfd_times/cfd_timesteps}")
	training_config.logger.info(f"Real Acceleration: {if_CFD_alone/(framework_end_time-framework_start_time)}\n\n")
	training_config.logger.info("###############################################")
	# save_loss(training_config=training_config, merge_initial_losses=False)

if __name__ == "__main__":
	
	training_config = NaturalConvectionConfig()
	openfoam_config = OpenfoamConfig()
	training_config.logger.info("Starting the hybrid training and prediction process...")
	hybrid_train_predict(training_config, 
					  openfoam_config,
					  saved_model_name=f"best_model_{training_config.model_type}.pth",
					  transfer_learning_epochs=2)
	training_config.logger.info("Hybrid training and prediction process completed.")
	

		   
			
	