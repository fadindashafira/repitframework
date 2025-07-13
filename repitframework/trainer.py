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

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset

torch.set_default_dtype(torch.float64)
torch.manual_seed(1004)
torch.cuda.manual_seed_all(1004)
np.random.seed(1004)

from repitframework.Dataset import BaseDataset, FVMNDataset
from repitframework.DataLoader import train_val_split
from repitframework.Models import FVMNetwork
from repitframework.config import TrainingConfig, NaturalConvectionConfig
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
		self.optimizer = self.optimizer_selection()
		self.scheduler = self.scheduler_selection(self.optimizer)
		if saved_model_name:
			self.model, self.optimizer, self.scheduler = self._from_checkpoint(
				saved_model_name,
				load_optimizer=load_optimizer,
				load_scheduler=load_scheduler)
		
		self.best_validation_loss = float('inf')

		self.variables = self.training_config.extend_variables()


	def _from_checkpoint(self, 
					  saved_model_name:str,
					  load_optimizer:bool=False,
					  load_scheduler:bool=False) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
		
		if load_optimizer:
			optim = self.optimizer_selection()
			if load_scheduler:
				scheduler = self.scheduler_selection(optim)
			else:
				scheduler = None
			return load_from_state_dict(
				self.model,
				self.training_config.model_dump_dir,
				saved_model_name,
				optim,
				scheduler
			)
		return load_from_state_dict(
			self.model,
			self.training_config.model_dump_dir,
			saved_model_name
		)

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
			label = labels[:, var_index:var_index + 1]  # Assuming labels are structured similarly
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
		
		for epoch in tqdm(range(epochs), desc="Training Epochs"):
			training_loss = self.train(train_loader)
			validation_loss = self.validate(val_loader)

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
			  model:torch.nn.Module,
			  prediction_input:np.ndarray):
		self.training_config = training_config
		self.model = model.to(self.training_config.device)
		self.prediction_input = prediction_input
		self.variables = self.training_config.extend_variables()

	def calculate_residual_mass(self, predictions:torch.Tensor, labels:torch.Tensor) -> float:
		return residual_mass(predictions, labels, self.training_config)
	
	def residual_mass_calculation(self, res_fn):
		'''
		Args:
			res_fn (callable): A function that takes ux and uy as inputs and returns the residual mass.
		'''
		return res_fn
	
	def _get_normalization_metrics(self, dataset_dir:Union[str, Path]) -> Dict[str:np.ndarray]:
		"While creatting the dataset instance, the normalization metrics are saved in a JSON file (if do_normalize is TRUE)."
		metrics_path = Path(dataset_dir) / "norm_denorm_metrics.json"
		with open(metrics_path, "r") as f:
			metrics = json.load(f)
		
		return metrics
	
	def apply_boundary_conditions(self, inputs:torch.Tensor) -> List[np.ndarray]:

		ux_index = self.variables.index("U_x")
		uy_index = self.variables.index("U_y")
		t_index = self.variables.index("T")


		ux_matrix = inputs[ux_index]
		uy_matrix = inputs[uy_index]
		t_matrix = inputs[t_index]

		inputs = FVMNDataset.hard_contraint_bc(
			data_list=inputs,
			extended_vars_list=self.variables,
			left_wall_temperature=self.training_config.left_wall_temperature,
			right_wall_temperature=self.training_config.right_wall_temperature,
		) 
		return inputs
	
	def prepare_prediction_inputs(self, prediction_input:np.ndarray,
							   dataset_dir:Union[str, Path]) -> torch.Tensor:
		
		pass
	def predict(self, prediction_input:np.ndarray,
			 prediction_start_time:float, 
			write_interval:float,
			dataset_dir:Union[str, Path])-> float:
		self.model.eval()

		# Initialize prediction start parameters:
		running_time = prediction_start_time
		self.relative_residual_mass = self.training_config.residual_threshold # This will be updated during the prediction loop.

		metrics = self._get_normalization_metrics(dataset_dir)
		self.true_residual_mass = metrics["true_residual_mass"]

		# Check if relative residual mass is within the threshold:
		condition1 = (self.relative_residual_mass <= self.training_config.residual_threshold)
		# Check if the prediction time is within the valid range
		condition2 = (running_time <= self.training_config.prediction_end_time)
	
		with torch.inference_mode():
			while condition1 and condition2:
				pass
	
	def predict_batched(self, model, inputs, batch_size=1024, device="cuda"):
		model.eval()
		all_outputs = []
		with torch.no_grad():
			for i in range(0, inputs.shape[0], batch_size):
				batch = inputs[i:i+batch_size].to(device)
				outputs = model(batch)
				all_outputs.append(outputs.cpu())
		return torch.cat(all_outputs)
			

if __name__ == "__main__":
	from repitframework.config import TrainingConfig, NaturalConvectionConfig
	from repitframework.DataLoader import train_val_split

	training_config = NaturalConvectionConfig()
	trainer = BaseHybridTrainer(training_config)
	print("BaseHybridTrainer initialized successfully.")

	dataset = FVMNDataset(
		start_time=10.0, 
		end_time=10.03,
		time_step=0.01,
		dataset_dir = str(training_config.assets_dir)+"_backup",
		first_training=True,
		vars_list=training_config.get_variables(),
		extended_vars_list=training_config.extend_variables(),
		output_dims="BD",
		do_normalize=training_config.do_normalize,
		do_feature_selection=training_config.do_feature_selection,
		left_wall_temperature=training_config.left_wall_temperature,
		right_wall_temperature=training_config.right_wall_temperature,
	)
	print(dataset)
	train_loader, val_loader = train_val_split(
		dataset,
		batch_size=training_config.batch_size)
	
	trainer.fit(train_loader, val_loader, epochs=training_config.epochs)

	

		   
			
	