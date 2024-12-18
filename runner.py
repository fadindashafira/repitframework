__doc__ ='''
Steps to re-create the RePIT project:

Step 1: Convert 10th timestamp data from numpy to foam.
Step 2: Generate the whole time steps from 10 to 20 using 0.01 timestamp as write interval (record the time it takes to run the simulation).
Step 3: Use 10.0 to 10.03 timestamp data as training data. 
Step 4: After the training is done using hyperparameters as same as in RePIT: lr= 0.001, epoch=5000, batch_size=10000, optimizer=Adam, loss=MSE.
Step 5: Use the trained model to predict from the timestamp 10.04 until residue exceeds 0.001 (Average residual mass).
Step 6: Use two time steps from the predicted timestamps to enable transfer learning. 
Step 7: And REPEAT the process until the timestamp 20.0 is reached.
'''

from typing import Tuple, List
from pathlib import Path
import timeit
from copy import deepcopy

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from repitframework.Dataset.fvmn import FVMNDataset
from repitframework.Models.FVMN.fvmn import FVMNetwork
from repitframework.config import TrainingConfig, OpenfoamConfig, BaseConfig
from repitframework.OpenFOAM.utils import OpenfoamUtils
from repitframework.Metrics.ResidualNaturalConvection import residual_mass, residual_momentum, residual_heat
from repitframework.Metrics.MetricsLogger import MetricsLogger
from repitframework.plot_utils import make_animation
from repitframework.OpenFOAM.numpyToFoam import numpyToFoam


def get_dataloader(training_config:TrainingConfig, 
                   dataset:Dataset, 
                   batch_size=None)->Tuple[DataLoader, DataLoader]:
    '''
    Get the dataloaders for training and validation.
    '''
    batch_size = batch_size if batch_size else training_config.batch_size

    # Indices for splitting
    data_size = len(dataset)
    indices = list(range(data_size))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=1004)

    # Subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

class Trainer:
    def __init__(self, training_config:TrainingConfig, 
                 model:torch.nn.Module, 
                 optimizer:torch.optim.Adam, 
                 loss_fn:torch.nn.MSELoss, 
                 model_name:str=None):
        self.training_config = training_config
        self.device = training_config.device
        self.model = model.to(self.device)
        self.model = self.model if model_name is None else self.load_model(model_name)
        self.optimizer = optimizer(self.model.parameters(), lr=training_config.learning_rate)
        self.loss_fn = loss_fn
        self.losses = {"train": [], "val": []}
        self.best_val_accuracy = 1

        # setting variables for residue calculation 
        self.residual_mass = list()
        self.residual_momentum = list()
        self.residual_heat = list()
        self.residual_threshold = training_config.residual_threshold
        self.relative_residual_mass = float()

        self.ux_matrix = torch.zeros((training_config.grid_y, training_config.grid_x))
        self.uy_matrix = torch.zeros((training_config.grid_y, training_config.grid_x))
        self.t_matrix = torch.zeros((training_config.grid_y, training_config.grid_x))
        self.t_matrix_prev = torch.zeros((training_config.grid_y, training_config.grid_x))
        self.ux_matrix_prev = torch.zeros((training_config.grid_y, training_config.grid_x))

        self.variables = self.training_config.get_variables()
        self.ux_index = self.variables.index("U_x")
        self.uy_index = self.variables.index("U_y")
        self.t_index = self.variables.index("T")

    def train(self, train_loader:DataLoader, 
              val_loader:DataLoader, 
              epochs) -> bool:
        self.model.train()  # Set the model to training mode
        for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
            epoch_loss = list()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                predictions = self.model(x)
                loss = self.loss_fn(predictions, y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss.append(loss.item())
            
            epoch_loss = np.mean(epoch_loss)
            self.losses["train"].append(epoch_loss)
            training_config.logger.info(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

            # Validation loss
            val_loss = self.validate(val_loader)
            if val_loss < self.best_val_accuracy:
                self.best_val_accuracy = val_loss
                self.save_model(f"best_model.pth")
            self.losses["val"].append(val_loss)

        
        return True

    def validate(self, val_loader):
        self.model.eval()  # Set the model to evaluation mode
        val_loss = list()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                predictions = self.model(x)
                val_loss.append(self.loss_fn(predictions, y).item())
        
        val_loss = np.mean(val_loss)
        training_config.logger.info(f"Validation Loss: {val_loss:.4f}")
        return val_loss
    
    def predict(self, prediction_start_time:int|float=None, 
                write_interval:int|float=None, 
                data_path:Path=None):
        """
        Parameters
        ----------
        prediction_start_time: int|float:
            From which time step to start the prediction.
            Default is the prediction_start_time from the training config.
        write_interval: int|float:
            The interval at which the data is written.
            Default is the write_interval from the training config.
        data_path: Path:
            The path to the data directory.
        """
        start_time = prediction_start_time if prediction_start_time else self.training_config.prediction_start_time
        time_step = write_interval if write_interval else self.training_config.write_interval
        data_path = Path(data_path) if data_path else self.training_config.assets_path 

        self.model.eval()
        prediction_input = None
        running_time = start_time # Because we saving the prediction data at prepare_input_for_prediction function. But, output is after calling this function.
        with torch.no_grad():
            while (self.relative_residual_mass <= self.residual_threshold) and (running_time <= self.training_config.prediction_end_time):
                first_prediction = True if running_time == start_time else False
                prediction_input = self.prepare_input_for_prediction(running_time, data_path, first_prediction, prediction_input)
                normalized_input,mean, std  = FVMNDataset.normalize(prediction_input)
                predicted_output:torch.Tensor = self.model(normalized_input.to(self.device))
                denormed_output = FVMNDataset.denormalize(predicted_output.cpu(), mean, std)
                prediction_input = prediction_input[:, ::5] + denormed_output
                running_time = round(running_time+time_step, self.training_config.round_to)
                
            # Because prepare_input_for_prediction function calculates the residual values.
            # Hence, even if the residue value exceeds the threshold, the running time will be updated.
            # So, we need to step down the running time by the write interval outside the loop.
        return round(running_time-time_step, self.training_config.round_to)
    
    def transfer_learning(self, start_time:int|float, 
                          end_time:int|float, 
                          write_interval:int|float):
        '''
        The first part of the training in the RePIT framework is done for 5000 epochs.
        Which takes, 1.5 hrs to train even on GPU--A100. 
        So, the best model is saved and we load the model for transfer learning.
        '''


    def save_model(self, model_name:str) -> Path:
        '''
        Parameters
        ----------
        model_name: str: 
            The name of the model to be saved.
            Example: "model.pth"

        Returns
        -------
        model_save_path: Path: 
            The path where the model is saved.
            It will be saved in the repiframework/ModelDump/{case_name} directory.
        '''
        path = Path.joinpath(self.training_config.model_dir, model_name)
        torch.save(self.model.state_dict(), path)
        self.training_config.logger.info(f"Model saved as {model_name} at {self.training_config.model_dir}")
        return path

    def load_model(self, model_name:str):
        """
        This is for transfer learning. We load the model from the saved model.
        Epochs= 20
        """
        path = Path.joinpath(self.training_config.model_dir, model_name)
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.training_config.logger.info(f"Model loaded from {path}")
        return self.model
    
    def get_ground_truth_data(self, time_step:int|float, 
                              data_path:Path=None, 
                              first_prediction:bool=False) -> List[np.ndarray]:
        '''
        Because in FVMN, we are only predicting the interior points, we need to add the boundary data to the model output.
        Also, we need to calculate the residue. Hence, we need the true data for the time step.

        Args
        ---- 
        data_path: Path: 
            If we predict for time step 5.03 then we need the original data for the 
            time step 5.03 to get the boundary data.This is the path to that data.
        time_step: float: 
            The time step for which we are predicting. e.g., 5.03
        first_prediction: bool: # TODO: This is not used. Remove it.
            If this is the first prediction, we return the whole data. 
            Else, we set the all the other except boundary to zero.

        Returns
        -------
        Each numpy array is the data for each variable separated dimension wise:
        e.g., [U_x, U_y, T] for each variable.
        
        ground_truth_data: List[np.ndarray]:
            Along with boundary values, we send the true values also.

        Functionality
        -------------
        1. Get the boundary data for the time step from ground truth data.
        2. Parse the numpy data for the variables.
        3. Separate the dimensions of the data if present. 
        5. Because to calculate the residue, we need true values also. 
           Hence, this method returns true values also. 
        '''
        data_path = data_path if data_path else self.training_config.assets_path
        variables = self.training_config.extend_variables()
        full_data_path = [data_path / f"{var}_{time_step}.npy" for var in variables]
        numpy_data = [FVMNDataset.parse_numpy(self.training_config, data_path) for data_path in full_data_path]
        temp = list()
        for data in numpy_data:
            if len(data.shape) > 2:
                for i in range(2):
                    temp.append(data[:,:, i])
            else:
                temp.append(data)
        return temp
        
    def prepare_input_for_prediction(self, time_step:int|float, 
                                     data_path:Path, 
                                     first_prediction:bool,
                                     data:torch.Tensor=None) -> torch.Tensor:
        '''
        Prepare the input for the model for prediction. This will include the boundary data as well.

        Args
        ----
        time_step: int|float:
            If we are predicting for t then time_step = t-dt.
        data: torch.Tensor: 
            The output from the model after denormalizing and adding with the input [batch_size, num_features]
        data_path: Path: 
            if we predict for time step 5.03 then we need the original data for the time step 5.03 to get the boundary data.
        first_prediction: bool: 
            If this is the first prediction, we return the whole data. Else, we set the all the other except boundary to zero.

        Functionality
        -------------
        1. Get the boundary data for the time step from ground truth data.
        2. If this is not the first prediction, do the zero padding on the boundary of predicted data [198,198] -> [200,200].
        3. If not first prediction, internal nodes in the true data are set to zero.
        4. Because, we are using the same training_config.get_variables() to get the variables.
           We leverage this to get the index of U_x, U_y, T.
        5. If it is not the first prediction, we are setting U_x and T values in that iteration as previous values 
           and as the process progresses, we update the previous values with the predicted values.
        6. We save the predicted values here. In the prediction loop, we get the output for time(running_time) + dt.
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
        ground_truth = self.get_ground_truth_data(time_step, data_path, first_prediction)
        temp = deepcopy(ground_truth)

        if not first_prediction:

            # Modelling predicted data: adding zero padding to the predicted data.
            data = data.numpy()
            predicted_data_grid_x = self.training_config.grid_x - 2
            predicted_data_grid_y = self.training_config.grid_y - 2
            assert data.shape[0] == predicted_data_grid_x * predicted_data_grid_y, f"Shape of the data is {data.shape} but should be {(predicted_data_grid_x * predicted_data_grid_y, data.shape[-1])}"
            data = [data[:, i].reshape(self.training_config.grid_y-2, self.training_config.grid_x-2) for i in range(data.shape[-1])]

            # Copying just the boundary values from the ground truth data:
            for i in range(len(temp)): temp[i][1:-1, 1:-1] = 0 # Setting the internal nodes to zero.
            data = [np.pad(data[i], 1, mode="constant",constant_values=0) for i in range(len(data))]

            # Adding the zero padded predicted data to the zeroed internal nodes in the ground truth data.
            temp = [np.add(t,d) for t,d in zip(temp, data)]

            # Now, we completely have the predicted data in the temp variable.
            self.ux_matrix = temp[self.ux_index]
            self.uy_matrix = temp[self.uy_index]
            self.t_matrix = temp[self.t_index]

            ux_matrix_true = ground_truth[self.ux_index]
            uy_matrix_true = ground_truth[self.uy_index]
            # for i, var in enumerate(temp):

            #     # TODO: just to test the framework, we are hardcoding it for now. To concatenate vectors into a single matrix. 
            #     # While saving it to the numpy file, we have to concatenate the vectors into a single matrix, because it will be harder to 
            #     # change to foam format then. 
            #     np.save(data_path / f"{self.variables[i]}_{time_step}_predicted.npy", var)
            #     self.training_config.logger.info(f"Saved {self.variables[i]}_{time_step}_predicted.npy")

            ##################### Saving the predicted values #####################
            u_vector = np.concatenate([self.ux_matrix.reshape(-1,1),
                                       self.uy_matrix.reshape(-1,1)], axis=1)
            t_scalar = self.t_matrix.reshape(-1,1)
            np.save(data_path / f"U_{time_step}_predicted.npy", u_vector)
            np.save(data_path / f"T_{time_step}_predicted.npy", t_scalar)
            self.training_config.logger.info(f"Saved variables at {data_path}")
            ##################### Saved the predicted values #####################

             # Calculate the residue
            predicted_residual_mass = residual_mass(ux_matrix=self.ux_matrix, uy_matrix=self.uy_matrix)
            true_residual_mass = residual_mass(ux_matrix=ux_matrix_true, uy_matrix=uy_matrix_true)
            self.relative_residual_mass = predicted_residual_mass / true_residual_mass

            self.residual_mass.append(predicted_residual_mass)
            self.residual_momentum.append(residual_momentum(ux_matrix=self.ux_matrix, ux_matrix_prev=self.ux_matrix_prev, 
                                                            uy_matrix=self.uy_matrix, t_matrix=self.t_matrix))
            self.residual_heat.append(residual_heat(ux_matrix=self.ux_matrix, uy_matrix=self.uy_matrix,
                                                    t_matrix=self.t_matrix, t_matrix_prev=self.t_matrix_prev))
            
            # Update the previous values
            self.ux_matrix_prev = self.ux_matrix
            self.t_matrix_prev = self.t_matrix

            # Logging the residue values
            self.training_config.logger.info(f"Relative Residual Mass: {self.relative_residual_mass}")
            self.training_config.logger.info(f"Residual Heat: {self.residual_heat[-1]}")
            self.training_config.logger.info(f"Residual Momentum: {self.residual_momentum[-1]}")
            self.training_config.logger.info(f"Residual Mass: {self.residual_mass[-1]}")
        else: 
            self.ux_matrix_prev = temp[self.ux_index]
            self.t_matrix_prev = temp[self.t_index]
        
        temp_ = [FVMNDataset.add_feature(data) for data in temp]
        data = np.concatenate(temp_, axis=1)

        return torch.Tensor(data)

def main(base_config:BaseConfig, 
         openfoam_config:OpenfoamConfig, 
         training_config:TrainingConfig,
         network:torch.nn.Module=FVMNetwork,
         dataset:Dataset=FVMNDataset,
         visualize:bool=False):
    
    # Variables:
    # Training
    training_start_time = training_config.training_start_time
    training_end_time = training_config.training_end_time
    running_time = training_start_time
    training_loss = list()
    validation_loss = list()
    optimizer = training_config.optimizer
    loss_fn = training_config.loss

    # Metrices:
    heat_residue = list()
    momentum_residue = list()
    mass_residue = list()

    # Create model instance
    model = network(training_config)
    openfoam_utils = OpenfoamUtils(openfoam_config)

    ##################### RePIT: START #####################
    framework_start_time = timeit.default_timer()
    training_config.logger.info(f"Framework started at {framework_start_time}")
    while running_time < training_config.prediction_end_time:
        print("Running time: ", running_time)
        # Run CFD first:
        is_solver_run = openfoam_utils.run_solver(start_time=training_start_time, 
                                                  end_time=training_end_time, 
                                                  save_to_numpy=True)

        # Create dataset instance
        dataset = FVMNDataset(training_config=training_config, 
                              start_time=training_start_time, 
                              end_time=training_end_time, 
                              time_step=training_config.write_interval)
        train_loader, val_loader = get_dataloader(training_config, dataset)

        # Create trainer instance
        trainer = Trainer(training_config=training_config, model=model, optimizer=optimizer, loss_fn=loss_fn)

        # Train the model
        trainer.train(train_loader, val_loader, training_config.epochs)

        # Before prediction, load the best model: because we are using the same instance of self.model for prediction, hence last trained parameters will be used.
        model = trainer.load_model("best_model.pth")
        running_time = trainer.predict(prediction_start_time=training_end_time, 
                                       write_interval=training_config.write_interval)

        # Convert predicted numpy to foam
        is_numpy_to_foam = numpyToFoam(openfoam_config=openfoam_config, 
                                       latestML_time=float(running_time), 
                                       latestCFD_time=training_end_time)

        # Transfer learning
        training_start_time = running_time
        training_end_time = round(training_start_time + 2*training_config.write_interval,2)
        training_config.epochs = 20

        training_loss.extend(trainer.losses["train"])
        validation_loss.extend(trainer.losses["val"])
        heat_residue.extend(trainer.residual_heat)
        momentum_residue.extend(trainer.residual_momentum)
        mass_residue.extend(trainer.residual_mass)

    framework_end_time = timeit.default_timer()
    training_config.logger.info(f"Framework ended at {framework_end_time}")
    # Save the losses and residues
    np.save(training_config.model_dir / "training_loss.npy", training_loss)
    np.save(training_config.model_dir / "validation_loss.npy", validation_loss)
    np.save(training_config.model_dir / "heat_residue.npy", heat_residue)
    np.save(training_config.model_dir / "momentum_residue.npy", momentum_residue)
    np.save(training_config.model_dir / "mass_residue.npy", mass_residue)


if __name__ == "__main__":
    openfoam_config = OpenfoamConfig()
    training_config = TrainingConfig()
    base_config = BaseConfig()

    # Run the main function
    main(base_config, openfoam_config, training_config)