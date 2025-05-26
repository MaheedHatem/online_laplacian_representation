from .trainer import Trainer
from src.env.grid.utils import load_eig
import gymnasium as gym
from src.env.wrapper.norm_obs import NormObs
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from itertools import product

class GDOOptimizer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coefficient_vector = torch.ones(self.d).to(self.device)
        self.additional_params = {
            'barrier_coefs': nn.Parameter(torch.tril(self.barrier_initial_val * torch.ones((1, 1)))).to(self.device),
            'quadratic_errors': torch.zeros((1, 1)).to(self.device),
        }

    def train_step(self, train_batch) -> None:
        # Compute the gradients and associated intermediate metrics
        self.encoder_optimizer.zero_grad()
        loss, aux = self.loss_function(train_batch)

        loss.backward()
        
        self.encoder_optimizer.step()
        self.update_training_state(aux[1])

        return aux[0]
       
    def compute_graph_drawing_loss(self, start_representation, end_representation):
        '''Compute reprensetation distances between start and end states'''
        
        graph_induced_norms = ((start_representation - end_representation)**2).mean(0)
        loss = graph_induced_norms.dot(self.coefficient_vector)
        
        graph_induced_norm_dict = {
            f'graph_norm({i})': graph_induced_norms[i]
            for i in range(self.d)
        }       

        return loss, graph_induced_norm_dict

    def compute_orthogonality_error_matrix(self, represetantation_batch_1, represetantation_batch_2):
        n = represetantation_batch_1.shape[0]

        inner_product_matrix_1 = torch.einsum(
            'ij,ik->jk',
            represetantation_batch_1,
            represetantation_batch_1.detach(),
        ) / n

        inner_product_matrix_2 = torch.einsum(
            'ij,ik->jk',
            represetantation_batch_2,
            represetantation_batch_2.detach(),
        ) / n

        error_matrix_1 = torch.tril(inner_product_matrix_1 - torch.eye(self.d).to(self.device))
        error_matrix_2 = torch.tril(inner_product_matrix_2 - torch.eye(self.d).to(self.device))
        quadratic_error_matrix = error_matrix_1 * error_matrix_2

        inner_dict = {
            f'inner({i},{j})': inner_product_matrix_1[i,j]
            for i, j in product(range(self.d), range(self.d))
            if i >= j
        }

        error_matrix_dict = {
            'quadratic_errors': quadratic_error_matrix,
        }

        return error_matrix_dict, inner_dict

    def compute_orthogonality_loss(self, error_matrix_dict):
        # Compute the losses
        barrier_coefficients = self.additional_params['barrier_coefs']
        quadratic_error_matrix = error_matrix_dict['quadratic_errors']

        barrier_loss = barrier_coefficients.detach()[0,0] * quadratic_error_matrix.sum()


        barrier_dict = {
            f'barrier_coeff': barrier_coefficients[0,0],
        }
        
        return barrier_loss

    def update_error_estimates(self, errors) -> Tuple[dict]:  
        with torch.no_grad():
            updates = {}
            # Get old error estimates
            old = self.additional_params ['quadratic_errors']
            norm_old = torch.linalg.norm(old)
            
            # Set update rate to 1 in the first iteration
            init_coeff = torch.isclose(norm_old, torch.tensor(0.0), rtol=1e-10, atol=1e-13) 
            non_init_update_rate = self.q_error_update_rate
            update_rate = init_coeff + (~init_coeff) * non_init_update_rate
            
            # Update error estimates
            update = old + update_rate * (errors['quadratic_errors'] - old)   # The first update might be too large
            updates['quadratic_errors'] = update
            
            return updates
        
    def loss_function(
            self, train_batch, **kwargs
        ) -> Tuple[np.ndarray]:

        # Get representations
        start_representation, end_representation, \
            constraint_representation_1, constraint_representation_2 \
                = self.encode_states(train_batch)
        
        # Compute primal loss
        graph_loss, graph_induced_norm_dict = self.compute_graph_drawing_loss(
            start_representation, end_representation
        )
        error_matrix_dict, inner_dict = self.compute_orthogonality_error_matrix(
            constraint_representation_1, constraint_representation_2,
        )

        # Compute dual loss
        barrier_loss = self.compute_orthogonality_loss(error_matrix_dict)
        
        # Update error estimates
        error_update = self.update_error_estimates(error_matrix_dict)

        # Compute total loss
        loss = graph_loss +  barrier_loss

        # Generate dictionary with losses for logging
        metrics_dict = {
            'train_loss': loss,
            'graph_loss': graph_loss,
            'dual_loss': 0,
            'barrier_loss': barrier_loss,
        }

        # Add additional metrics
        metrics_dict.update(graph_induced_norm_dict)
        metrics_dict.update(inner_dict)
        metrics = (loss, graph_loss, torch.as_tensor(0), barrier_loss, metrics_dict)
        aux = (metrics, error_update)

        return loss, aux

    def additional_update_step(self):        
        self.update_barrier_coefficients()
        

    def update_barrier_coefficients(self):
        '''
            Update barrier coefficients using some approximation 
            of the barrier gradient in the modified lagrangian.
        '''
        with torch.no_grad():
            barrier_coefficients = self.additional_params['barrier_coefs']
            quadratic_error_matrix = self.additional_params['quadratic_errors']
            updates = torch.clip(quadratic_error_matrix, min=0, max=None).mean()

            updated_barrier_coefficients = barrier_coefficients + self.lr_barrier_coefs * updates

            # Clip coefficients to be in the range [min_barrier_coefs, max_barrier_coefs]
            updated_barrier_coefficients = torch.clip(
                updated_barrier_coefficients,
                min=self.min_barrier_coefs,
                max=self.max_barrier_coefs,
            ) 

            # Update params, making sure that the coefficients are lower triangular
            self.additional_params['barrier_coefs'] = updated_barrier_coefficients

    def update_training_state(self, error_update):
        '''Update error estimates'''

        self.additional_params['quadratic_errors'] = error_update['quadratic_errors']