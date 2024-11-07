import torch
import torch.nn as nn
import torch.optim as optim

class FVMN(nn.Module):
    def __init__(self, net_a, net_p, net_ph, net_u, net_v, Rs, x_val, y_val, x_pinn, y_pinn):
        super(FVMN, self).__init__()
        self.net_a = net_a
        self.net_p = net_p
        self.net_ph = net_ph
        self.net_u = net_u
        self.net_v = net_v
        self.Rs = Rs
        self.x_val = x_val
        self.y_val = y_val
        self.x_pinn = x_pinn
        self.y_pinn = y_pinn

        # Define loss trackers for monitoring
        self.val_loss_tracker = []
        self.loss_tot_tracker = []

    def forward(self, x):
        # The forward method isn't used directly since we're handling multiple networks separately
        pass

    def train_step(self, data, optimizer):
        x_pre, y = data
        x = x_pre[:, 0:25]
        x_ex = x_pre[:, 25:30].float()

        y_a = y[:, 0:1]
        y_p = y[:, 1:2]
        y_ph = y[:, 2:3]
        y_u = y[:, 3:4]
        y_v = y[:, 4:5]

        optimizer.zero_grad()

        # Forward pass
        y_pred_a = self.net_a(x)
        y_pred_p = self.net_p(x)
        y_pred_ph = self.net_ph(x)
        y_pred_u = self.net_u(x)
        y_pred_v = self.net_v(x)

        # Calculate loss
        loss_fn = nn.MSELoss()
        loss_a_mse = loss_fn(y_pred_a, y_a)
        loss_p_mse = loss_fn(y_pred_p, y_p)
        loss_ph_mse = loss_fn(y_pred_ph, y_ph)
        loss_u_mse = loss_fn(y_pred_u, y_u)
        loss_v_mse = loss_fn(y_pred_v, y_v)

        # Total loss
        loss_tot = loss_a_mse + loss_p_mse + loss_ph_mse + loss_u_mse + loss_v_mse

        # Backward pass and optimization
        loss_tot.backward()
        optimizer.step()

        # Validation loss
        with torch.no_grad():
            x_val = self.x_val
            y_val = self.y_val

            x_val_2 = x_val[:, 0:25]
            y_val_a = y_val[:, 0:1]
            y_val_p = y_val[:, 1:2]
            y_val_ph = y_val[:, 2:3]
            y_val_u = y_val[:, 3:4]
            y_val_v = y_val[:, 4:5]

            y_val_pred_a = self.net_a(x_val_2)
            y_val_pred_p = self.net_p(x_val_2)
            y_val_pred_ph = self.net_ph(x_val_2)
            y_val_pred_u = self.net_u(x_val_2)
            y_val_pred_v = self.net_v(x_val_2)

            loss_val_a_mse = loss_fn(y_val_pred_a, y_val_a)
            loss_val_p_mse = loss_fn(y_val_pred_p, y_val_p)
            loss_val_ph_mse = loss_fn(y_val_pred_ph, y_val_ph)
            loss_val_u_mse = loss_fn(y_val_pred_u, y_val_u)
            loss_val_v_mse = loss_fn(y_val_pred_v, y_val_v)

            val_loss = loss_val_a_mse + loss_val_p_mse + loss_val_ph_mse + loss_val_u_mse + loss_val_v_mse

        # Update loss trackers
        self.val_loss_tracker.append(val_loss.item())
        self.loss_tot_tracker.append(loss_tot.item())

        return {'val_loss': val_loss.item(), 'loss_tot': loss_tot.item()}

    def get_metrics(self):
        return {'val_loss': self.val_loss_tracker, 'loss_tot': self.loss_tot_tracker}
