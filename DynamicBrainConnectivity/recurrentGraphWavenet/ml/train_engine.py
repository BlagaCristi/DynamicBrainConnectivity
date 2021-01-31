import copy
import datetime
import os

import numpy as np
import plotly
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR

from dualEnsembleClassifier.ml.WeightInitializer import WeightInitializer
from recurrentGraphWavenet.ml.GraphWavenetModel import GraphWavenetModel
from util.util import log


class TrainEngine:

    def __init__(self, number_of_nodes, blocks, layers, loader_splits, log_file, widget, output_directory,
                 use_previous_model, input_length, output_length, device, clipping_gradient = 3, learning_rate = 0.001,
                 weight_decay = 0.0001, dropout = 0.3, supports = None, gcn_bool = True, addaptadj = True,
                 aptinit = None, node_input_dimension = 1, residual_channels = 40,
                 dilation_channels = 40, skip_channels = 256, end_channels = 512, kernel_size = 2,
                 number_of_epochs = 100, learning_rate_adapt = 0.97, reconstruct_signal = True):
        self.number_of_nodes = number_of_nodes
        self.blocks = blocks
        self.layers = layers
        self.clipping_gradient = clipping_gradient
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.supports = supports
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.aptinit = aptinit
        self.node_input_dimension = node_input_dimension
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels
        self.kernel_size = kernel_size
        self.loader_splits = loader_splits
        self.number_of_epochs = number_of_epochs
        self.log_file = log_file
        self.widget = widget
        self.output_directory = output_directory
        self.learning_rate_adapt = learning_rate_adapt
        self.use_previous_model = use_previous_model
        self.reconstruct_signal = reconstruct_signal
        self.input_length = input_length
        self.output_length = output_length
        self.device = device

        self.train_loss_figure = None
        self.cross_signal_reconstruction_figure = None
        self.cross_signal_reconstruction_feedback_loop_figure = None
        self.index_list = None
        self.index_list_feedback_loop = None
        self.was_real_plotted = False
        self.was_real_plotted_feedback = False

        self.best_model = []

        self.loss_function = self.masked_mean_absolute_error

    def create_model(self):
        self.model = GraphWavenetModel(
            dropout = self.dropout,
            supports = self.supports,
            gcn_bool = self.gcn_bool,
            addaptadj = self.addaptadj,
            aptinit = self.aptinit,
            in_dim = self.node_input_dimension,
            out_dim = self.output_length,
            residual_channels = self.residual_channels,
            dilation_channels = self.dilation_channels,
            skip_channels = self.skip_channels,
            end_channels = self.end_channels,
            kernel_size = self.kernel_size,
            num_nodes = self.number_of_nodes,
            blocks = self.blocks,
            layers = self.layers,
            device = self.device
        )

        self.model.to(self.device)

        weightInit = WeightInitializer()
        weightInit.init_weights(self.model, 'xavier_normal_', { 'gain': 0.02 })

    def create_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate,
                                    weight_decay = self.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda = lambda epoch: self.learning_rate_adapt ** epoch)

    def masked_mean_absolute_error(self, preds, labels, null_val = 0.0):
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels != null_val)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds - labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

    def mean_absolute_scaled_error(self, input, real, predicted):

        mase = [0 for x in range(self.number_of_nodes)]

        for node in range(self.number_of_nodes):

            aux = []
            for index in range(1, len(input[0][0][node])):
                aux.append(float(abs(input[0][0][node][index] - input[0][0][node][index - 1])))

            mean = np.array(aux).mean()
            mase[node] += abs((real[0][0][node] - predicted[0][0][node]).detach().cpu().numpy().mean())

            if mean != 0:
                mase[node] /= mean

        return mase

    def train(self):

        log(f'Start training time: {str(datetime.datetime.now())}', self.log_file, self.widget)

        for split_index in range(len(self.loader_splits)):

            log(f'Train {split_index + 1}/{len(self.loader_splits)}. Start time: {str(datetime.datetime.now())}',
                self.log_file, self.widget)
            loader_split = self.loader_splits[split_index]
            train_loader = loader_split[0]
            cross_loader = loader_split[1]
            mean = loader_split[2]
            std = loader_split[3]

            if split_index == 0 or not self.use_previous_model:
                self.create_model()

            if split_index != 0 and self.use_previous_model:
                self.model = copy.deepcopy(self.best_model[-1])
                self.best_model.append(copy.deepcopy(self.model))
            else:
                self.best_model.append(copy.deepcopy(self.model))

            min_cross_error = 100000
            last_update = 0

            self.create_optimizer()

            losses = []
            cross_epoch_loss = []

            self.actual_epochs = 0

            for epoch in range(self.number_of_epochs):
                self.actual_epochs += 1

                self.model.train()

                log(f'Epoch: {epoch + 1}/{self.number_of_epochs}. Start time: {str(datetime.datetime.now())}',
                    self.log_file, self.widget)

                for input, real in train_loader:
                    input = input.float()
                    real = real.float()

                    input = input.to(self.device)
                    real = real.to(self.device)

                    self.optimizer.zero_grad()

                    # pad one zero at the beginning
                    input = nn.functional.pad(input, (1, 0, 0, 0))

                    # get prediction
                    predicted = self.model(input)

                    # transpose 2nd and 4th dimension (channel and time)
                    predicted = predicted.transpose(1, 3)

                    # compute loss
                    loss = self.loss_function(predicted, real)

                    # compute gradient
                    loss.backward()

                    # clip gradient
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_gradient)

                    # update model
                    self.optimizer.step()

                    # log loss
                    log(f'\tLoss: {float(loss)}; Time: {str(datetime.datetime.now())}', self.log_file, self.widget)
                    losses.append(float(loss))

                self.scheduler.step()

                cross_loss = 0
                count = 0

                self.model.eval()

                for input, real in cross_loader:
                    input = input.float()
                    real = real.float()

                    input = input.to(self.device)
                    real = real.to(self.device)

                    # pad one zero at the beginning
                    input = nn.functional.pad(input, (1, 0, 0, 0))

                    # get prediction
                    predicted = self.model(input)

                    # transpose 2nd and 4th dimension (channel and time)
                    predicted = predicted.transpose(1, 3)

                    # compute loss
                    loss = self.loss_function(real, predicted)

                    cross_loss += float(loss)
                    count += 1

                cross_epoch_loss.append(cross_loss / count)
                log(f'Cross Loss: {cross_epoch_loss[-1]}; Last update: {last_update}; '
                    f'Time: {str(datetime.datetime.now())}', self.log_file,
                    self.widget)

                if cross_epoch_loss[-1] <= min_cross_error:
                    self.best_model[-1] = copy.deepcopy(self.model)
                    last_update = 0
                    min_cross_error = cross_epoch_loss[-1]

                    log('New best model!', self.log_file, self.widget)

                else:
                    last_update += 1

                if last_update >= 10:
                    log("EARLY STOP", self.log_file, self.widget)
                    break

            if last_update < 10:
                self.best_model[split_index] = copy.deepcopy(self.model)

            self.best_model[split_index].eval()

            self.plot_train_loss(losses, cross_epoch_loss, split_index)

            self.best_model[split_index].save_model_to_file(
                os.path.join(self.output_directory, f'model_{split_index}.model'))

            with torch.no_grad():
                self.reconstruct_signal_from_loader(split_index, cross_loader, mean, std)

        log(f'End training time: {str(datetime.datetime.now())}', self.log_file, self.widget)

    def full_train(self):
        log(f'Train start time: {str(datetime.datetime.now())}',
            self.log_file, self.widget)

        loader_split = self.loader_splits
        train_loader = loader_split[0][0]

        self.create_model()
        self.create_optimizer()

        losses = []
        self.actual_epochs = 0

        for epoch in range(self.number_of_epochs):
            self.actual_epochs += 1

            self.model.train()

            log(f'Epoch: {epoch + 1}/{self.number_of_epochs}. Start time: {str(datetime.datetime.now())}',
                self.log_file, self.widget)

            for input, real in train_loader:
                input = input.float()
                real = real.float()

                input = input.to(self.device)
                real = real.to(self.device)

                self.optimizer.zero_grad()

                # pad one zero at the beginning
                input = nn.functional.pad(input, (1, 0, 0, 0))

                # get prediction
                predicted = self.model(input)

                # transpose 2nd and 4th dimension (channel and time)
                predicted = predicted.transpose(1, 3)

                # compute loss
                loss = self.loss_function(predicted, real)

                # compute gradient
                loss.backward()

                # clip gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_gradient)

                # update model
                self.optimizer.step()

                # log loss
                log(f'\tLoss: {float(loss)}; Time: {str(datetime.datetime.now())}', self.log_file, self.widget)
                losses.append(float(loss))

            self.scheduler.step()

        self.best_model = [self.model]
        self.best_model[0].eval()

        self.plot_train_loss(losses, None, 0)

        self.best_model[0].save_model_to_file(
            os.path.join(self.output_directory, f'model.model'))

        log(f'End training time: {str(datetime.datetime.now())}', self.log_file, self.widget)

    def reconstruct_signal_from_loader(self, split_index, loader, mean, std):
        cross_mase_file = os.path.join(self.output_directory, f'cross_mase_{split_index}.txt')
        cross_loss_file = os.path.join(self.output_directory, f'cross_loss_{split_index}.txt')

        if self.output_length == 1:
            real_list = [[] for x in range(self.number_of_nodes)]
            predicted_list = [[] for x in range(self.number_of_nodes)]

            mase = np.array([0.0 for x in range(self.number_of_nodes)])
            count = 0

            for input, real in loader:
                input = input.float()
                real = real.float()

                input = input.to(self.device)
                real = real.to(self.device)

                # pad one zero at the beginning
                input = nn.functional.pad(input, (1, 0, 0, 0))

                # get prediction
                predicted = self.best_model[split_index](input)

                # transpose 2nd and 4th dimension (channel and time)
                predicted = predicted.transpose(1, 3)

                loss = self.loss_function(real, predicted)
                log(f'Loss: {loss}', file = cross_loss_file)

                for node in range(self.number_of_nodes):
                    real_list[node].append(float(real[0][0][node][0]))
                    predicted_list[node].append(float(predicted[0][0][node][0]))

                mase += self.mean_absolute_scaled_error(input, real, predicted)

                count += 1

            mase = mase / count

            for node in range(self.number_of_nodes):
                log(f'Channel {node}. MASE: {mase[node]}', file = cross_mase_file, widget = None)

            log(f'Overall MASE: {mase.mean()}', file = cross_mase_file, widget = None)

            if self.reconstruct_signal:
                real_list = np.array(real_list)
                predicted_list = np.array(predicted_list)

                real_list = (real_list * std) + mean
                predicted_list = (predicted_list * std) + mean

                if self.cross_signal_reconstruction_figure is None:
                    self.cross_signal_reconstruction_figure = go.Figure()
                    self.index_list = [i for i in range(len(real_list[0]))]

                for node in range(self.number_of_nodes):
                    if not self.was_real_plotted:
                        self.cross_signal_reconstruction_figure.add_trace(
                            go.Scatter(
                                x = self.index_list[-len(real_list[node]):],
                                y = real_list[node],
                                mode = 'lines',
                                name = f'Real ch. {node}_{split_index}'
                            )
                        )

                    self.cross_signal_reconstruction_figure.add_trace(
                        go.Scatter(
                            x = self.index_list[-len(real_list[node]):],
                            y = predicted_list[node],
                            mode = 'lines',
                            name = f'Predict ch. {node}_{split_index}'
                        )
                    )

                plotly.offline.plot(self.cross_signal_reconstruction_figure,
                                    filename = os.path.join(self.output_directory, f'CrossSignal_{split_index}.html'),
                                    auto_open = False)
        else:
            raise NotImplementedError('GW with output greater than 1 not implemented.')

        self.was_real_plotted = True

    def plot_train_loss(self, losses, cross_epoch_loss, split_index):
        number_of_batches = len(losses) // self.actual_epochs

        epoch_losses = []
        epoch_index = []

        if self.train_loss_figure is None:
            self.train_loss_figure = go.Figure()

        for epoch in range(self.actual_epochs):

            epoch_loss = 0

            for batch in range(number_of_batches):
                self.train_loss_figure.add_trace(
                    go.Scatter(
                        x = [epoch * number_of_batches + batch],
                        y = [losses[epoch * number_of_batches + batch]],
                        mode = 'markers',
                        name = f'{epoch}_{batch}_{split_index}'
                    )
                )
                epoch_loss += losses[epoch * number_of_batches + batch]

            epoch_loss /= number_of_batches
            epoch_losses.append(epoch_loss)
            epoch_index.append(epoch * number_of_batches)

        self.train_loss_figure.add_trace(
            go.Scatter(
                x = [i for i in range(len(losses))],
                y = losses,
                mode = 'lines',
                name = f'Train Loss_{split_index}'
            )
        )

        self.train_loss_figure.add_trace(
            go.Scatter(
                x = epoch_index,
                y = epoch_losses,
                mode = 'lines',
                name = f'Epoch train loss_{split_index}'
            )
        )

        if cross_epoch_loss is not None:
            self.train_loss_figure.add_trace(
                go.Scatter(
                    x = epoch_index,
                    y = cross_epoch_loss,
                    mode = 'lines',
                    name = f'Epoch cross loss_{split_index}'
                )
            )

        plotly.offline.plot(self.train_loss_figure,
                            filename = os.path.join(self.output_directory, f'TrainCrossLoss_{split_index}.html'),
                            auto_open = False)

    def save_weight_matrix(self, path):
        # get matrix
        nodevec1, nodevec2, matrix = self.best_model[-1].get_node_embeddings()

        matrix = F.relu(torch.mm(nodevec1, nodevec2))

        matrix.detach().cpu().double().numpy().tofile(
            os.path.join(path, f'WeightMatrix.dat'))

