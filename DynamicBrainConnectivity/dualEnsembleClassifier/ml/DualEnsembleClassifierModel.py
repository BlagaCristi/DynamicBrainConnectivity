import numpy as np
import pandas
import pandas as pd
import plotly
import plotly.express as px
import torch
from sklearn.metrics import classification_report
from torch import nn

from util.util import log, plot_histogram


class DualEnsembleClassifierModel(nn.Module):

    def __init__(self, sizes_array, number_of_channels):
        """
        Sizes array has the following structure:
        (
            [first_input_size, first_hidden_size_1,...,first_hidden_size_n,first_output_size],
            [second_input_size, second_hidden_size_1,...,second_hidden_size_n,second_output_size]
        )
        The number of elements MUST BE THE SAME
        Create layers based on this array
        """
        super(DualEnsembleClassifierModel, self).__init__()

        self.create_DEC(number_of_channels, sizes_array)

    def create_DEC(self, number_of_channels, sizes_array):
        self.layers = []
        for channel in range(number_of_channels):
            first_layers, second_layers = self.create_DNN(sizes_array)

            self.layers.append(nn.ModuleList([
                nn.ModuleList(first_layers),
                nn.ModuleList(second_layers)
            ]
            ))
        self.layers = nn.ModuleList(self.layers)
        self.number_of_channels = number_of_channels

    def create_DNN(self, sizes_array):
        count = 0
        first_layers = []
        second_layers = []
        # create all the layers except the last one
        while count < len(sizes_array[0]) - 2:
            first_layers.append(
                nn.Linear(
                    sizes_array[0][count],
                    sizes_array[0][count + 1]
                )
            )
            first_layers.append(
                nn.Sigmoid()
            )
            first_layers.append(
                nn.Dropout(0.5)
            )
            second_layers.append(
                nn.Linear(
                    sizes_array[1][count],
                    sizes_array[1][count + 1]
                )
            )
            second_layers.append(
                nn.Sigmoid()
            )
            second_layers.append(
                nn.Dropout(0.5)
            )
            count += 1
        # the last layer does not need to have Sigmoid after it
        first_layers.append(
            nn.Linear(
                sizes_array[0][-2],
                sizes_array[0][-1]
            )
        )
        second_layers.append(
            nn.Linear(
                sizes_array[1][-2],
                sizes_array[1][-1]
            )
        )
        return first_layers, second_layers

    def forward(self, first_tensor, second_tensor, channel):
        """
        Passes the input tensor through the neural network, one layer at a time
        """
        first_output_tensor = first_tensor
        second_output_tensor = second_tensor

        # for the first neural network
        for index, layer in enumerate(self.layers[channel][0]):
            # pass through each layer
            first_output_tensor = layer(first_output_tensor)

        # for the second neural network
        for index, layer in enumerate(self.layers[channel][1]):
            # pass through each layer
            second_output_tensor = layer(second_output_tensor)

        return first_output_tensor, second_output_tensor

    def fit(self, viz, viz_name, train_loader, cross_loader, html_file, number_epochs, learning_rate, widget):
        """
        Fits the model based on the train dataset and number of epochs.
        """

        # define cost
        first_criterion = nn.CrossEntropyLoss()

        # weighted classes for response
        weights = [1.0, 2.5, 1.5]
        weights_class = torch.FloatTensor(weights)
        second_criterion = nn.CrossEntropyLoss(weight = weights_class)

        # define optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr = learning_rate)

        epochs_number = [x for x in range(1, number_epochs + 1)]
        el_array = []
        cv_array = []
        first_array = []
        second_array = []
        first_cross_array = []
        second_cross_array = []

        # for each pass through the examples
        for epoch in range(number_epochs):
            epoch_loss = 0
            first_loss_epoch = 0
            second_loss_epoch = 0
            count = 0

            log(f'Epoch {epoch + 1}/{number_epochs}', file = None, widget = widget)

            # switch to train mode (Dropout used)
            self.train()

            # adjust the model one batch at a time
            for batch in train_loader:

                first_agg_output = None
                second_agg_output = None

                # for each channel
                for channel in range(self.number_of_channels):

                    # set the tensors to require grad
                    batch[channel * 2].requires_grad = True
                    batch[channel * 2 + 1].requires_grad = True

                    # compute output
                    first_output, second_output = self(batch[channel * 2].float(), batch[channel * 2 + 1].float(),
                                                       channel)

                    # "vote" = ensemble
                    if channel == 0:
                        first_agg_output = first_output
                        second_agg_output = second_output
                    else:
                        first_agg_output += first_output
                        second_agg_output += second_output

                # reset the gradients
                optimizer.zero_grad()

                # compute loss
                first_loss = 1 + first_criterion(first_agg_output, batch[-3].long())
                second_loss = 1 + second_criterion(second_agg_output, batch[-2].long())

                loss = self.dual_loss_aggregation(first_loss, second_loss)

                # backward propagate through the network
                loss.backward()

                # update weights
                optimizer.step()

                # compute epoch loss
                epoch_loss += loss.item()
                first_loss_epoch += first_loss.item()
                second_loss_epoch += second_loss.item()
                count += 1.0

            # compute cross validation loss

            # set to eval mode (Dropout not used)
            self.eval()
            cv_loss = 0
            first_cross_loss = 0
            second_cross_loss = 0
            count = 0

            # for each batch
            for batch in cross_loader:
                first_agg_output = None
                second_agg_output = None

                # for each channel
                for channel in range(self.number_of_channels):

                    # set the tensors to require grad
                    batch[channel * 2].requires_grad = True
                    batch[channel * 2 + 1].requires_grad = True

                    # compute output
                    first_output, second_output = self(batch[channel * 2].float(), batch[channel * 2 + 1].float(),
                                                       channel)

                    # "vote" = ensemble
                    if channel == 0:
                        first_agg_output = first_output
                        second_agg_output = second_output
                    else:
                        first_agg_output += first_output
                        second_agg_output += second_output

                # compute loss
                first_loss = 1 + first_criterion(first_agg_output, batch[-3].long())
                second_loss = 1 + second_criterion(second_agg_output, batch[-2].long())

                loss = self.dual_loss_aggregation(first_loss, second_loss)

                first_cross_loss += first_loss.item()
                second_cross_loss += second_loss.item()

                cv_loss += loss.item()

                count += 1

            # append losses
            el_array.append(epoch_loss / count)
            first_array.append(first_loss_epoch / count)
            second_array.append(second_loss_epoch / count)
            cv_array.append(cv_loss / count)
            first_cross_array.append(first_cross_loss / count)
            second_cross_array.append(second_cross_loss / count)

            # print to VISDOM if available
            if viz:
                self.plot_to_vizdom(count, cv_loss, epoch, epoch_loss, first_cross_array, first_loss_epoch,
                                    second_cross_array, second_loss_epoch, viz, viz_name)

        # plot losses
        self.plot_elt_and_elcv(epochs_number, el_array, cv_array, first_array, second_array, first_cross_array,
                               second_cross_array, html_file)

    def plot_to_vizdom(self, count, cv_loss, epoch, epoch_loss, first_cross_array, first_loss_epoch, second_cross_array,
                       second_loss_epoch, viz, viz_name):
        viz.line(X = np.array([epoch]), Y = np.array([epoch_loss / count]), win = 'Epoch number',
                 name = 'ELT ' + viz_name, update = 'append')
        viz.line(X = np.array([epoch]), Y = np.array([cv_loss]), win = 'Epoch number',
                 name = 'ELCV ' + viz_name, update = 'append')
        viz.line(X = np.array([epoch]), Y = np.array([first_loss_epoch / count]), win = 'Epoch number',
                 name = 'Stimulus Loss ' + viz_name, update = 'append')
        viz.line(X = np.array([epoch]), Y = np.array([second_loss_epoch / count]), win = 'Epoch number',
                 name = 'Response Loss ' + viz_name, update = 'append')
        viz.line(X = np.array([epoch]), Y = np.array([first_cross_array[-1]]), win = 'Epoch number',
                 name = 'Stimulus Cross Loss ' + viz_name, update = 'append')
        viz.line(X = np.array([epoch]), Y = np.array([second_cross_array[-1]]), win = 'Epoch number',
                 name = 'Response Cross Loss ' + viz_name, update = 'append')

    def plot_elt_and_elcv(self, epochs_number, elt_array, elcv_array, stimulus_array, response_array,
                          stimulus_cross_array,
                          response_cross_array, html_file):
        """
        Plot the loss for train together with the loss on cross validation on the same plot.
        """

        epochs_number = np.array(epochs_number)
        elt_array = np.array(elt_array)
        elcv_array = np.array(elcv_array)

        elt_frame = pd.DataFrame()
        elt_frame['epoch_number'] = epochs_number
        elt_frame['loss'] = elt_array
        elt_frame['source'] = 'Train loss aggregated'

        elcv_frame = pd.DataFrame()
        elcv_frame['epoch_number'] = epochs_number
        elcv_frame['loss'] = elcv_array
        elcv_frame['source'] = 'Cross loss aggregated'

        stimulus_frame = pd.DataFrame()
        stimulus_frame['epoch_number'] = epochs_number
        stimulus_frame['loss'] = stimulus_array
        stimulus_frame['source'] = 'Stimulus loss'

        response_frame = pd.DataFrame()
        response_frame['epoch_number'] = epochs_number
        response_frame['loss'] = response_array
        response_frame['source'] = 'Response loss'

        stimulus_cross_frame = pd.DataFrame()
        stimulus_cross_frame['epoch_number'] = epochs_number
        stimulus_cross_frame['loss'] = stimulus_cross_array
        stimulus_cross_frame['source'] = 'Stimulus cross loss'

        response_cross_frame = pd.DataFrame()
        response_cross_frame['epoch_number'] = epochs_number
        response_cross_frame['loss'] = response_cross_array
        response_cross_frame['source'] = 'Response cross loss'

        data_frames = [elt_frame, elcv_frame, stimulus_frame, response_frame, stimulus_cross_frame,
                       response_cross_frame]
        final_frame = pd.concat(data_frames, axis = 0)
        fig = px.line(
            final_frame,
            x = 'epoch_number',
            y = 'loss',
            color = 'source',
            title = 'Train/Cross loss'
        )
        plotly.offline.plot(fig, filename = html_file, auto_open = False)

    def dual_loss_aggregation(self, first_loss, second_loss):
        loss = first_loss * second_loss
        if first_loss < second_loss:
            loss += (second_loss - first_loss)
        else:
            loss += (first_loss - second_loss)
        return loss

    def predict(self, test_loader, first_file_csv, first_file_txt, second_file_csv, second_file_txt, channel_file,
                first_class_names, second_class_names, number_of_subjects):

        """
        Computes the confusion matrix for the test dataset
        """

        # the two needed arrays for computation
        first_actual_output = []
        first_expected_output = []

        second_actual_output = []
        second_expected_output = []

        # set to eval mode (Dropout not used)
        self.eval()

        # for each batch
        for batch in test_loader:
            first_agg_output = None
            second_agg_output = None

            # for each channel
            for channel in range(self.number_of_channels):

                # set the tensors to require grad
                batch[channel * 2].requires_grad = True
                batch[channel * 2 + 1].requires_grad = True

                # compute output
                first_output, second_output = self(batch[channel * 2].float(), batch[channel * 2 + 1].float(), channel)

                # "vote" = ensemble
                if channel == 0:
                    first_agg_output = first_output
                    second_agg_output = second_output
                else:
                    first_agg_output += first_output
                    second_agg_output += second_output

            # Get predictions from the maximum value
            _, first_predicted = torch.max(first_agg_output.data, 1)
            _, second_predicted = torch.max(second_agg_output.data, 1)

            first_predicted = first_predicted.tolist()
            second_predicted = second_predicted.tolist()
            first_labels = batch[-3].tolist()
            second_labels = batch[-2].tolist()

            # extend the arrays with the predicted values and corresponding labels
            first_actual_output.extend(first_predicted)
            first_expected_output.extend(first_labels)
            second_actual_output.extend(second_predicted)
            second_expected_output.extend(second_labels)

        # print the classification reports
        first_report = classification_report(y_true = np.array(first_expected_output),
                                             y_pred = np.array(first_actual_output),
                                             target_names = first_class_names, output_dict = True)
        first_df = pandas.DataFrame(first_report).transpose()
        first_df.to_csv(first_file_csv, index = False)

        second_report = classification_report(y_true = np.array(second_expected_output),
                                              y_pred = np.array(second_actual_output),
                                              target_names = second_class_names, output_dict = True)

        second_df = pandas.DataFrame(second_report).transpose()
        second_df.to_csv(second_file_csv, index = False)

        log(
            classification_report(y_true = np.array(first_expected_output),
                                  y_pred = np.array(first_actual_output),
                                  target_names = first_class_names)
            , file = first_file_txt,
            widget = None
        )

        log(
            classification_report(y_true = np.array(second_expected_output),
                                  y_pred = np.array(second_actual_output),
                                  target_names = second_class_names)
            , file = second_file_txt,
            widget = None
        )

    def predict_for_classification_statistics(self, loader, trial_lengths, stimulus_output, response_output,
                                              output_directory, label, median_value):

        """
        Computes the confusion matrix for the test dataset
        """
        stimulus_correctly_classified = [[] for x in range(stimulus_output)]
        stimulus_incorrectly_classified = [[] for x in range(stimulus_output)]
        response_incorrectly_classified = [[] for x in range(response_output)]
        response_correctly_classified = [[] for x in range(response_output)]

        # set to eval mode (Dropout not used)
        self.eval()

        # for each batch
        for batch in loader:
            first_agg_output = None
            second_agg_output = None

            # for each channel
            for channel in range(self.number_of_channels):

                # set the tensors to require grad
                batch[channel * 2].requires_grad = True
                batch[channel * 2 + 1].requires_grad = True

                # compute output
                first_output, second_output = self(batch[channel * 2].float(), batch[channel * 2 + 1].float(), channel)

                # "vote" = ensemble
                if channel == 0:
                    first_agg_output = first_output
                    second_agg_output = second_output
                else:
                    first_agg_output += first_output
                    second_agg_output += second_output

            # Get predictions from the maximum value
            _, first_predicted = torch.max(first_agg_output.data, 1)
            _, second_predicted = torch.max(second_agg_output.data, 1)

            first_predicted = first_predicted.tolist()
            second_predicted = second_predicted.tolist()
            first_labels = batch[-4].tolist()
            second_labels = batch[-3].tolist()
            subject = batch[-2].tolist()
            trial_index = batch[-1].tolist()

            trial_length = trial_lengths[subject[0] - 1][trial_index[0] - 30]

            if first_predicted[0] == first_labels[0]:
                stimulus_correctly_classified[first_predicted[0]].append(trial_length)
            else:
                stimulus_incorrectly_classified[first_labels[0]].append(trial_length)

            if second_predicted[0] == second_labels[0]:
                response_correctly_classified[second_predicted[0]].append(trial_length)
            else:
                response_incorrectly_classified[second_labels[0]].append(trial_length)

        for i in range(0, stimulus_output):
            plot_histogram(stimulus_correctly_classified[i], stimulus_incorrectly_classified[i],
                           'Stimulus Correctly Classified', 'Stimulus Incorrectly Classified',
                           output_directory,
                           f"stimulus_classified_{label}_{i}", median_value)

        for i in range(0, response_output):
            plot_histogram(response_correctly_classified[i], response_incorrectly_classified[i],
                           'Response Correctly Classified', 'Response Incorrectly Classified', output_directory,
                           f"response_classified_{label}_{i}", median_value)

        stimulus_correctly_classified = [item for sublist in stimulus_correctly_classified for item in sublist]
        stimulus_incorrectly_classified = [item for sublist in stimulus_incorrectly_classified for item in sublist]
        response_correctly_classified = [item for sublist in response_correctly_classified for item in sublist]
        response_incorrectly_classified = [item for sublist in response_incorrectly_classified for item in sublist]

        plot_histogram(stimulus_correctly_classified, stimulus_incorrectly_classified, 'Stimulus Correctly Classified',
                       'Stimulus Incorrectly Classified', output_directory,
                       f"stimulus_classified_{label}", median_value)
        plot_histogram(response_correctly_classified, response_incorrectly_classified, 'Response Correctly Classified',
                       'Response Incorrectly Classified', output_directory,
                       f"response_classified_{label}", median_value)
        return stimulus_correctly_classified, stimulus_incorrectly_classified, response_correctly_classified, response_incorrectly_classified

    def save_model_to_file(self, full_path):
        """
            Saves the current model to a file in order to be able to use it later
        """

        # save model
        torch.save(self.state_dict(), full_path)

    def load_model_from_file(self, full_path):
        """
            Load model from file
        """

        # load model
        self.load_state_dict(torch.load(full_path))

        # necessary step for loading
        self.eval()
