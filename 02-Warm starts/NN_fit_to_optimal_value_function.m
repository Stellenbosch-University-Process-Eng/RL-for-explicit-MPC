%% Code used to fit a neural network to the optimal value function for the 
%% nominal process parameter set.
%% Name: Edward Bras
%% Date: 2024-04-26
clc;clearvars -except ans;close all;
rng(2);

%% specify valve position and number of levels for each state
gamma_val = 0.3;
numLevels = 2;

%% load structure containing all training data
gammaMult100 = 100*gamma_val;
filename_test = sprintf('data_gamma_%03d_numlevels_%d',gammaMult100,numLevels);
test_file = load(filename_test);
test_varname = fieldnames(test_file);
test_varname = test_varname{1};

training_data = test_file.(test_varname);

%% data processing
data.SP_input_1_reshaped = reshape(training_data.SP_1_grid,1,size(training_data.SP_1_grid,1)*size(training_data.SP_1_grid,2)*size(training_data.SP_1_grid,3)*size(training_data.SP_1_grid,4)*size(training_data.SP_1_grid,5)*size(training_data.SP_1_grid,6));       % reshape SP coordinates
data.SP_input_2_reshaped = reshape(training_data.SP_2_grid,1,size(training_data.SP_2_grid,1)*size(training_data.SP_2_grid,2)*size(training_data.SP_2_grid,3)*size(training_data.SP_2_grid,4)*size(training_data.SP_2_grid,5)*size(training_data.SP_2_grid,6));       % reshape SP coordinates (2023-11-16)
data.H_one_reshaped = reshape(training_data.H_1_grid,1,size(training_data.H_1_grid,1)*size(training_data.H_1_grid,2)*size(training_data.H_1_grid,3)*size(training_data.H_1_grid,4)*size(training_data.H_1_grid,5)*size(training_data.H_1_grid,6));    % reshape H1 coordinates (2023-11-01)
data.H_two_reshaped = reshape(training_data.H_2_grid,1,size(training_data.H_2_grid,1)*size(training_data.H_2_grid,2)*size(training_data.H_2_grid,3)*size(training_data.H_2_grid,4)*size(training_data.H_2_grid,5)*size(training_data.H_2_grid,6));    % reshape H2 coordinates (2023-11-01)
data.H_three_reshaped = reshape(training_data.H_3_grid,1,size(training_data.H_3_grid,1)*size(training_data.H_3_grid,2)*size(training_data.H_3_grid,3)*size(training_data.H_3_grid,4)*size(training_data.H_3_grid,5)*size(training_data.H_3_grid,6));  % reshape H3 coordinates (2023-11-01)
data.H_four_reshaped = reshape(training_data.H_4_grid,1,size(training_data.H_4_grid,1)*size(training_data.H_4_grid,2)*size(training_data.H_4_grid,3)*size(training_data.H_4_grid,4)*size(training_data.H_4_grid,5)*size(training_data.H_4_grid,6));   % reshape H4 coordinates (2023-11-01)

data.P_mat = [data.SP_input_1_reshaped;data.SP_input_2_reshaped;data.H_one_reshaped;data.H_two_reshaped;data.H_three_reshaped;data.H_four_reshaped]; % create a matrix containing all inputs (2023-11-16)

[~,PS_input] = mapminmax(data.P_mat,-1,1); % structure containing state scaling data

%%
data.optimal_costs_reshaped = reshape(training_data.optimalCosts,1,size(training_data.optimalCosts,1)*size(training_data.optimalCosts,2)*size(training_data.optimalCosts,3)*size(training_data.optimalCosts,4)*size(training_data.optimalCosts,5)*size(training_data.optimalCosts,6)); % reshape u1 coordinates (2023-11-01)

data.T_mat = [data.optimal_costs_reshaped]; % create a matrix containing all targets (2023-11-01)

[~,PS_Value_targets] = mapminmax(data.T_mat,-1,1); % structure containing value function scaling data

%% prepare a feedforward NN for training
numHidden = 30;%60;
net = feedforwardnet(numHidden); % create feedforward NN

net.layers{1}.transferFcn = 'poslin'; %'logsig'; % set hidden layer activation
net.layers{2}.transferFcn = 'tansig'; % set output layer activation

net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'}; % specify processing functions for input one

net.layers{1}.initFcn = 'initnw'; % specify weight initialization algorithm
net.layers{2}.initFcn = 'initnw'; % specify weight initialization algorithm

%% specify initialization method, performance function, and training algorithm
net.initFcn = 'initlay'; % used specified layer-to-layer initialization
net.performFcn = 'mse';
net.trainFcn = 'trainbr'; 

net = configure(net,data.P_mat,data.T_mat); % configure network specifications for the data set

%% specify what training information must be displayed 
net.plotFcns = {'plotperform','plottrainstate'}; % plot performance on 

%% initialize and train NN
net = init(net);
net.trainParam.epochs = 1500;%5000;
% net.trainParam.max_fail = 100;
% net.performParam.regularization = 0.9;
[net,tr] = train(net,data.P_mat,data.T_mat);

%% simulate the trained NN
Y_predictions = sim(net,data.P_mat); 

%% extract the weights and biases of the trained neural network
NN.hidden_layer_parameters(1,:) = net.b{1}';
A = net.IW{1}';
NN.hidden_layer_parameters = [NN.hidden_layer_parameters;A];

%%
NN.output_layer_parameters(1,:) = net.b{2};
B = net.LW{2}';
NN.output_layer_parameters = [NN.output_layer_parameters;B];
% NN.output_layer_parameters(2:(numHidden+1),1) = net.LW{2}';

NN.L = 2; % number of layers after the input layer
NN.k_hidden = numHidden;
NN.a = 1; % set the shape parameter of the sigmoid function equal to one (corresponds with its use in the built-in function)
NN.k_output = 1; % number of output nodes

%% plot the predictions and the targets
plot(data.T_mat(1,:),'k-','LineWidth',2); hold on;
plot(Y_predictions(1,:),'b--','LineWidth',2); 
xlabel('Input pair #'); ylabel('V^*_\pi(S) (-)');
set(gca,'FontSize',25); set(gcf,'Color','white');
legend('Targets','NN approximations');

%% plot linear regressions for training, validation, and testing data
figure
Outputs = net(data.P_mat);
trOut = Outputs(tr.trainInd);
valOut = Outputs(tr.valInd);
testingOut = Outputs(tr.testInd);
trTargets = data.T_mat(tr.trainInd);
valTargets = data.T_mat(tr.valInd);
testingTargets = data.T_mat(tr.testInd);

plotregression(trTargets,trOut,'Train',valTargets,valOut,'Validation',testingTargets,testingOut,'Testing');