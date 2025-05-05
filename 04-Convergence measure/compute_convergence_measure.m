%% Code used to evaluate the sum of squared differences between the MPC actions
%% and the RL actions w.r.t. the respective controlled liquid heights (H1 and H2).
%% Name: Edward Bras
%% Date: 2025-05-05
clc;clear;

tic 

%% load saved data
load('Exp_many_scenarios.mat');      % load all measured states during RL training
load('Policies_many_scenarios.mat'); % load policies generated during RL training
load('par_many_scenarios.mat');      % load parameters (including state- and action scaling structures)

%% determine parameters required to process stored training data
numProbs = size(all_scenarios_out_Experience,1); % number of control problems
num_k = size(all_scenarios_out_Experience{1,1}(1).State_1,2); % number of discrete time steps
num_RL_policies = size(all_scenarios_out_Policies{1,1}(1).NN,2); % number of policies saved
num_steps_skip = num_k/num_RL_policies; % number of time steps between successive policies
policy_times_vec = num_steps_skip:num_steps_skip:num_k; % vector of time steps for which policies

numStates = 6;  % number of observed RL states (SP1,SP2,H1,H2,H3,H4)
numActions = 2; % number of actions applied to the RL environment

%% specifying size of the problem
nx = 4; % states
ny = 4; % output variables
nu = 2; % measured disturbances, MV, unmeasured disturbances
% Inputs to the model include disturbances and MV...
nlobj = nlmpc(nx,ny,nu); % create the non-linear MPC object

%% specifying controller parameters
nlobj.Ts = 1;                           % set the sample time within the MPC object
nlobj.PredictionHorizon = 10;          % prediction horizon
nlobj.ControlHorizon = 5;               % number of steps to adjust across the horizon

%% define parameters for model
param.A1 = 28;      % cross-sectional area (cm^2)
param.A3 = param.A1;
param.A2 = 28;
param.A4 = param.A2;
param.a1 = 0.071;   % cross-section of tank outlet (cm^2)
param.a3 = param.a1;
param.a2 = 0.071;
param.a4 = param.a2;
param.g = 981;       % gravitational acceleration (cm/s^2)
param.k1 = 2;     % pump 1 gain (cm^3/V)
param.k2 = 2;     % pump 2 gain (cm^3/V)
param.gamma1 = 0.6; % fraction opening pump 1 three-way valve (-)
param.gamma2 = 0.6; % fraction opening pump 2 three-way valve (-)

%% specifying dynamic model
% https://www.mathworks.com/help/mpc/ug/specify-prediction-model-for-nonlinear-mpc.html
nlobj.Model.StateFcn = @(x,u,p) myStateFunction(x,u,param);  % specify model used to generate next states
nlobj.Model.NumberOfParameters = 1; % set the number of optional parameters equal to one (the parameter is the structure "p" which contains the model parameters)

%% specify Jacobian for dynamic model
% https://www.mathworks.com/help/mpc/ug/specify-prediction-model-for-nonlinear-mpc.html#mw_6eb5a593-c403-47f7-b2d8-fed832e17a61
nlobj.Jacobian.StateFcn = @(x,u,p) myStateJacobian(x,u,param); % specify Jacobian for the predictive model

%% specify initial state, SP, and prediction model inputs
v_1_SS = 3.15; % pump voltage (V)
v_2_SS = 3.15; 
h_1_SS_initial_guess = 10.18; % liquid height (cm)
h_2_SS_initial_guess = 15.70;
h_3_SS_initial_guess = 6.05;
h_4_SS_initial_guess = 9.28;

%% find model steady state
final_time = 1e6;
tspan = linspace(0,final_time,final_time); % time span (s)
[~,Output] = ode23s(@(t,x) QTProcess_NL_solve_SS(t,x,param,v_1_SS,v_2_SS),tspan,[h_1_SS_initial_guess,h_2_SS_initial_guess,h_3_SS_initial_guess,h_4_SS_initial_guess]');

% extract steady states from simulation output
h_1_SS = Output(end,1);
h_2_SS = Output(end,2);
h_3_SS = Output(end,3);
h_4_SS = Output(end,4);

x0 = [h_1_SS,h_2_SS,h_3_SS,h_4_SS]'; % nominal states
u0 = [v_1_SS,v_2_SS]'; % nominal inputs to the model
SP = [h_1_SS,h_2_SS,0,0]; % set point

%% linear constraints on the MVs
for cntr = 1:1:nu
    nlobj.MV(cntr).Min = 0.1;
    nlobj.MV(cntr).Max = 30;
end

nlobj.ManipulatedVariables(1).MaxECR = 0;

%% set constraints on the measured output
for cntr = 1:1:nx
    nlobj.States(cntr).Min = 0.005;
    nlobj.States(cntr).Max = 100;
end

%% specify a custom cost function
param.Q = [1,0;0,1];   % weighting matrix for cost function (2023-10-28)
param.R = [0,0;0,0];   % weighting matrix for control input cost (2023-11-10)
param.MVr = [0,0;0,0]; % weighting matrix for rate of control input adjustments (2023-11-13)

%% https://www.mathworks.com/help/mpc/ug/specify-cost-function-for-nonlinear-mpc.html
data.Ts = nlobj.Ts; % sampling period
data.CurrentStates = x0; % current states
data.LastMV = u0; % last control input
data.References = SP; % set point
data.MVTarget = []; 
data.PredictionHorizon = nlobj.PredictionHorizon;
data.NumOfStates = nlobj.Dimensions.NumberOfStates;
data.NumOfOutputs = nlobj.Dimensions.NumberOfOutputs;
data.NumOfInputs = nlobj.Dimensions.NumberOfInputs;
data.MVIndex = nlobj.Dimensions.MVIndex;
data.MDIndex = nlobj.Dimensions.MDIndex;
data.UDIndex = nlobj.Dimensions.UDIndex;

e = 0;       % only hard constraints are applicable 
% See https://www.mathworks.com/help/mpc/ug/specifying-constraints.html for
% information and tips on constraint softening and the tuning of Equal
% Concern for Relaxation (ECR) values.

% also see https://www.mathworks.com/help/mpc/ug/optimization-problem.html
% for discussion of default MPC cost functions

param.S_tracking = 1; % scale factor for SP tracking cost (2023-11-13)
param.S_MV = nlobj.MV(1).Max - nlobj.MV(1).Min; % scale factor for MV adjustment (2023-11-13)
param.S_MV_rate = 1;%nlobj.MV(1).Max - nlobj.MV(1).Min; % scale factor used for rate of MV adjustment (2023-11-13)
param.MV_ref = 0; % reference for MV tracking penalty (2023-11-13)

%% https://www.mathworks.com/help/optim/ug/tolerances-and-stopping-criteria.html
nlobj.Optimization.CustomCostFcn = @(X,U,e,data,params) CostFunction_for_two_states(X,U,e,data,params); % (2023-10-28)
nlobj.Optimization.ReplaceStandardCost = true;

nlobj.Optimization.SolverOptions.Display = "none";%'off'; % "iter";
nlobj.Optimization.SolverOptions.FiniteDifferenceType = 'forward';%'central';

nlobj.Optimization.SolverOptions.Algorithm = 'sqp-legacy';

nlobj.Optimization.SolverOptions.OptimalityTolerance = 1e-1;
nlobj.Optimization.SolverOptions.FunctionTolerance = 1e-1;
nlobj.Optimization.SolverOptions.MaxIterations = 40;

%% validate the prediction model's functions
validateFcns(nlobj,x0,u0,[],{param}); % validate -> nlobj = object, x0 = starting states, u0 = control inputs, [] = no measured disturbances, Ts = an optional parameter

%% simulate one optimization across the prediction horizon
nloptions = nlmpcmoveopt;
nloptions.Parameters = {param};
[~,~,Info_OL] = nlmpcmove(nlobj,x0,u0,SP,[],nloptions); 

J_OL_func = @(x_1,x_2) (x_1 - SP(1)).^2 + (x_2 - SP(2)).^2;

for cntr = 1:1:( size(Info_OL.Xopt,1) - 1 )
    J_OL_traj(cntr) = J_OL_func(Info_OL.Xopt(cntr+1,1),Info_OL.Xopt(cntr+1,2));
    X_1_OL_traj(cntr) = Info_OL.Xopt(cntr+1,1);
    X_2_OL_traj(cntr) = Info_OL.Xopt(cntr+1,2);
end

%% compute r.r
states_array = nan(numStates,numProbs,num_RL_policies); % initialise array containing all states for which RL policies are available
MPC_actions_array = nan(numProbs,num_RL_policies,numActions); % initialise array containing all MPC actions
RL_actions_array = MPC_actions_array; % initialise array containing all RL agent actions

% populate array containing the coordinates in state space corresponding to
% the observations made during RL agent training for each of the saved
% actor networks.
for probCntr = 1:1:numProbs
    states_array(1,probCntr,:) = all_scenarios_out_Experience{probCntr,1}(end).State_1(policy_times_vec);
    states_array(2,probCntr,:) = all_scenarios_out_Experience{probCntr,1}(end).State_2(policy_times_vec);
    states_array(3,probCntr,:) = all_scenarios_out_Experience{probCntr,1}(end).State_3(policy_times_vec);
    states_array(4,probCntr,:) = all_scenarios_out_Experience{probCntr,1}(end).State_4(policy_times_vec);
    states_array(5,probCntr,:) = all_scenarios_out_Experience{probCntr,1}(end).State_5(policy_times_vec);
    states_array(6,probCntr,:) = all_scenarios_out_Experience{probCntr,1}(end).State_6(policy_times_vec);

end % end loop through number of problems

% determine MPC and RL actions
for probCntr = 1:1:numProbs
    for numPolCntr = 1:1:num_RL_policies
        SP(1) = states_array(1,probCntr,numPolCntr);
        SP(2) = states_array(2,probCntr,numPolCntr);

        x_k(1) = states_array(3,probCntr,numPolCntr);
        x_k(2) = states_array(4,probCntr,numPolCntr);
        x_k(3) = states_array(5,probCntr,numPolCntr);
        x_k(4) = states_array(6,probCntr,numPolCntr);

        % provide initial control input
        if numPolCntr == 1
            u_k = u0;
        else
            u_k = [Info.MVopt(1,1),Info.MVopt(1,2)]';
        end

        % compute MPC action
        [~,~,Info] = nlmpcmove(nlobj,x_k,u_k,SP,[],nloptions);
        MPC_actions_array(probCntr,numPolCntr,1) = Info.MVopt(1,1); % control action 1
        MPC_actions_array(probCntr,numPolCntr,2) = Info.MVopt(1,2); % control action 2

        % calculate RL action
        % scale states
        PS_input = all_scenarios_p_outputs{probCntr,1}(end).PS_input;
        [InputsScaled,~] = mapminmax('apply',[SP(1),SP(2),x_k(1),x_k(2),x_k(3),x_k(4)]',PS_input); % scale inputs

        % evaluate neural network (2023-11-26)
        NN = all_scenarios_out_Policies{probCntr,1}(end).NN{1,numPolCntr};
        [u_opt_1_s,u_opt_2_s,~,~] = evaluate_ReLU_tanh_six_states(NN,InputsScaled(1),InputsScaled(2),InputsScaled(3),InputsScaled(4),InputsScaled(5),InputsScaled(6));
    
        % scale control inputs back to actual numerical values (2023-11-26)
        PS_targets = all_scenarios_p_outputs{probCntr,1}(end).PS_targets;
        [true_Us,~] = mapminmax('reverse',[u_opt_1_s,u_opt_2_s]',PS_targets);

        RL_actions_array(probCntr,numPolCntr,1) = true_Us(1);
        RL_actions_array(probCntr,numPolCntr,2) = true_Us(2);

    end % end loop through policies

    fprintf('\n Problem number: %d\n',probCntr);

end % end loop through problems

% chatGPT consulted for the code (up until before functions)
r = (MPC_actions_array - RL_actions_array).^2;
rdotr.all_data = sum(r,3);
rdotr.mean_vals = squeeze(mean(rdotr.all_data,1))';
rdotr.std_vals = squeeze(std(rdotr.all_data,0,1))';

rdotr.summary_data = [rdotr.mean_vals,...
    rdotr.mean_vals + rdotr.std_vals,...
    rdotr.mean_vals - rdotr.std_vals];

%% plots (chatGPT)
% Assuming summary_stats is 50x3:
mean_vals = rdotr.summary_data(:, 1);
upper_bound = rdotr.summary_data(:, 2);
lower_bound = rdotr.summary_data(:, 3);

x = policy_times_vec;  

figure;
hold on;

% Plot the mean line
plot_1 = plot(x, mean_vals, 'k:o', 'LineWidth', 2);

% Fill the area between upper and lower bounds (shaded region)
plot_2 = fill([x, fliplr(x)], [upper_bound', fliplr(lower_bound')], ...
        [0 0 0], 'EdgeColor', 'none', 'FaceAlpha', 0.1);  

% reference line corresponding to a residual of zero
plot_3 = yline(0,'k--','LineWidth',1);

xlabel('k (s)');
ylabel('r_\pi (-)');
legend([plot_1,plot_2],{'mean','±1 std dev'});
set(gca,'FontSize',25);
set(gcf,'Color','w');

toc 

%% functions
% cost function for two-dimensional states (2023-10-21)
function J = CostFunction_for_two_states(X,U,e,data,params)
    N = data.PredictionHorizon;
    X_1 = X(2:N+1,1); % state 1
    X_2 = X(2:N+1,2); % state 2
    Refs_1 = data.References(1:N,1);
    Refs_2 = data.References(1:N,2);

    U_1 = U(1:N,1); % control input 1 (2023-11-13)
    U_2 = U(1:N,2); % control input 2 (2023-11-13)

    U_1_shifted = U(2:N+1,1); % shifted control input 1 (2023-11-13)
    U_2_shifted = U(2:N+1,2); % shifted control input 2 (2023-11-13)
    
    Refs_U_1 = params.MV_ref*ones(size(Refs_1)); % reference values for control input 1 (2023-11-10)
    Refs_U_2 = params.MV_ref*ones(size(Refs_2)); % reference values for control input 2 (2023-11-10)

    SP_tracking_cost = ( params.Q(1)/params.S_tracking )*(X_1 - Refs_1).^2 + ...
        ( ( params.Q(2)+params.Q(3) )/params.S_tracking ).*(X_1 - Refs_1).*(X_2 - Refs_2) +...
        ( params.Q(4)/params.S_tracking )*(X_2 - Refs_2).^2;

    MV_tracking_cost = ( params.R(1)/params.S_MV )*(U_1 - Refs_U_1).^2 + ...
        ( ( params.R(2) + params.R(3) )/params.S_MV ).*(U_1-Refs_U_1).*(U_2 - Refs_U_2) +...
        ( params.R(4)/params.S_MV )*(U_2-Refs_U_2).^2;

    MV_rate_cost = ( params.MVr(1)/params.S_MV_rate )*(U_1_shifted - U_1).^2 + ...
        ( ( params.MVr(2) + params.MVr(3) )/params.S_MV_rate ).*(U_1_shifted - U_1).*(U_2_shifted - U_2) +...
        ( params.MVr(4)/params.S_MV_rate )*(U_2_shifted - U_2).^2;

    J = sum(  SP_tracking_cost + MV_tracking_cost + MV_rate_cost ) ... 
        + e;

end

% mixing tank model for call to MPC optimization routine
function z = myStateFunction(x,u,param)
    % Function that contains the differential equations for the non-linear
    % Quadruple Tank benchmark process.
    % x(1) = h1
    % x(2) = h2
    % x(3) = h3
    % x(4) = h4
    dh1dt = ( -1*param.a1*sqrt(2*param.g*x(1)) + param.a3*sqrt(2*param.g*x(3)) + param.gamma1*param.k1*u(1) )/param.A1;
    dh2dt = ( -1*param.a2*sqrt(2*param.g*x(2)) + param.a4*sqrt(2*param.g*x(4)) + param.gamma2*param.k2*u(2) )/param.A2;
    dh3dt = ( -1*param.a3*sqrt(2*param.g*x(3)) + (1-param.gamma2)*param.k2*u(2) )/param.A3;
    dh4dt = ( -1*param.a4*sqrt(2*param.g*x(4)) + (1-param.gamma1)*param.k1*u(1) )/param.A4;
    
    z = [dh1dt,dh2dt,dh3dt,dh4dt]';

end

%% non-linear dynamic model used to calculate initial steady-state liquid heights
function dHdt = QTProcess_NL_solve_SS(t,x,param,v_1,v_2)
% Function that contains the differential equations for the non-linear
% Quadruple Tank benchmark process.
% x(1) = h1
% x(2) = h2
% x(3) = h3
% x(4) = h4
dh1dt = ( -1*param.a1*sqrt(2*param.g*x(1)) + param.a3*sqrt(2*param.g*x(3)) + param.gamma1*param.k1*v_1 )/param.A1;
dh2dt = ( -1*param.a2*sqrt(2*param.g*x(2)) + param.a4*sqrt(2*param.g*x(4)) + param.gamma2*param.k2*v_2 )/param.A2;
dh3dt = ( -1*param.a3*sqrt(2*param.g*x(3)) + (1-param.gamma2)*param.k2*v_2 )/param.A3;
dh4dt = ( -1*param.a4*sqrt(2*param.g*x(4)) + (1-param.gamma1)*param.k1*v_1 )/param.A4;

dHdt = [dh1dt,dh2dt,dh3dt,dh4dt]';

end

% Specify analytical Jacobian for predictive model
function [A,Bmv] = myStateJacobian(x,u,param)
    A(1,1) = -1*( param.a1/(2*param.A1) )*sqrt(2*param.g)*x(1)^-0.5;
    A(1,3) = ( param.a3/(2*param.A1) )*sqrt(2*param.g)*x(3)^-0.5;
    Bmv(1,1) = param.gamma1*param.k1/param.A1;

    A(2,2) = -1*( param.a2/(2*param.A2) )*sqrt(2*param.g)*x(2)^-0.5;
    A(2,4) = ( param.a4/(2*param.A2) )*sqrt(2*param.g)*x(4)^-0.5;
    Bmv(2,2) = param.gamma2*param.k2/param.A2;

    A(3,3) = -1*(param.a3/(2*param.A3))*sqrt(2*param.g)*x(3)^-0.5;
    Bmv(3,2) = (1-param.gamma2)*param.k2/param.A3;

    A(4,4) = -1*(param.a4/(2*param.A4))*sqrt(2*param.g)*x(4)^-0.5;
    Bmv(4,1) = (1-param.gamma1)*param.k1/param.A4;

end