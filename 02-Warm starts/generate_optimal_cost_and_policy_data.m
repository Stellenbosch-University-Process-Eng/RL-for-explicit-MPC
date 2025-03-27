%% Script used to generate optimal policy and optimal value function data points across the state space.
%% Date: 2023-10-04
%% Link for MPC toolbox:  https://www.mathworks.com/help/mpc/ref/nlmpc.html
%% Link for creating a wait bar: https://www.mathworks.com/matlabcentral/answers/400243-wait-bar-in-matlab-gui-in-nested-for-loop
clc;clear;close all
rng(1)
tic

%% load matrix of coordinates (2024-02-08)
nmberLevels = 5;
nmberVariables = 6;
myCoordMat = zeros(nmberLevels^nmberVariables,nmberVariables);
totalCntr = 1;
for cntr_1 = 1:1:nmberLevels
    for cntr_2 = 1:1:nmberLevels
        for cntr_3 = 1:1:nmberLevels
            for cntr_4 = 1:1:nmberLevels
                for cntr_5 = 1:1:nmberLevels
                    for cntr_6 = 1:1:nmberLevels
                        myCoordMat(totalCntr,:) = [cntr_1,cntr_2,cntr_3,cntr_4,cntr_5,cntr_6];
                        totalCntr = totalCntr + 1; % increment total counter
                    end
                end
            end
        end
    end
end

%% specifying size of the problem
nx = 4; % states
ny = 4; % output variables
nu = 2; % measured disturbances, MV, unmeasured disturbances
% Inputs to the model include disturbances and MV...
nlobj = nlmpc(nx,ny,nu); % create the non-linear MPC object

%% specifying controller parameters
nlobj.Ts = 1; % set the sample time within the MPC object
nlobj.PredictionHorizon = 300;%10;  % prediction horizon
nlobj.ControlHorizon = 2;     %5;      % number of steps to adjust across the horizon

%% define parameters for model
param.A1 = 28;      % cross-sectional area (cm^2)
param.A3 = param.A1;
param.A2 = 32;
param.A4 = param.A2;
param.a1 = 0.071;   % cross-section of tank outlet (cm^2)
param.a3 = param.a1;
param.a2 = 0.057;
param.a4 = param.a2;
param.g = 981;       % gravitational acceleration (cm/s^2)
param.k1 = 3.14;     % pump 1 gain (cm^3/V)
param.k2 = 3.29;     % pump 2 gain (cm^3/V)
param.gamma1 = 0.43;%0.4;%0.42; % fraction opening pump 1 three-way valve (-)
param.gamma2 = 0.34;%0.4;%0.35; % fraction opening pump 2 three-way valve (-)

%% specifying dynamic model
nlobj.Model.StateFcn = @(x,u,p) myStateFunction(x,u,param);  % specify model used to generate next states
nlobj.Model.NumberOfParameters = 1; % set the number of optional parameters equal to one (the parameter is the structure "p" which contains the model parameters)

%% specify Jacobian for dynamic model
nlobj.Jacobian.StateFcn = @(x,u,p) myStateJacobian(x,u,param); % specify Jacobian for the predictive model

%% specify initial state, SP, and prediction model inputs
v_1_SS = 3.15; % pump voltage (V)
v_2_SS = 3.15; 
h_1_SS_initial_guess = 10.18; % liquid height (cm)
h_2_SS_initial_guess = 15.70;
h_3_SS_initial_guess = 6.05;
h_4_SS_initial_guess = 9.28;

%% simulate the non-linear process model
final_time = 1e6;
tspan = linspace(0,final_time,final_time); % time span (s)
[~,Output] = ode23s(@(t,x) QTProcess_NL_const_time(t,x,param,v_1_SS,v_2_SS),tspan,[h_1_SS_initial_guess,h_2_SS_initial_guess,h_3_SS_initial_guess,h_4_SS_initial_guess]');%,opts);

%% save steady states
h_1_SS = Output(end,1);
h_2_SS = Output(end,2);
h_3_SS = Output(end,3);
h_4_SS = Output(end,4);

x0 = [h_1_SS,h_2_SS,h_3_SS,h_4_SS]'; % nominal states
u0 = [3,3]'; % nominal inputs to the model
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
data.LastMV = u0(1); % last control input
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

% params = []; % no parameters in cost function
param.S_tracking = 1;%nlobj.States(1).Max - nlobj.States(1).Min; % scale factor for SP tracking cost (2023-11-13)
param.S_MV = nlobj.MV(1).Max - nlobj.MV(1).Min; % scale factor for MV adjustment (2023-11-13)
param.S_MV_rate = 1;%nlobj.MV(1).Max - nlobj.MV(1).Min; % scale factor used for rate of MV adjustment (2023-11-13)
param.MV_ref = 0; % reference for MV tracking penalty (2023-11-13)

%% load state scaling structure for cost calculation
% filename_state_scaling_structure = 'C:\Users\Edward\Stellenbosch University\Machine Learning at Process Engineering - Edward Bras (1)\Code\PhD code\12-Article 2\00-Data\03-Runs with warm-started critic\00-Minimum phase region\P_+_state_scaling_data.mat'; % state scaling
% load(filename_state_scaling_structure); % load state scaling data
% param.PS_input = PS_input;

%%
nlobj.Optimization.CustomCostFcn = @(X,U,e,data,params) CostFunction_for_two_states(X,U,e,data,params); % (2023-10-28)
nlobj.Optimization.ReplaceStandardCost = true;

nlobj.Optimization.SolverOptions.Display = "none";%'off'; % "iter";
nlobj.Optimization.SolverOptions.FiniteDifferenceType = 'forward';%'central';

nlobj.Optimization.SolverOptions.Algorithm = 'sqp-legacy';

nlobj.Optimization.SolverOptions.OptimalityTolerance = 1e-1;
nlobj.Optimization.SolverOptions.FunctionTolerance = 1e-1;
nlobj.Optimization.SolverOptions.MaxIterations = 40;

% nlobj.Optimization.SolverOptions.Algorithm = 'active-set';

%% validate the prediction model's functions
validateFcns(nlobj,x0,u0,[],{param}); % validate -> nlobj = object, x0 = starting states, u0 = control inputs, [] = no measured disturbances, Ts = an optional parameter

%% simulate one optimization across the prediction horizon
nloptions = nlmpcmoveopt;
nloptions.Parameters = {param};
[~,~,Info_OL] = nlmpcmove(nlobj,x0,u0,SP,[],nloptions); % nlmpcmove(nlobj,x0,u0(1),SP,u0(2),nloptions);

J_OL_func = @(x_1,x_2) (x_1 - SP(1)).^2 + (x_2 - SP(2)).^2;

for cntr = 1:1:( size(Info_OL.Xopt,1) - 1 )
    J_OL_traj(cntr) = J_OL_func(Info_OL.Xopt(cntr+1,1),Info_OL.Xopt(cntr+1,2));
    X_1_OL_traj(cntr) = Info_OL.Xopt(cntr+1,1);
    X_2_OL_traj(cntr) = Info_OL.Xopt(cntr+1,2);
end

%% simulate closed-loop control under non-linear MPC
simulationTime = 200;  % number of time steps to simulate
x_CL_trajectory = x0';       % initialize liquid height trajectory 
u_CL_trajectory = u0';   % initialize MV trajectory
SP_trajectory = [SP(1),SP(2),0,0];        % initialize SP trajectory

%% number of entries in tspan vector
nmberTspanEntries = 100;

%% Create grid to evaluate optimal solution over
resolution4Grid_except_SP_2 = nmberLevels;%5;    % (2023-11-23)
resolution4Grid_SP_2 = nmberLevels;%5;           % number of SPs for regulation portion of problem (2023-11-23)

% maximum and minimum states expected (states 1 through 4)
minimum_SP_1 = 0;
minimum_SP_2 = 0;
minimum_H_1 = 0;
minimum_H_2 = 0;

maximum_SP_1 = 50;
maximum_SP_2 = 50;
maximum_H_1 = 50;
maximum_H_2 = 50;

% maximum and minimum states expected for liquid levels 3 and 4
% (2023-11-03)
minimum_H_3 = 0;
minimum_H_4 = 0;
maximum_H_3 = 50;
maximum_H_4 = 50;

SP_1_dimension = linspace(minimum_SP_1,maximum_SP_1,resolution4Grid_except_SP_2); % dimension for H1 SP
SP_2_dimension = linspace(minimum_SP_2,maximum_SP_2,resolution4Grid_SP_2); % dimension for H2 SP
H_1_dimension = linspace(minimum_H_1,maximum_H_1,resolution4Grid_except_SP_2);  % dimension for H1 
H_2_dimension = linspace(minimum_H_2,maximum_H_2,resolution4Grid_except_SP_2);  % dimension for H2
H_3_dimension = linspace(minimum_H_3,maximum_H_3,resolution4Grid_except_SP_2); % dimension for H3
H_4_dimension = linspace(minimum_H_4,maximum_H_4,resolution4Grid_except_SP_2);  % dimension for H4

% set scale factors
nlobj.States(1).ScaleFactor = 1;%spSample.setPoint_high - spSample.setPoint_low; 
nlobj.States(2).ScaleFactor = 1;%spSample.setPoint_high - spSample.setPoint_low;
nlobj.States(3).ScaleFactor = 1;%spSample.setPoint_high - spSample.setPoint_low;
nlobj.States(4).ScaleFactor = 1;%spSample.setPoint_high - spSample.setPoint_low;

[SP_1_grid,SP_2_grid,H_1_grid,H_2_grid,H_3_grid,H_4_grid] = ndgrid(SP_1_dimension,SP_2_dimension,H_1_dimension,H_2_dimension,H_3_dimension,H_4_dimension);

x_k = zeros(4,1); % state comprises four liquid heights and two SPs (for liquid heights one and two) (2023-10-31)

%% Determine optimal solution for constant inlet flow rate
nmberOpts = resolution4Grid_SP_2*resolution4Grid_except_SP_2^5; % number of optimizations to be solved (2023-11-16)
cntr_wait = 0; % initialize counter for waitbar (2023-11-16)
% H = waitbar(0,'Please wait...');

%% vectorize nested for-loop using predefined matrix of coordinates (2024-02-08)
size_vec = resolution4Grid_except_SP_2.*ones(1,6);
u_opt_one = zeros(size_vec);
u_opt_two = u_opt_one;
optimalCosts = u_opt_one; % initialize vector that will be used to store optimal costs (2024-04-25)

for cntr = 1:1:resolution4Grid_except_SP_2^6
    row = myCoordMat(cntr,:); % obtain coordinate containing the levels of the different states
    
    %% select SP and DV
    SP(1) = SP_1_grid(row(1),row(2),row(3),row(4),row(5),row(6)); % select SP for liquid height 1
    SP(2) = SP_2_grid(row(1),row(2),row(3),row(4),row(5),row(6)); % select SP for liquid height 2
    x_k(1) = H_1_grid(row(1),row(2),row(3),row(4),row(5),row(6));
    x_k(2) = H_2_grid(row(1),row(2),row(3),row(4),row(5),row(6));
    x_k(3) = H_3_grid(row(1),row(2),row(3),row(4),row(5),row(6));
    x_k(4) = H_4_grid(row(1),row(2),row(3),row(4),row(5),row(6));

    % provide initial control input
    if cntr == 1
        u_k = u0;
    else
        u_k = [Info.MVopt(1,1),Info.MVopt(1,2)]';
    end

    % determine vectors of optimal control inputs
    [~,~,Info] = nlmpcmove(nlobj,x_k,u_k,SP,[],nloptions);  
    u_opt_one(row(1),row(2),row(3),row(4),row(5),row(6)) = Info.MVopt(1,1); % save optimal control input 1
    u_opt_two(row(1),row(2),row(3),row(4),row(5),row(6)) = Info.MVopt(1,2); % save optimal control input 2
    optimalCosts(row(1),row(2),row(3),row(4),row(5),row(6)) = -1*Info.Cost; % save optimal cost (2024_04_25)
    
    % display error message if waitbar is closed:
%     if ~ishghandle(H) % (2023-11-16)
%         error('Waitbar has been closed.');
%     end

    cntr_wait = cntr_wait + 1; % update waitbar counter (2023-11-16)
%     waitbar(cntr_wait/nmberOpts,H,sprintf('Optimization %d of %d',cntr_wait,nmberOpts)); % update waitbar (2023-11-16)
    fprintf('\n%d\n',cntr_wait);

end

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
function dHdt = QTProcess_NL_const_time(t,x,param,v_1,v_2)
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
    A(1,3) = ( param.a1/(2*param.A1) )*sqrt(2*param.g)*x(3)^-0.5;
    Bmv(1,1) = param.gamma1*param.k1/param.A1;

    A(2,2) = -1*( param.a2/(2*param.A2) )*sqrt(2*param.g)*x(2)^-0.5;
    A(2,4) = ( param.a2/(2*param.A2) )*sqrt(2*param.g)*x(4)^-0.5;
    Bmv(2,2) = param.gamma2*param.k2/param.A2;

    A(3,3) = -1*(param.a1/(2*param.A1))*sqrt(2*param.g)*x(3)^-0.5;
    Bmv(3,2) = (1-param.gamma2)*param.k2/param.A1;

    A(4,4) = -1*(param.a2/(2*param.A2))*sqrt(2*param.g)*x(4)^-0.5;
    Bmv(4,1) = (1-param.gamma1)*param.k1/param.A2;

end