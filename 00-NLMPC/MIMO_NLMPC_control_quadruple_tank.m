%% Script used to control liquid levels one and two in the quadruple tank benchmark
%% using NMPC.
%% Date: 2023-12-04
%% Name: Edward Bras
clc;clearvars -except ans;
rng(1)

tic 

%% specifying size of the problem
nx = 4; % states
ny = 4; % output variables
nu = 2; % measured disturbances, MV, unmeasured disturbances
% Inputs to the model include disturbances and MV...
nlobj = nlmpc(nx,ny,nu); % create the non-linear MPC object

%% specifying controller parameters
nlobj.Ts = 1;                   % set the sample time within the MPC object
nlobj.PredictionHorizon = 10;  % prediction horizon
nlobj.ControlHorizon = 3;       % number of steps to adjust across the horizon

%% set valve positions
param.gamma1 = 0.42;     % valve position 1
param.gamma2 = 0.42;     % valve position 2

%% define parameters for model
param.A1 = 1; % cross-sectional area (cm^2)
param.A3 = param.A1;
param.A2 = 1;
param.A4 = param.A2;
param.a1 = 0.071; % cross-section of tank outlet (cm^2)
param.a3 = param.a1;
param.a2 = 0.071;
param.a4 = param.a2;
param.g = 981;   % gravitational acceleration (cm/s^2)
param.k1 = 3.33; % pump 1 gain (cm^3/V)
param.k2 = 3.33; % pump 2 gain (cm^3/V)
param.observedGamma1 = param.gamma1;  % fraction valve 1 opening "seen" by the MPC (2023-12-04)
param.observedGamma2 = param.gamma2;  % fraction valve 2 opening "seen" by the MPC (2023-12-04)

%% specifying dynamic model
% https://www.mathworks.com/help/mpc/ug/specify-prediction-model-for-nonlinear-mpc.html
nlobj.Model.StateFcn = @(x,u,p) myStateFunction(x,u,param);  % specify model used to generate next states
nlobj.Model.NumberOfParameters = 1; % set the number of optional parameters equal to one (the parameter is the structure "p" which contains the model parameters)

%% specify Jacobian for dynamic model
% https://www.mathworks.com/help/mpc/ug/specify-prediction-model-for-nonlinear-mpc.html#mw_6eb5a593-c403-47f7-b2d8-fed832e17a61
nlobj.Jacobian.StateFcn = @(x,u,p) myStateJacobian(x,u,param); % specify Jacobian for the predictive model

%% specify initial state, SP, and prediction model inputs
v_1_SS = 3; % pump voltage (V)
v_2_SS = 3; 
h_1_SS_initial_guess = 10.18; % liquid height (cm)
h_2_SS_initial_guess = 15.70;
h_3_SS_initial_guess = 6.05;
h_4_SS_initial_guess = 9.28;

%% find model steady state
final_time = 1e6;
tspan = linspace(0,final_time,final_time); % time span (s)
[~,Output] = ode23s(@(t,x) QTProcess_NL_const_time(t,x,param,v_1_SS,v_2_SS),tspan,[h_1_SS_initial_guess,h_2_SS_initial_guess,h_3_SS_initial_guess,h_4_SS_initial_guess]');

% extract steady states from simulation output
h_1_SS = Output(end,1);
h_2_SS = Output(end,2);
h_3_SS = Output(end,3);
h_4_SS = Output(end,4);

x0 = [h_1_SS,h_2_SS,h_3_SS,h_4_SS]'; % nominal states
u0 = [v_1_SS,v_2_SS]';               % nominal inputs to the model
SP = [h_1_SS,h_2_SS,0,0];            % set points

%% linear constraints on the MVs
for cntr = 1:1:nu
    nlobj.MV(cntr).Min = 0.1;
    nlobj.MV(cntr).Max = 30;
end

nlobj.ManipulatedVariables(1).MaxECR = 0; % constraints on MV are hard constraints

% %% set constraints on the measured output
% for cntr = 1:1:nx
%     nlobj.States(cntr).Min = 0.005;%0.1;%1.3; 
%     nlobj.States(cntr).Max = 100;%20;
% end

%% specify a custom cost function
param.Q = [1,0;0,1];    % weighting matrix for cost function (2023-10-28)
param.R = [0,0;0,0];    % weighting matrix for control input cost (2023-11-10)
param.MVr = [0,0;0,0];  % weighting matrix for rate of control input adjustments (2023-11-13)

%% https://www.mathworks.com/help/mpc/ug/specify-cost-function-for-nonlinear-mpc.html
data.Ts = nlobj.Ts;                 % sampling period
data.CurrentStates = x0; % current states
data.LastMV = u0;     % last control input
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

param.S_tracking = 1;                           % scale factor for SP tracking cost (2023-11-13)
param.S_MV = nlobj.MV(1).Max - nlobj.MV(1).Min; % scale factor for MV adjustment (2023-11-13)
param.S_MV_rate = 1;                            % scale factor used for rate of MV adjustment (2023-11-13)
param.MV_ref = 0; % reference for MV tracking penalty (2023-11-13)

%% https://www.mathworks.com/help/optim/ug/tolerances-and-stopping-criteria.html
nlobj.Optimization.CustomCostFcn = @(X,U,e,data,params) CostFunction_for_two_states(X,U,e,data,params); % (2023-10-28)
nlobj.Optimization.ReplaceStandardCost = true;

nlobj.Optimization.SolverOptions.Display = "none";%"iter";%'off'; % "iter";
nlobj.Optimization.SolverOptions.FiniteDifferenceType = 'forward';%'central';

nlobj.Optimization.SolverOptions.Algorithm = 'sqp-legacy';

% nlobj.Optimization.SolverOptions.OptimalityTolerance = 1e0;
% nlobj.Optimization.SolverOptions.UseParallel = true;
% nlobj.Optimization.SolverOptions.ConstraintTolerance = 1e-1;
nlobj.Optimization.SolverOptions.OptimalityTolerance = 1e-1;
nlobj.Optimization.SolverOptions.FunctionTolerance = 1e-1;
nlobj.Optimization.SolverOptions.MaxIterations = 40;

% For details on cost function Jacobians, see https://www.mathworks.com/help/mpc/ug/specify-cost-function-for-nonlinear-mpc.html#mw_ec61fbdc-3879-4dae-8569-b48ba271111e

% %% specify Jacobian for cost function
% nlobj.Jacobian.CustomCostFcn = @(X,U,e,data,params) myCostJacobian(X,U,e,data,params);

%% validate the prediction model's functions
validateFcns(nlobj,x0,u0,[],{param}); % validate -> nlobj = object, x0 = starting states, u0 = control inputs, [] = no measured disturbances, Ts = an optional parameter

%% simulate one optimization across the prediction horizon
nloptions = nlmpcmoveopt;
nloptions.Parameters = {param};
[~,~,Info_OL] = nlmpcmove(nlobj,x0,u0,SP,[],nloptions); 

%% simulate closed-loop control under non-linear MPC
simulationTime = 1000;    % number of time steps to simulate
x_CL_trajectory = x0';   % initialize liquid height trajectory 
u_CL_trajectory = u0';   % initialize MV trajectory
SP_trajectory = [SP(1),SP(2),0,0];        % initialize SP trajectory

% see https://www.mathworks.com/help/mpc/ref/mpc.mpcmove.html for
% discussion on obtaining the optimal costs computed using MPC (across the
% prediction horison)
Cost_trajectory = 0; % initialize variable for optimal costs

%% parameters used for SP sampling
% SP 1
spSample.setPoint_low = 10; 
spSample.setPoint_high = 20; 
spSample.nmberTimes = 10;
spSample.stepLim = 1; % step size constraint incorporated on 2024-12-04

% SP 2
spSample_2.setPoint_low = 10;  
spSample_2.setPoint_high = 20; 
spSample_2.nmberTimes = 10;
spSample_2.stepLim = 1; % step size constraint incorporated on 2024-12-04

% sample SPs
spSample.sampleTimes = randi([1,simulationTime],1,spSample.nmberTimes);
spSample = quadruple_tank_step_constrained_SP_sampling(spSample,SP,1);

% sample SP 2 (2023-11-26)
spSample_2.sampleTimes = randi([1,simulationTime],1,spSample_2.nmberTimes);
spSample_2 = quadruple_tank_step_constrained_SP_sampling(spSample_2,SP,2);

%% number of entries in tspan vector
nmberTspanEntries = 100;

% set scale factors (2023-10-31)
nlobj.States(1).ScaleFactor = 1;%spSample.setPoint_high - spSample.setPoint_low; 
nlobj.States(2).ScaleFactor = 1;%spSample.setPoint_high - spSample.setPoint_low;
nlobj.States(3).ScaleFactor = 1;%spSample.setPoint_high - spSample.setPoint_low;
nlobj.States(4).ScaleFactor = 1;%spSample.setPoint_high - spSample.setPoint_low;

for currentTimeStamp = 1:1:(simulationTime/nlobj.Ts)

    %% sampling of SP and DV
            %% assign sampled SP changes
            % sample H1 SP
            for i = 1:1:size(spSample.sampleTimes,2)
                if spSample.sampleTimes(i) == currentTimeStamp
                    SP(1) = spSample.SP_samples(i);
                end

            end
            % sample H2 SP 
            for j = 1:1:size(spSample_2.sampleTimes,2)
                if spSample_2.sampleTimes(j) == currentTimeStamp
                    SP(2) = spSample_2.SP_samples(j); % (2023-11-17)
                end

            end

            %% select valve positions (2023-11-09)
            if exist('param.gamma_vec','var')
            if currentTimeStamp <= size(param.gamma_vec,2)
                param.gamma1 = param.gamma_vec(currentTimeStamp); % change valve position 1 
                param.gamma2 = param.gamma1; % set valve position 2 equal to valve position 1
            end
            else
                % valve positions have been set independently
            end
            %%
    x_k = x_CL_trajectory(currentTimeStamp,:);
    u_k = u_CL_trajectory(currentTimeStamp,:);

    % determine vector of optimal control inputs
    [~,~,Info] = nlmpcmove(nlobj,x_k,u_k,SP,[],nloptions);
    % implement first control move from optimized trajectory
    tspan = linspace(currentTimeStamp*nlobj.Ts,(currentTimeStamp + 1)*nlobj.Ts,nmberTspanEntries);

    [~,Output] = ode23s(@(t,x) myQTPDEs(t,x,param,Info.MVopt(1,:)),tspan,x_k');
    x_kPlus1 = Output(end,:);

    % extend saved trajectories
    x_CL_trajectory = [x_CL_trajectory;x_kPlus1];
    u_CL_trajectory = [u_CL_trajectory;Info.MVopt(1,:)];
    SP_trajectory = [SP_trajectory;SP];
    Cost_trajectory = [Cost_trajectory,-1*Info.Cost]; % expand record of optimal costs (summed over prediction horison)
    fprintf('\n %d \n',currentTimeStamp);

end

%% update trajectories to incorporate zero-order holds
u_Reconstruct_CL_trajectory = repelem(u_CL_trajectory,nlobj.Ts,1);     % control input
SP_Reconstruct_trajectory = repelem(SP_trajectory,nlobj.Ts,1);         % SP

x_Reconstruct_CL_trajectory = x_CL_trajectory(1,:); % state initialization

for cntr_Reconstruct = 1:1:simulationTime
    x_k(:) = x_Reconstruct_CL_trajectory(cntr_Reconstruct,:);

    u_k = u_Reconstruct_CL_trajectory(cntr_Reconstruct,:);     % current control input
    SP_k = SP_Reconstruct_trajectory(cntr_Reconstruct,:);      % current SP
    tspan = linspace(cntr_Reconstruct*nlobj.Ts,(cntr_Reconstruct+1)*nlobj.Ts,nmberTspanEntries);
    [~,Output_reconstruct] = ode45(@(t,x) myQTPDEs(t,x,param,u_k),tspan,x_k');
    x_kPlus1_Reconstruct = Output_reconstruct(end,:);
    % extend saved trajectories
    x_Reconstruct_CL_trajectory = [x_Reconstruct_CL_trajectory;x_kPlus1_Reconstruct];
    
end

%% save relevant results in a structure
NLMPC_Outputs.x_CL_SIMoutput_trajectory = x_Reconstruct_CL_trajectory;
NLMPC_Outputs.u_CL_SIMoutput_trajectory = u_Reconstruct_CL_trajectory;
NLMPC_Outputs.SP_SIMoutput_trajectory = SP_Reconstruct_trajectory;
NLMPC_Outputs.Cost_SIMoutput_trajectory = Cost_trajectory;

% % save data
% filename = '/scratch3/20068530/NLMPC_Outputs.mat';
% save(filename,'NLMPC_Outputs',"-v7.3");

%% plots
subplot(2,1,1)
plot(x_Reconstruct_CL_trajectory(:,1),'b-','LineWidth',2); hold on; 
plot(x_Reconstruct_CL_trajectory(:,2),'k:','LineWidth',2);
plot(x_Reconstruct_CL_trajectory(:,3),'m-','LineWidth',2);
plot(x_Reconstruct_CL_trajectory(:,4),'r-','LineWidth',2);

plot(SP_Reconstruct_trajectory(:,1),'b--','LineWidth',2);
plot(SP_Reconstruct_trajectory(:,2),'k--','LineWidth',2);
xlabel('Time (s)'); ylabel('H (cm)'); axis tight;
set(gca,'fontsize',25); legend('H_1 (cm)','H_2 (cm)','H_3 (cm)','H_4 (cm)','SP_1 (cm)','SP_2 (cm)');

subplot(2,1,2)
plot(u_Reconstruct_CL_trajectory(:,1),'k-','LineWidth',2); hold on;
plot(u_Reconstruct_CL_trajectory(:,2),'k--','LineWidth',2);
xlabel('Time (s)'); ylabel('u (-)'); axis tight;
legend('u_1 (V)','u_2 (V)');
set(gca,'fontsize',25); 

set(gcf,'Color','white');

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

% Cost function Jacobian
function [G,Gmv,Ge] = myCostJacobian(X,U,e,data,param)
    N = data.PredictionHorizon;
    X_1 = X(2:N+1,1); % state 1
    X_2 = X(2:N+1,2); % state 2
    Refs_1 = data.References(1:N,1);
    Refs_2 = data.References(1:N,2);
    Nx = data.NumOfStates;
    Nu = data.NumOfInputs;

    G = zeros(N,Nx); % initialize Jacobian
    G(1:N,1) = -2.*(Refs_1 - X_1);
    G(1:N,2) = -2.*(Refs_2 - X_2);
    G(1:N,3) = 0;
    G(1:N,4) = 0;

    Gmv = zeros(N,Nu); % Jacobian w.r.t. the manipulated variables
    Ge = 0;  % Jacobian w.r.t. slack variables
end

% mixing tank model for call to MPC optimization routine
function z = myStateFunction(x,u,param)
    % Function that contains the differential equations for the non-linear
    % Quadruple Tank benchmark process.
    % x(1) = h1
    % x(2) = h2
    % x(3) = h3
    % x(4) = h4
    dh1dt = ( -1*param.a1*sqrt(2*param.g*x(1)) + param.a3*sqrt(2*param.g*x(3)) + param.observedGamma1*param.k1*u(1) )/param.A1;
    dh2dt = ( -1*param.a2*sqrt(2*param.g*x(2)) + param.a4*sqrt(2*param.g*x(4)) + param.observedGamma2*param.k2*u(2) )/param.A2;
    dh3dt = ( -1*param.a3*sqrt(2*param.g*x(3)) + (1-param.observedGamma2)*param.k2*u(2) )/param.A3;
    dh4dt = ( -1*param.a4*sqrt(2*param.g*x(4)) + (1-param.observedGamma1)*param.k1*u(1) )/param.A4;
    
    z = [dh1dt,dh2dt,dh3dt,dh4dt]';

end

% Specify analytical Jacobian for predictive model
function [A,Bmv] = myStateJacobian(x,u,param)
    A(1,1) = -1*( param.a1/(2*param.A1) )*sqrt(2*param.g)*x(1)^-0.5;
    A(1,3) = ( param.a3/(2*param.A1) )*sqrt(2*param.g)*x(3)^-0.5;
    Bmv(1,1) = param.gamma1*param.k1/param.A1;

    A(2,2) = -1*( param.a2/(2*param.A2) )*sqrt(2*param.g)*x(2)^-0.5;
    A(2,4) = ( param.a4/(2*param.A2) )*sqrt(2*param.g)*x(4)^-0.5;
    Bmv(2,2) = param.gamma2*param.k2/param.A2;

    A(3,3) = -1*(param.a3/(2*param.A1))*sqrt(2*param.g)*x(3)^-0.5;
    Bmv(3,2) = (1-param.gamma2)*param.k2/param.A3;

    A(4,4) = -1*(param.a4/(2*param.A2))*sqrt(2*param.g)*x(4)^-0.5;
    Bmv(4,1) = (1-param.gamma1)*param.k1/param.A4;

end

% mixing tank model for call to DE solver
function dHdt = myQTPDEs(t,x,param,u)
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
    
    dHdt = [dh1dt,dh2dt,dh3dt,dh4dt]';

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