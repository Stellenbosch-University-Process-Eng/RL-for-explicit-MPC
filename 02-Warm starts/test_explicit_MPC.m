%% Script that simulates closed-loop MIMO control using the explicit MPC generated
%% in the script "fit_ff_net_MIMO_data.m".
%% Name: Edward Bras
%% Date: 2023-11-24

clc;clearvars -except ans;%close all;
rng(1)%rng(2)

tic 

%% specify path to warm-starting policy
filename_warm_policy = "C:\Users\Edward\Stellenbosch University\Machine Learning at Process Engineering - Edward Bras (1)\Code\PhD code\12-Article 2\00-Data\03-Runs with warm-started critic\01-Non-minimum phase region\P_-_NN_Policy.mat";
filename_warm_net_obj = "C:\Users\Edward\Stellenbosch University\Machine Learning at Process Engineering - Edward Bras (1)\Code\PhD code\12-Article 2\00-Data\03-Runs with warm-started critic\01-Non-minimum phase region\P_-_OBJ_Policy.mat";

%% set decay rate for valve positions
param.gamma1 = 0.43;%0.95; % valve 1 fraction opening
param.gamma2 = 0.34;%0.95; % valve 2 fraction opening

%% load warm-starting policy
load(filename_warm_policy);
NN.k_hidden_warm = NN.k_hidden; % no cold-starting node addition
NN.activation_type = 1; % set output activation to linear
NN.linearActM = 1;    

load(filename_warm_net_obj);

%% define parameters for model
param.A1 = 28;       % cross-sectional area (cm^2)
param.A3 = param.A1;
param.A2 = 32;
param.A4 = param.A2;
param.a1 = 0.071;   % cross-section of tank outlet (cm^2)
param.a3 = param.a1;
param.a2 = 0.057;
param.a4 = param.a2;
% param.kc = 0.50; % gain for level measurement signal (V/cm)
param.g = 981;   % gravitational acceleration (cm/s^2)
param.k1 = 3.14;%3.33;%2;  % pump 1 gain (cm^3/V)
param.k2 = 3.29;%3.35;%2; % pump 2 gain (cm^3/V)

%% specify initial state, SP, and prediction model inputs
v_1_SS = 3.15;
v_2_SS = 3.15;
h_1_SS_initial_guess = 10.18; % liquid height (cm)
h_2_SS_initial_guess = 15.70;
h_3_SS_initial_guess = 6.05;
h_4_SS_initial_guess = 9.28;

%% simulate the non-linear process model
final_time = 1e6;
tspan = linspace(0,final_time,final_time); % time span (s)
[~,Output] = ode23s(@(t,x) QTProcess_NL_solve_SS(t,x,param,v_1_SS,v_2_SS),tspan,[h_1_SS_initial_guess,h_2_SS_initial_guess,h_3_SS_initial_guess,h_4_SS_initial_guess]');%,opts);

%% save steady states
h_1_SS = Output(end,1);
h_2_SS = Output(end,2);
h_3_SS = Output(end,3);
h_4_SS = Output(end,4);

x0 = [h_1_SS,h_2_SS,h_3_SS,h_4_SS]'; % nominal states
u0 = [v_1_SS,v_2_SS]'; % nominal inputs to the model
SP = [h_1_SS,h_2_SS,0,0]; % set point

%% simulate closed-loop control under non-linear MPC
simulationTime = 50000;%10000;  % number of time steps to simulate
x_CL_trajectory = x0';       % initialize liquid height trajectory 
u_CL_trajectory = u0';   % initialize MV trajectory
SP_trajectory = SP;%[SP(1),SP(2),0,0];        % initialize SP trajectory

%% set lower control input limit (2023-11-09)
Limits.u_lower = 0.1;%0.0001; % minimum pump voltage (V)
Limits.u_upper = 30;     % maximum pump voltage (V)

%% parameters used for SP sampling
% SP 1
spSample.setPoint_low = 20;
spSample.setPoint_high = 30;
spSample.nmberTimes = 10;%20;

% SP 2
spSample.setPoint_low_2 = 20; 
spSample.setPoint_high_2 = 30; 
spSample.nmberTimes_2 = 10;%20;

% sample SPs
spSample.sampleTimes = randi([1,simulationTime],1,spSample.nmberTimes);
spSample.SP_samples = spSample.setPoint_low + (spSample.setPoint_high - spSample.setPoint_low_2)*rand(1,spSample.nmberTimes);

% sample SP 2 (2023-11-26)
spSample.sampleTimes_2 = randi([1,simulationTime],1,spSample.nmberTimes_2);
spSample.SP_samples_2 = spSample.setPoint_low_2 + (spSample.setPoint_high_2 - spSample.setPoint_low_2)*rand(1,spSample.nmberTimes_2);

%% number of entries in tspan vector
nmberTspanEntries = 100;

Ts = 1; % sampling period

%% specify paths for preprocessing- and postprocessing data
filename_state_scaling_structure = 'C:\Users\Edward\Stellenbosch University\Machine Learning at Process Engineering - Edward Bras (1)\Code\PhD code\12-Article 2\00-Data\03-Runs with warm-started critic\00-Minimum phase region\P_+_state_scaling_data.mat'; % state scaling
filename_control_input_scaling_structure = 'C:\Users\Edward\Stellenbosch University\Machine Learning at Process Engineering - Edward Bras (1)\Code\PhD code\12-Article 2\00-Data\03-Runs with warm-started critic\00-Minimum phase region\P_+_action_scaling_data.mat'; % control input scaling 

%% load preprocessing- and postprocessing data
load(filename_state_scaling_structure); % load state scaling data
load(filename_control_input_scaling_structure); % load control input scaling data

for currentTimeStamp = 1:1:(simulationTime/Ts)

    %% sampling of SP and DV
            %% assign sampled SP changes
            % sample H1 SP
            for i = 1:1:size(spSample.sampleTimes,2)
                if spSample.sampleTimes(i) == currentTimeStamp
                    SP(1) = spSample.SP_samples(i);
%                     SP(2) = spSample.SP_samples_2(i); % (2023-11-16)
                end
            end
            % sample H2 SP (2023-11-17)
            for j = 1:1:size(spSample.sampleTimes_2,2)
                if spSample.sampleTimes_2(j) == currentTimeStamp
                    SP(2) = spSample.SP_samples_2(j); % (2023-11-17)
                end
            end
    x_k = x_CL_trajectory(currentTimeStamp,:);
    u_k = u_CL_trajectory(currentTimeStamp,:);

    %% select valve positions (2023-11-09)
    if exist('param.gamma_vec','var')
        if currentTimeStamp <= size(param.gamma_vec,2)
            param.gamma1 = param.gamma_vec(currentTimeStamp); % change valve position 1 
            param.gamma2 = param.gamma1; % set valve position 2 equal to valve position 1
        end
    else
        % valve positions have been set independently
    end
    % scale states
    [InputsScaled,~] = mapminmax('apply',[SP(1),SP(2),x_k(1),x_k(2),x_k(3),x_k(4)]',PS_input); % scale inputs

    % evaluate neural network (2023-11-26)
    [u_opt_1_s,u_opt_2_s,~,~] = evaluate_ReLU_tanh_six_states(NN,InputsScaled(1),InputsScaled(2),InputsScaled(3),InputsScaled(4),InputsScaled(5),InputsScaled(6));

    % scale control inputs back to actual numerical values (2023-11-26)
    [true_Us,~] = mapminmax('reverse',[u_opt_1_s,u_opt_2_s]',PS_targets);
    u_opt_1 = true_Us(1); % scale control input one to true numerical value
    u_opt_2 = true_Us(2); % scale control input two to true numerical value

%     % use built-in evaluation of NN object to enact control (2023-11-04)
%     U_approx = sim(net,[SP(1),SP(2),x_k(1),x_k(2),x_k(3),x_k(4)]');
%     % scale control inputs back to actual numerical values (2023-11-26)
%     u_opt_1 = U_approx(1);
%     u_opt_2 = U_approx(2);
    
    % saturate the control inputs if required (2023-11-03)
    if u_opt_1 < Limits.u_lower
        u_opt_1 = Limits.u_lower;
    end
    if u_opt_2 < Limits.u_lower
        u_opt_2 = Limits.u_lower;
    end

    if u_opt_1 > Limits.u_upper
        u_opt_1 = Limits.u_upper;
    end
    if u_opt_2 > Limits.u_upper
        u_opt_2 = Limits.u_upper;
    end
    
    % implement first control move from optimized trajectory
    tspan = linspace(currentTimeStamp,currentTimeStamp + 1*Ts,nmberTspanEntries);
%     [~,Output] = ode45(@(t,x) myDEmodel(t,x,u0(2),p,Info.MVopt(1)),tspan,x_k );
    [~,Output] = ode23s(@(t,x) myQTPDEs(t,x,param,[u_opt_1,u_opt_2]),tspan,x_k');
    x_kPlus1 = Output(end,:);
    % extend saved trajectories
    x_CL_trajectory = [x_CL_trajectory;x_kPlus1];
    u_CL_trajectory = [u_CL_trajectory;[u_opt_1,u_opt_2]];
    SP_trajectory = [SP_trajectory;SP];
    fprintf('\n %d \n',currentTimeStamp);

end

%% plots
subplot(2,2,1)
plot(x_CL_trajectory(:,1),'b-','LineWidth',2); hold on; 
plot(x_CL_trajectory(:,2),'k:','LineWidth',2);
plot(x_CL_trajectory(:,3),'m-','LineWidth',2);
plot(x_CL_trajectory(:,4),'r-','LineWidth',2);

plot(SP_trajectory(:,1),'b--','LineWidth',2);
plot(SP_trajectory(:,2),'k--','LineWidth',2);
xlabel('Time (min)'); ylabel('H (cm)'); axis tight;
set(gca,'fontsize',25); legend('H_1 (cm)','H_2 (cm)','H_3 (cm)','H_4 (cm)','SP_1 (cm)','SP_2 (cm)');

subplot(2,2,2)
plot(u_CL_trajectory(:,1),'k-','LineWidth',2); hold on;
plot(u_CL_trajectory(:,2),'k--','LineWidth',2);
xlabel('Time (min)'); ylabel('u (V)'); axis tight;
legend('u_1 (V)','u_2 (V)');
set(gca,'fontsize',25); 

subplot(2,2,3)
plot(2.*param.gamma_vec,'k-','LineWidth',3);
yline(1,'Color',[0.6510,0.6510,0.6510],'LineWidth',3); xline(simulationTime,'Color',[0.6510,0.6510,0.6510],'LineWidth',2,'LineStyle',':');
ylim([0,2]);
ylabel('2*\gamma'); xlabel('Time (s)');

set(gcf,'Color','white');

toc

%% functions
% QTP model for call to DE solver
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