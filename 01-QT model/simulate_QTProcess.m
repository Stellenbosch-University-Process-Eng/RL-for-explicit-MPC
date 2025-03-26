%% Code used to simulate the non-linear quadruple tank benchmark model in open loop.
% Date: 2025-03-26
% Name: Edward Bras

%% define parameters for model
param.A1 = 28;  % cross-sectional area (cm^2)
param.A3 = param.A1; 
param.A2 = 28;
param.A4 = param.A2;
param.a1 = 0.071; % cross-section of tank outlet (cm^2)
param.a3 = param.a1;
param.a2 = 0.071;
param.a4 = param.a2;
param.g = 981;       % gravitational acceleration (cm/s^2)
param.k1 = 3.14;%2;%3.33; %1;    % pump 1 gain (cm^3/V)
param.k2 = 3.29;%2;%3.33;     % pump 2 gain (cm^3/V)
param.gamma1 = 0.43;%param.initial_gammas; % fraction opening pump 1 three-way valve (-)
param.gamma2 = 0.34;%param.initial_gammas; % fraction opening pump 2 three-way valve (-)

%% specify initial state, SP, and prediction model inputs
v_1_SS = 3.15;%3; %-(3-miniumSetting)*(t>3000); % pump voltage (V)
v_2_SS = 3.15;%3; %-(3-miniumSetting)*(t>3000);
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