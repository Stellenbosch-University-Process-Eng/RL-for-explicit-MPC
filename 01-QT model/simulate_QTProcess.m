%% Code used to simulate the non-linear quadruple tank benchmark model in open loop.
% Date: 2025-03-26
% Name: Edward Bras

%%
clc;clear;close all;

%% define parameters for model
param.A1 = 28;  % cross-sectional area (cm^2)
param.A3 = param.A1; 
param.A2 = 32;
param.A4 = param.A2;
param.a1 = 0.071; % cross-section of tank outlet (cm^2)
param.a3 = param.a1;
param.a2 = 0.057;
param.a4 = param.a2;
param.g = 981;       % gravitational acceleration (cm/s^2)
param.k1 = 3.33;     % pump 1 gain (cm^3/V)
param.k2 = 3.35;     % pump 2 gain (cm^3/V)
param.gamma1 = @(t) 0.02 + 0*t; % fraction opening pump 1 three-way valve (-)
param.gamma2 = @(t) 0.02 + 0*t; % fraction opening pump 2 three-way valve (-)

%% specify initial state, SP, and prediction model inputs
v_1_SS = @(t) 3 + 0*t; % pump 1 voltage (V)
v_2_SS = @(t) 3 + 0*t; % pump 2 voltage (V)
h_1_SS_initial_guess = 10.18; % liquid height (cm)
h_2_SS_initial_guess = 15.70;
h_3_SS_initial_guess = 6.05;
h_4_SS_initial_guess = 9.28;

%% find model steady states
final_time = 1e6;
tspan = linspace(0,final_time,final_time); % time span (s)
[~,Output] = ode23s(@(t,x) QTProcess_NL(t,x,param,v_1_SS,v_2_SS),tspan,[h_1_SS_initial_guess,h_2_SS_initial_guess,h_3_SS_initial_guess,h_4_SS_initial_guess]');%,opts);

% extract steady states from simulation output
h_1_SS = Output(end,1);
h_2_SS = Output(end,2);
h_3_SS = Output(end,3);
h_4_SS = Output(end,4);

%% simulate non-linear model
final_time_dyn = 1e3;
res_dyn = 1000;
tspan_dyn = linspace(0,final_time_dyn,res_dyn);
v_1_dyn = @(t) 3 + 1*(t>10); % pump 1 voltage (V)
v_2_dyn = @(t) 3 - 1*(t>10); % pump 2 voltage (V)

[t_dyn,Output_dyn] = ode23s(@(t,x) QTProcess_NL(t,x,param,v_1_dyn,v_2_dyn),tspan_dyn,[h_1_SS,h_2_SS,h_3_SS,h_4_SS]');%,opts);

%% visualise simulation outputs
myFontSize = 12;

subplot(4,2,1);
plot(t_dyn,Output_dyn(:,1),'b-','LineWidth',2);
xlabel('t (s)'); ylabel('H_1 (cm)');
set(gca,'FontSize',myFontSize);

subplot(4,2,2);
plot(t_dyn,Output_dyn(:,2),'b-','LineWidth',2);
xlabel('t (s)'); ylabel('H_2 (cm)');
set(gca,'FontSize',myFontSize);

subplot(4,2,3);
plot(t_dyn,Output_dyn(:,3),'b-','LineWidth',2);
xlabel('t (s)'); ylabel('H_3 (cm)');
set(gca,'FontSize',myFontSize);

subplot(4,2,4);
plot(t_dyn,Output_dyn(:,4),'b-','LineWidth',2);
xlabel('t (s)'); ylabel('H_4 (cm)');
set(gca,'FontSize',myFontSize);

subplot(4,2,5);
plot(t_dyn,param.gamma1(t_dyn),'b-','LineWidth',2);
xlabel('t (s)'); ylabel('\gamma_1 (-)');
set(gca,'FontSize',myFontSize);

subplot(4,2,6);
plot(t_dyn,param.gamma2(t_dyn),'b-','LineWidth',2);
xlabel('t (s)'); ylabel('\gamma_2 (-)');
set(gca,'FontSize',myFontSize);

subplot(4,2,7);
plot(t_dyn,v_1_dyn(t_dyn),'b-','LineWidth',2);
xlabel('t (s)'); ylabel('v_1 (V)');
set(gca,'FontSize',myFontSize);

subplot(4,2,8);
plot(t_dyn,v_2_dyn(t_dyn),'b-','LineWidth',2);
xlabel('t (s)'); ylabel('v_2 (V)');
set(gca,'FontSize',myFontSize);

set(gcf,'Color','w');