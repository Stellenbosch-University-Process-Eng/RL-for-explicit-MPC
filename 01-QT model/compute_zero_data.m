%% Code used to determine zero locations and input and output zero vectors
%% of the linearised quadruple tank model for specified valve position.
%% Name: Edward Bras
%% Date: 2025-10-04

%%
clc;clear;%close all;

%% define valve positions
param.gamma1 = 0.45;
param.gamma2 = param.gamma1;

if param.gamma1 < 0.5 % if non-minimum phase region
    pos_zero_flag = 1; % flag that index of positive zero must be found
else
    pos_zero_flag = 0;
end

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
param.k1 = 3.14;     % pump 1 gain (cm^3/V)
param.k2 = 3.29;     % pump 2 gain (cm^3/V)

%% determine initial model steady
v_1_SS = 3; % pump 1 voltage (V)
v_2_SS = 3; % pump 2 voltage (V)
h_1_SS_initial_guess = 10.18; % liquid height (cm)
h_2_SS_initial_guess = 15.70;
h_3_SS_initial_guess = 6.05;
h_4_SS_initial_guess = 9.28;

% find model steady state
final_time = 1e6;
tspan = linspace(0,final_time,final_time); % time span (s)
[~,Output] = ode23s(@(t,x) QTProcess_NL_no_handles(t,x,param,v_1_SS,v_2_SS),tspan,[h_1_SS_initial_guess,h_2_SS_initial_guess,h_3_SS_initial_guess,h_4_SS_initial_guess]');%,opts);

% extract steady states from simulation output
h_1_SS = Output(end,1);
h_2_SS = Output(end,2);
h_3_SS = Output(end,3);
h_4_SS = Output(end,4);

%% define linear model's parameters
% control action and controlled variable scaling factors
lin_param.v1_Fact = 30;
lin_param.v2_Fact = lin_param.v1_Fact;
lin_param.h1_Fact = 50;
lin_param.h2_Fact = lin_param.h1_Fact;

% gains and time constants
lin_param.K1 = sqrt(2*param.g*h_1_SS)/sqrt(2*param.g*h_3_SS);
lin_param.K2 = param.gamma1*param.k1*sqrt(2*param.g*h_1_SS)/( param.a1*param.g );
lin_param.K3 = sqrt(2*param.g*h_2_SS)/sqrt(2*param.g*h_4_SS);
lin_param.K4 = param.gamma2*param.k2*sqrt(2*param.g*h_2_SS)/( param.a2*param.g );
lin_param.K5 = param.k2*( 1 - param.gamma2 )*sqrt( 2*param.g*h_3_SS )/( param.a3*param.g );
lin_param.K6 = param.k1*( 1 - param.gamma1 )*sqrt(2*param.g*h_4_SS)/( param.a4*param.g );

lin_param.tau_h1 = param.A1*sqrt(2*param.g*h_1_SS)/( param.a1*param.g );
lin_param.tau_h2 = param.A2*sqrt(2*param.g*h_2_SS)/( param.a2*param.g );
lin_param.tau_h3 = param.A1*sqrt(2*param.g*h_3_SS)/( param.a1*param.g );
lin_param.tau_h4 = param.A2*sqrt(2*param.g*h_4_SS)/( param.a2*param.g );

%% define transfer function
s = tf('s');
G_s(1,1) = (lin_param.v1_Fact/lin_param.h1_Fact)*(lin_param.K2/( lin_param.tau_h1*s + 1 ));
G_s(1,2) = (lin_param.v2_Fact/lin_param.h1_Fact)*(lin_param.K1*lin_param.K5)/( (lin_param.tau_h1*s + 1)*(lin_param.tau_h3*s + 1) );
G_s(2,1) = (lin_param.v1_Fact/lin_param.h2_Fact)*(lin_param.K3*lin_param.K6)/( (lin_param.tau_h2*s + 1)*(lin_param.tau_h4*s + 1) );
G_s(2,2) = (lin_param.v2_Fact/lin_param.h2_Fact)*(lin_param.K4/( lin_param.tau_h2*s + 1 ));

% %% find zero and pole locations
zero_locs = tzero(G_s); % second zero is the one that can move to the right-half plane
pole_locs = pole(G_s);

%% test for observability
sys_ss = ss(G_s);    % Convert tf to state-space
sys_min = minreal(ss(sys_ss));
Ob = obsv(sys_min.A, sys_min.C);
isObservable = rank(Ob) == size(sys_min.A,1);

%% find tranmission zeros and poles
min_zero_locs = tzero(sys_min);
min_pole_locs = pole(sys_min);

% zero to evaluate
if pos_zero_flag == 1 % if in non-minimum phase region
    zero_to_eval = find(min_zero_locs > 0);
else
    zero_to_eval = 1;
end

%% test for controllability
% Controllability test
Co = ctrb(sys_min.A, sys_min.B);
is_controllable = rank(Co) == size(sys_min.A,1);

%% determine input and output directions of the second zero
G_s_11 = @(x) (lin_param.v1_Fact/lin_param.h1_Fact)*(lin_param.K2/( lin_param.tau_h1*x + 1 ));
G_s_12 = @(x) (lin_param.v2_Fact/lin_param.h1_Fact)*(lin_param.K1*lin_param.K5)/( (lin_param.tau_h1*x + 1)*(lin_param.tau_h3*x + 1) );
G_s_21 = @(x) (lin_param.v1_Fact/lin_param.h2_Fact)*(lin_param.K3*lin_param.K6)/( (lin_param.tau_h2*x + 1)*(lin_param.tau_h4*x + 1) );
G_s_22 = @(x) (lin_param.v2_Fact/lin_param.h2_Fact)*(lin_param.K4/( lin_param.tau_h2*x + 1 ));

G_s_at_z = [G_s_11(min_zero_locs(zero_to_eval)), G_s_12(min_zero_locs(zero_to_eval)); G_s_21(min_zero_locs(zero_to_eval)), G_s_22(min_zero_locs(zero_to_eval))];

[U,S,V] = svd(G_s_at_z);

%% figures
myFontSize = 20;
myMarkerSize = 20;

subplot(1,3,1)
plot(min_zero_locs(zero_to_eval),0,'kx','LineWidth',3,'MarkerSize',myMarkerSize); hold on;
xlabel('Re (-)'), ylabel('Im (-)');
set(gca,'FontSize',myFontSize,'FontName','Times New Roman');
% axis equal
yline(0,'k--','LineWidth',2);
xline(0,'k--','LineWidth',2);
legend('zero location');
xlim([0,1.5]);
ylim([-0.1,0.1]);
% set(gca, 'XScale', 'log')

subplot(1,3,2)
% add vector direction
z = [0,0];
quiver(z(1), z(2), U(1,end), U(2,end), 0, 'LineWidth', 2, 'MaxHeadSize', 0.5,'Color','k'); hold on;
axis equal
xlabel('H_1 (-)'), ylabel('H_2 (-)');
xlim([-1,1]);
ylim([-1,1]);
yline(0,'k--','LineWidth',2);
xline(0,'k--','LineWidth',2);
set(gca,'FontSize',myFontSize,'FontName','Times New Roman');
set(gcf,'Color','w');

% visualise step response of +1 in v1 and -1 in v2
subplot(1,3,3)
t = 0:0.01:1000;  % simulation from 0 to 10 seconds with 0.01 s step
u = [ones(length(t),1), -ones(length(t),1)];
y = lsim(sys_ss, u, t);  % y will be [length(t) x 2] corresponding to outputs
plot(t, y(:,1), 'Color',[44,162,95]/255,'LineWidth',3); hold on;
plot(t, y(:,2), 'Color',[43,140,190]/255, 'LineWidth', 3);
xlabel('Time (s)');
ylabel('Scaled liquid height');
grid on;
set(gca,'FontSize',myFontSize,'FontName','Times New Roman');
set(gcf,'Color','w'); 
hold on;
yline(0,'k--','LineWidth',2);
legend('H_1 (-)','H_2 (-)');