%% Code used to determine zero locations and input and output zero vectors
%% of the linearised quadruple tank model for specified valve position.
%% Name: Edward Bras
%% Date: 2025-10-04

%%
clc;clear;%close all;

%% define valve positions
num_gammas = 100;    % number of valve position for which to determine zero locations
init_gamma = 0.01;  % initial valve fraction opening
fin_gamma = 0.99;   % final valve fraction opening
gamma_vals = linspace(init_gamma,fin_gamma,num_gammas); % valve fractions

step_response_cell = cell(1,num_gammas); % cell array used to store all timeseries data

for cntr = 1:1:num_gammas

param.gamma1 = gamma_vals(cntr);
param.gamma2 = param.gamma1;

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
min_zero_locs(:,cntr) = tzero(sys_min);
min_pole_locs(:,cntr) = pole(sys_min);

%% compute step responses
t = 0:0.01:1000;  % simulation from 0 to 10 seconds with 0.01 s step
u = [ones(length(t),1), -ones(length(t),1)];
y = lsim(sys_ss, u, t);  % y will be [length(t) x 2] corresponding to outputs

step_response_cell{cntr} = y; % store step response in two liquid heights corresponding to the current valve position

fprintf('\nvalve fraction # = %d\n',cntr);

end % end loop over valve positions

Sorted_min_zero_locs  = sort_rows_smoothest_output(min_zero_locs); % arrange zero positions for the two respective rows

%% figures
myLabelFontSize = 20;
myAxisNumberFontSize = 20;
myMarkerSize = 15;
myLineWidth = 3;

tls = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile
plot(gamma_vals,Sorted_min_zero_locs(2,:),'k:','LineWidth',myLineWidth,'MarkerSize',myMarkerSize); hold on;
hold on;
plot(gamma_vals,Sorted_min_zero_locs(1,:),'k:','LineWidth',myLineWidth,'MarkerSize',myMarkerSize); hold on;
set(gca,'FontSize',myAxisNumberFontSize,'FontName','Times New Roman');

lgnd_1 = legend('z_1','z_2');
lgnd_1.Location = "best";
lgnd_1.FontSize = myLabelFontSize;

% xlim([0.55,0.95]);
% set(gca,'xdir','reverse'); % show valve positions decreasing
axis tight

xlbl = xlabel('\gamma (-)'); 
ylbl = ylabel('z (-)');

xlbl.FontSize = myLabelFontSize;
ylbl.FontSize = myLabelFontSize;

% visualise step response of +1 in v1 and -1 in v2
nexttile

for step_plot_cntr = 1:1:num_gammas
    plot(t,step_response_cell{step_plot_cntr}(:,1),'Color',[44,162,95]/255,'LineWidth',myLineWidth); hold on;
    plot(t,step_response_cell{step_plot_cntr}(:,2),'Color',[43,140,190]/255,'LineWidth',myLineWidth); hold on;

end % end loop through valve positions

xlbl_2 = xlabel('Time (s)');
ylbl_2 = ylabel('Scaled liquid height');
grid on;
set(gca,'FontSize',myAxisNumberFontSize,'FontName','Times New Roman');
set(gcf,'Color','w'); 
hold on;
yline(0,'k--','LineWidth',2);
lgnd_2 = legend('H_1 (-)','H_2 (-)');
% lgnd_2.Location = "best";
lgnd_2.FontSize = myLabelFontSize;

grid off;

xlbl_2.FontSize = myLabelFontSize;
ylbl_2.FontSize = myLabelFontSize;

%% functions
function M_new = sort_rows_smoothest_output(M_original)
    M_new = M_original(:,1);
    for col = 2:1:size(M_original,2)
        crnt_col = M_original(:,col);

        % compare distances of two possible orders
        keep_dist = sum(abs(crnt_col - M_new(:,end)));
        swap_dist = sum(abs(flip(crnt_col) - M_new(:,end)));

        if swap_dist < keep_dist
            M_new = [M_new,flip(crnt_col)];
        else
            M_new = [M_new,crnt_col];
        end

    end % end loop across columns

end % end index sorting function