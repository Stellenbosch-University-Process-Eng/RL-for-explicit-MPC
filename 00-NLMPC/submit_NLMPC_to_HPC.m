%% Code used to submit a script to the SU HPC (parallel computations).
%% Date: 2024-10-16

c = parcluster(); % initialize cluster using the default profile
c.AdditionalProperties.Host = 'comp030'; 
ncpu = 16; % number of cpu cores

% submit job to the SU cluster
test_NLMPC_qudruple_tank = batch(c,'MIMO_NLMPC_control_quadruple_tank','Pool',ncpu,'CaptureDiary',true);