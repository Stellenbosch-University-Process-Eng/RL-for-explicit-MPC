%% Code used to submit a script to the SU HPC (parallel computations).
%% Date: 2024-10-16

c = parcluster(); % initialize cluster using the default profile
c.AdditionalProperties.Host = 'comp044'; 
ncpu = 16; % number of cpu cores

% submit job to the SU cluster
test_NLMPC_qudruple_tank = batch(c,'generate_optimal_cost_and_policy_data','Pool',ncpu,'CaptureDiary',true);