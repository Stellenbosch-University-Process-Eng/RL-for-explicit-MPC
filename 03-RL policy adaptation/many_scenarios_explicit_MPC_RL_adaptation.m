%% Script that uses a warm-starting state-value function estimated by fitting 
%% a neural network to optimal cost data calculated using MPC.
%% Name: Edward Bras
%% Date: 2024-04-26
clc
clearvars -except ans; close all;
rng(2)

%%
tic % start measuring wall time

%% start parallel pool, 2022-10-17
myPool = parpool('local'); 

%% load policy and value function data
load('policy_data_gamma_001_numlevels_5.mat');
load('value_data_gamma_001_numlevels_5.mat');

% allocate variables using prompts to prevent user prompts from appearing
% at the start of each training scenario
nmberStepsSpecified = 20000;
upperReportVec = nmberStepsSpecified;
upperPolicySavingVec = nmberStepsSpecified;
numberScenarios = 2;
for scenarioCntr = 1:1:numberScenarios
    fprintf('\n%d\n\n',scenarioCntr); % display the scenario number
    %% set decay rate for valve positions
    gammaSpecs.gamma_vec_start = 0;
    gammaSpecs.gamma_vec_end = nmberStepsSpecified;
    gammaSpecs.gamma_final = 0.1;
    gammaSpecs.initial_gammas = 0.01; % (2023-11-09)
    
    gammaSpecs.rateRepeat = 1; % how many subsequent time steps should the valve position be maintained 
    
    t_vec = linspace(gammaSpecs.gamma_vec_start,gammaSpecs.gamma_vec_end,gammaSpecs.gamma_vec_end);   
    param.gamma_vec = ((gammaSpecs.gamma_final-gammaSpecs.initial_gammas)/(nmberStepsSpecified))*t_vec + gammaSpecs.initial_gammas; 
    param.gamma_vec = repelem(param.gamma_vec,gammaSpecs.rateRepeat);
    
    %% saturation penalty settings (2023-05-03)
    param.includeSaturationPenalty = 0; % if = 1, agent penalized for saturating the final element.  if = 0, agent is not penalized (2023-05-03)
    param.satPenaltyMagnitude = 0.2;    % magnitude of penalty for saturating the final element (2023-05-03)
    
    %% load warm-starting policy (2023-06-14)
    actor.NN = policy_data.NN; % 2023-02-08
    actor.NN.linearActM = 1;                    % gradient of linear output activation function (2023-01-17)
    actor.NN.stdDev_defined = 0.05;%0.06;              % standard deviation used when sampling exploratory actions (2023-01-17)
    actor.NN.activation_type = 1;               % activation type of output layer set as linear (2023-01-17)
    actor.NN.k_hidden_warm = actor.NN.k_hidden; % define "actor.NN.k_hidden_warm" for use in network evaluation function (2023-10-10)
    
    %% load critic network
    critic.NN = value_data.NN;
    critic.NN.alpha = 0.5; % critic learning rate
    
    %% load preprocessing- and postprocessing data and store data processing 
    %% structures as fields of the parameter structure
    param.PS_input = policy_data.PS_input; 
    param.PS_targets = policy_data.PS_targets;
    param.PS_Value_targets = value_data.PS_Value_targets;
    
    %% initialize average reward and relevant learning rate (2023-06-29)
    param.avgRAlpha = 0.5;     % learning rate used to update the average reward (2023-06-29)
    param.avgR = 0;            % initialize average reward (2023-06-29)
    
    %% set interval between reported step numbers (2023-06-30)
    param.reportingVec = 1000:1000:upperReportVec;      % relevant data are saved at these time steps of training
    
    %% Gaussian noise added to liquid level measurement (2023-08-13)
    param.noiseToMeasurements = 0; % flag used to indicate whether noise should be added to CV measurements (1 = add noise to measurements, 0 = don't add noise to measurements) (2023-08-13)
    param.measurementStDev = 0.01; % standard deviation used to generate Gaussian measurement noise (2023-08-13)
    
    %% define parameters for model
    param.A1 = 28;  % cross-sectional area (cm^2)
    param.A3 = param.A1; 
    param.A2 = 32;
    param.A4 = param.A2;
    param.a1 = 0.071; % cross-section of tank outlet (cm^2)
    param.a3 = param.a1;
    param.a2 = 0.057;
    param.a4 = param.a2;
    param.g = 981;          % gravitational acceleration (cm/s^2)
    param.k1 = 3.14;        % pump 1 gain (cm^3/V)
    param.k2 = 3.29;        % pump 2 gain (cm^3/V)
    param.gamma1 = 0.1;    % fraction opening pump 1 three-way valve (-)
    param.gamma2 = 0.1;    % fraction opening pump 2 three-way valve (-)
    
    %% specify initial state, SP, and prediction model inputs
    v_1_SS = 3.15;          % pump voltage (V)
    v_2_SS = 3.15;
    h_1_SS_initial_guess = 10.18; % liquid height (cm)
    h_2_SS_initial_guess = 15.70;
    h_3_SS_initial_guess = 6.05;
    h_4_SS_initial_guess = 9.28;
    
    %% simulate the non-linear process model
    final_time = 1e6;
    tspan = linspace(0,final_time,final_time); % time span (s)
    [~,Output] = ode23s(@(t,x) myQTDEs(t,x,param,[v_1_SS,v_2_SS]),tspan,[h_1_SS_initial_guess,h_2_SS_initial_guess,h_3_SS_initial_guess,h_4_SS_initial_guess]');%,opts);
    
    %% save steady states
    h_1_SS = Output(end,1);
    h_2_SS = Output(end,2);
    h_3_SS = Output(end,3);
    h_4_SS = Output(end,4);
    
    param.x0 = [h_1_SS,h_2_SS,h_3_SS,h_4_SS]'; % nominal states
    param.u0 = [v_1_SS,v_2_SS]'; % nominal inputs to the model
    param.SP = [h_1_SS,h_2_SS];% set points (2023-11-17)
    
    %% RL agent training settings
    param.sampling_period = 1;              % process time associated with transition from T to (T + 1)
    param.nmberOfSteps = nmberStepsSpecified;   % number of steps per episode
    
    %% bounds for reward scaling
    R_bounds.low = 0;                % low bound for squared error reward function, 2022-10-14
    R_bounds.high = 1;               % high bound for squared error reward function, 2022-10-14
    
    %% parameters used for SP sampling
    % SP 1
    spSample.setPoint_low = 20; 
    spSample.setPoint_high = 30; 
    spSample.nmberTimes = 6;
    spSample.stepLim = 10; % step size constraint incorporated on 2024-12-04
    
    % SP 2
    spSample_2.setPoint_low = 20;  
    spSample_2.setPoint_high = 30; 
    spSample_2.nmberTimes = 6;
    spSample_2.stepLim = 10; % step size constraint incorporated on 2024-12-04
    
    % sample SPs
    spSample.sampleTimes = randi([1,nmberStepsSpecified],1,spSample.nmberTimes);
    spSample = quadruple_tank_step_constrained_SP_sampling(spSample,[param.x0(1),param.x0(2)],1);
    
    % sample SP 2 (2023-11-26)
    spSample_2.sampleTimes = randi([1,nmberStepsSpecified],1,spSample_2.nmberTimes);
    spSample_2 = quadruple_tank_step_constrained_SP_sampling(spSample_2,[param.x0(1),param.x0(2)],2);

    %% initialize counters and levels for the step size parameter (2022-10-17)
    StepSizes.nmberOfStepSizeLevels = 10;    % number of step size levels to consider, 2022-10-17
    StepSizes.low_stepSizeBound = 0;        % lower bound for step sizes, 2022-10-17
    StepSizes.high_stepSizeBound = 3.33e-4; % upper bound for step sizes, 2022-10-17
    StepSizes.hyperparameterValues = linspace(StepSizes.low_stepSizeBound,StepSizes.high_stepSizeBound,StepSizes.nmberOfStepSizeLevels); % step sizes at which to conduct training, 2022-10-17
    
    % ranges for pump voltages (2022-11-05).  These are hard constraints.
    param.lower_u_1 = 0.1;
    param.lower_u_2 = 0.1;
    
    param.upper_u_1 = 30;
    param.upper_u_2 = 30;
    
    %% specify the number of output nodes for the actor
    actor.NN.k_output = 2; % set the number of output nodes for the actor equal to two (2023-11-30)
    
    %% specify time steps for which actor networks must be saved
    param.policySavingVec = 0:1000:upperPolicySavingVec; 
    param.saveCntr = 1; % counter used to index saved actor networks
    
    %% define eligibility trace vectors
    param.actor_lambda = 0;     % actor trace-decay parameter
    param.critic_lambda = 0;    % critic trace-decay parameter
    
    % actor
    param.trace_actor_hidden = zeros(size(actor.NN.hidden_layer_parameters,1),size(actor.NN.hidden_layer_parameters,2)); % eligibility trace for hidden layer parameters
    param.trace_actor_output = zeros(size(actor.NN.output_layer_parameters,1),size(actor.NN.output_layer_parameters,2)); % eligibility trace for output layer parameters
    
    % critic
    param.trace_critic_hidden = zeros(size(critic.NN.hidden_layer_parameters,1),size(critic.NN.hidden_layer_parameters,2)); % eligibility trace for hidden layer parameters
    param.trace_critic_output = zeros(size(critic.NN.output_layer_parameters,1),size(critic.NN.output_layer_parameters,2)); % eligibility trace for output layer parameters

    %% train the agents
    parfor hyperparameterCntr = 1:StepSizes.nmberOfStepSizeLevels
        [example_agentExperience,example_Policy,example_traces,example_Critic] = trainFunction(actor,critic,...
            param,R_bounds,spSample,spSample_2,...
            StepSizes.hyperparameterValues(hyperparameterCntr));
        out_Policies(hyperparameterCntr) = example_Policy;
        out_Critics(hyperparameterCntr) = example_Critic; % store value function 2023-01-16
        out_agent_Experience(hyperparameterCntr) = example_agentExperience;
    %     out_Adjustments = example_Adjustments; % store Adjustments to parameter vectors, 2023-02-20
        p_outputs(hyperparameterCntr) = example_traces;
    
    end
    
    %% save training data for all cases
    if scenarioCntr == 1
        %% initialize variables used to store results of the different training scenarios
        all_scenarios_out_Policies = cell(numberScenarios,1);   % initialize variable saving policies
        all_scenarios_out_Critics = cell(numberScenarios,1);    % initialize variable saving value functions
        all_scenarios_out_Experience = cell(numberScenarios,1); % initialize variable saving agent experiences
        all_scenarios_p_outputs = cell(numberScenarios,1);      % initialize variable saving modelling parameters and eligibility traces

        %% allocate first entries
        all_scenarios_out_Policies{scenarioCntr} = out_Policies(1:1:StepSizes.nmberOfStepSizeLevels); % save policies
        all_scenarios_out_Critics{scenarioCntr} = out_Critics(1:1:StepSizes.nmberOfStepSizeLevels); % save value function
        all_scenarios_out_Experience{scenarioCntr} = out_agent_Experience(1:1:StepSizes.nmberOfStepSizeLevels); % save agent's experiences
        all_scenarios_p_outputs{scenarioCntr} = p_outputs(1:1:StepSizes.nmberOfStepSizeLevels); % save modelling parameters and eligibility traces
    else
        all_scenarios_out_Policies{scenarioCntr} = out_Policies(1:1:StepSizes.nmberOfStepSizeLevels); % save policies
        all_scenarios_out_Critics{scenarioCntr} = out_Critics(1:1:StepSizes.nmberOfStepSizeLevels); % save value function
        all_scenarios_out_Experience{scenarioCntr} = out_agent_Experience(1:1:StepSizes.nmberOfStepSizeLevels); % save agent's experiences
        all_scenarios_p_outputs{scenarioCntr} = p_outputs(1:1:StepSizes.nmberOfStepSizeLevels); % save modelling parameters and eligibility traces
    end


end % end loop through scenarios
    
%%
delete(myPool)

%% save results
test_results.Experience = all_scenarios_out_Experience; % save experience generated during training
test_results.Policies = all_scenarios_out_Policies;     % save policy networks generated during training
test_results.logging.model_parameters = param;          % save dynamic model's parameters

% filename = '/scratch3/20068530/test_results';
% save(filename,'test_results',"-v7.3");

toc % moved 2022-10-17

%% functions 
% training function
function [out_agentExperience,outPol,p,outCrit] = trainFunction(actor,...
    critic,p,R_bounds,spSample,spSample_2,stepSize) % changed to function on 2022-10-17
    actor.NN.alpha = stepSize; % initialize step size level (2022-10-17)
    for epCntr = 1:1:1 % start loop through episodes (outer loop of earlier code maintained, but there are no episodes here)
        currentTimeStamp = 1; % initialize time stamp
        %% clear state components and action
        clear State_1
        clear State_2
        clear State_3
        clear State_4
        clear State_5
        clear State_6 % (2023-11-17)

        %% select first state components
        if currentTimeStamp == 1
            State_1(currentTimeStamp) = p.SP(1);      % SP for height 1 state 1 (2023-11-05)
            State_2(currentTimeStamp) = p.SP(2);      % SP for height 2 state 2 (2023-11-17)
            State_3(currentTimeStamp) = p.x0(1);      % H_1 is state 2 (2023-11-05)
            State_4(currentTimeStamp) = p.x0(2);      % H_2 is state 3 (2023-11-05)
            State_5(currentTimeStamp) = p.x0(3);      % H_3 is state 4 (2023-11-05)
            State_6(currentTimeStamp) = p.x0(4);      % H_4 is state 5 (2023-11-05)
        end

        %% start loop through steps
        for stepCntr = 1:1:p.nmberOfSteps
            %% assign sampled SP changes
            for i = 1:1:size(spSample.sampleTimes,2)
                if spSample.sampleTimes(i) == currentTimeStamp
                    p.SP(1) = spSample.SP_samples(i);
                    State_1(currentTimeStamp) = p.SP(1); % current SP (2023-06-04)
                end
                % determine the SP at the next time step (next state 3)
                % (2023-06-04)
                if spSample.sampleTimes(i) == currentTimeStamp + 1
                    true_nxtState_1(currentTimeStamp) = spSample.SP_samples(i); % assign next SP as the next state 3 (2023-06-04)
                else
                    true_nxtState_1(currentTimeStamp) = p.SP(1); % updated 2023-06-04
                end
            end

            % sample H2 SP (2023-11-17)
            for j = 1:1:size(spSample_2.sampleTimes,2)
                if spSample_2.sampleTimes(j) == currentTimeStamp
                    p.SP(2) = spSample_2.SP_samples(j);
                    State_2(currentTimeStamp) = p.SP(2); % current SP (2023-06-04)
                end
                % determine the SP at the next time step (next state 3)
                % (2023-06-04)
                if spSample_2.sampleTimes(j) == currentTimeStamp + 1
                    true_nxtState_2(currentTimeStamp) = spSample_2.SP_samples(j); % assign next SP as the next state 3 (2023-06-04)
                else
                    true_nxtState_2(currentTimeStamp) = p.SP(2); % updated 2023-06-04
                end
            end

            %% select valve positions (2023-11-09)
            if isfield(p,'gamma_vec')
                if currentTimeStamp <= size(p.gamma_vec,2)
                    p.gamma1 = p.gamma_vec(currentTimeStamp); % change valve position 1 
                    p.gamma2 = p.gamma1; % set valve position 2 equal to valve position 1
                end
            end
    
            %% initialize control inputs as necessary
            if epCntr ~= 1 && currentTimeStamp == 1
                p.u_1 = p.u_1_prevEnd;   % initial u_1 (2023-11-05)
                p.u_2 = p.u_2_prevEnd;   % initial u_2 (2023-11-05)
            elseif stepCntr == p.nmberOfSteps
                p.u_1_prevEnd = p.u_1;
                p.u_2_prevEnd = p.u_2;
            elseif epCntr == 1 && currentTimeStamp == 1
                p.u_1 = p.u0(1);         % initial u_1 (2023-11-05)
                p.u_2 = p.u0(2);         % initial u_2 (2023-11-05)

            end


            %% scale observed states (2023-11-27)
            [InputsScaled,~] = mapminmax('apply',[State_1(currentTimeStamp),State_2(currentTimeStamp),State_3(currentTimeStamp),State_4(currentTimeStamp),State_5(currentTimeStamp),State_6(currentTimeStamp)]',p.PS_input); % scale inputs
            
            % scaled states for the current time step (2023-12-14)
            S_1_s = InputsScaled(1); % SP 1 scaled
            S_2_s = InputsScaled(2); % SP 2 scaled
            S_3_s = InputsScaled(3); % H1 scaled
            S_4_s = InputsScaled(4); % H2 scaled
            S_5_s = InputsScaled(5); % H3 scaled
            S_6_s = InputsScaled(6); % H4 scaled

            %% obtain control inputs by evaluating the actor network (2023-11-27)
            [u_1_s,u_2_s,~,~,~,~] = evaluate_ReLU_tanh_six_states(actor.NN,S_1_s,S_2_s,S_3_s,S_4_s,S_5_s,S_6_s);
            
            %% add exploratory noise to the selected actions
            u_1_s = u_1_s + actor.NN.stdDev_defined*randn(1,1); % add exploratory noise to the selected action (2023-11-15)
            u_2_s = u_2_s + actor.NN.stdDev_defined*randn(1,1); % add exploratory noise to the selected action (2023-11-15)  

            % scale control inputs back to actual numerical values (2023-11-26)
            [true_Us,~] = mapminmax('reverse',[u_1_s,u_2_s]',p.PS_targets);
            p.u_1 = true_Us(1); % scale control input one to true numerical value
            p.u_2 = true_Us(2); % scale control input two to true numerical value
            
            
            %% saturate control inputs if required (2023-11-05)
            if p.u_1 < p.lower_u_1
                p.u_1 = p.lower_u_1;
                SATFLAG = 1;
            end
            if p.u_2 < p.lower_u_2
                p.u_2 = p.lower_u_2;
                SATFLAG = 1;
            end

            if p.u_1 > p.upper_u_1
                p.u_1 = p.upper_u_1;
                SATFLAG = 1;
            end
            if p.u_2 > p.upper_u_2
                p.u_2 = p.upper_u_2;
                SATFLAG = 1;
            end

            if p.u_1 < p.upper_u_1 && p.u_1 > p.lower_u_1 && p.u_2 < p.upper_u_2 && p.u_2 > p.lower_u_2 % (fixed statement on 2023-11-14)
                SATFLAG = 0;
            end

            %% perform scaling for sampling period (i.e., simulated time for transition from T to (T + 1))
            start = currentTimeStamp*p.sampling_period;
            stop = (currentTimeStamp + 1)*p.sampling_period;
            simTspan = linspace(start,stop,100); % (2023-11-06)

            %% simulate the RL environment dynamics (2023-11-06)
            [~,control_output] = ode23s(@(t,x) myQTDEs(t,x,p,[p.u_1,p.u_2]),simTspan,[State_3(currentTimeStamp),State_4(currentTimeStamp),State_5(currentTimeStamp),...
                State_6(currentTimeStamp)]'); % (UPDATED STATES SENT TO ODE SOLVER TO ACCOMMODATE SEPARATE H1 AND H2 SPs ON 2023-11-17)

            %% next states (2023-11-06)
            nxtState_1(currentTimeStamp) = true_nxtState_1(currentTimeStamp); % H1 SP at next time step
            nxtState_2(currentTimeStamp) = true_nxtState_2(currentTimeStamp); % H2 SP at next time step (2023-11-17)
            nxtState_3(currentTimeStamp) = control_output(end,1); % H_1
            nxtState_4(currentTimeStamp) = control_output(end,2); % H_2
            nxtState_5(currentTimeStamp) = control_output(end,3); % H_3
            nxtState_6(currentTimeStamp) = control_output(end,4); % H_4

            %% scale observed states at next time step (2023-11-27)
            [nxtInputsScaled,~] = mapminmax('apply',[nxtState_1(currentTimeStamp),nxtState_2(currentTimeStamp),nxtState_3(currentTimeStamp),nxtState_4(currentTimeStamp),nxtState_5(currentTimeStamp),nxtState_6(currentTimeStamp)]',p.PS_input); % scale inputs

            S_1_s_nxt = nxtInputsScaled(1);
            S_2_s_nxt = nxtInputsScaled(2);
            S_3_s_nxt = nxtInputsScaled(3);
            S_4_s_nxt = nxtInputsScaled(4);
            S_5_s_nxt = nxtInputsScaled(5);
            S_6_s_nxt = nxtInputsScaled(6);

            %% update critic parameters            
            R = -1*( (nxtState_1(currentTimeStamp) - nxtState_3(currentTimeStamp) )^2 + (nxtState_2(currentTimeStamp) - nxtState_4(currentTimeStamp) )^2);
            
            R = scaleRewards(R,R_bounds); % 2022-10-14

            if exist('SATFLAG','var') && SATFLAG == 1
                if p.includeSaturationPenalty == 1
                    R = R - p.satPenaltyMagnitude; % penalty for saturating the control input (2023-05-03)
                end
            end
            
            %% update critic and actor parameters (fixed 2023-11-15)
            [critic.NN,temporal_diff,p] = RBF_TD_0(critic.NN,S_1_s,S_2_s,S_3_s,S_4_s,S_5_s,S_6_s,S_1_s_nxt,S_2_s_nxt,S_3_s_nxt,S_4_s_nxt,S_5_s_nxt,S_6_s_nxt,p,R); % update the critic network (2023-05-24) (updated 2023-11-17)
            [actor.NN,p] = ActorbackProp_ReLU_tanh_EL_traces(actor.NN,S_1_s,S_2_s,S_3_s,S_4_s,S_5_s,S_6_s,temporal_diff,u_1_s,u_2_s,p); % backpropagation function updated so that action applied is used during the backward pass (2023-01-17) (updated 2023-06-04) (updated again 2023-11-17)

            %% store training data
            out_agentExperience.avgR(stepCntr) = p.avgR; % average reward (2023-06-29)
            out_agentExperience.R(stepCntr) = R;
            out_agentExperience.S1_sqrd_dev(stepCntr) = (nxtState_1(currentTimeStamp) - nxtState_3(currentTimeStamp) )^2; % squared deviation from SP 1
            out_agentExperience.S2_sqrd_dev(stepCntr) = (nxtState_2(currentTimeStamp) - nxtState_4(currentTimeStamp) )^2; % squared deviation from SP 2
            out_agentExperience.State_1(stepCntr) = State_1(currentTimeStamp);
            out_agentExperience.State_2(stepCntr) = State_2(currentTimeStamp);
            out_agentExperience.State_3(stepCntr) = State_3(currentTimeStamp); % 2023-06-04
            out_agentExperience.State_4(stepCntr) = State_4(currentTimeStamp); % 2023-06-09
            out_agentExperience.State_5(stepCntr) = State_5(currentTimeStamp); % 2023-06-12
            out_agentExperience.State_6(stepCntr) = State_6(currentTimeStamp); % 2023-11-17
            out_agentExperience.u_1_true(stepCntr) = p.u_1;   % true control input 1 (2023-11-06)
            out_agentExperience.u_2_true(stepCntr) = p.u_2;   % true control input 1 (2023-11-06)
            out_agentExperience.temp_diff(stepCntr) = temporal_diff; % temporatl difference error (2023-11-17)

            out_agentExperience.scaledState_1(stepCntr) = S_1_s;
            out_agentExperience.scaledState_2(stepCntr) = S_2_s;
            out_agentExperience.scaledState_3(stepCntr) = S_3_s;
            out_agentExperience.scaledState_4(stepCntr) = S_4_s;
            out_agentExperience.scaledState_5(stepCntr) = S_5_s;
            out_agentExperience.scaledState_6(stepCntr) = S_6_s;
            
            %% record the valve positions applied to the process (2023-11-09)
            out_agentExperience.gamma1(stepCntr) = p.gamma1;
            out_agentExperience.gamma2(stepCntr) = p.gamma2;

            %% Save neural networks
            if ismember(stepCntr,p.policySavingVec) % save policy if required
                outPol.NN{1,p.saveCntr} = actor.NN; % save actor network at current time stamp (2023-12-14)
                outCrit.NN{1,p.saveCntr} = critic.NN; % save critic network at current time stamp (2023-12-14)
                p.saveCntr = p.saveCntr + 1; % increment counter
            end

            %% increment time stamp with 1
            currentTimeStamp = currentTimeStamp + 1;
    
            %% update state S <- S_nxt
            State_1(currentTimeStamp) = nxtState_1(currentTimeStamp - 1);
            State_2(currentTimeStamp) = nxtState_2(currentTimeStamp - 1);
            State_3(currentTimeStamp) = nxtState_3(currentTimeStamp - 1); % 2023-06-04
            State_4(currentTimeStamp) = nxtState_4(currentTimeStamp - 1); % 2023-06-09
            State_5(currentTimeStamp) = nxtState_5(currentTimeStamp - 1); % 2023-06-12
            State_6(currentTimeStamp) = nxtState_6(currentTimeStamp - 1); % 2023-11-17
    
            if ismember(currentTimeStamp,p.reportingVec)
                fprintf('%d\n',currentTimeStamp) % provide an update on training progress (2023-06-29)
            end
        end % end loop through steps

    end % end loop through episodes

end % end function

% function to scale reward to a value between 0 and 1
function reward_scaled = scaleRewards(R,R_bounds)
    reward_scaled = ( R - R_bounds.low )/...
        ( R_bounds.high - R_bounds.low );
end

% QTP model for call to DE solver (2023-11-06)
function dHdt = myQTDEs(t,x,param,u)
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

%% FUNCTIONS FOR NN CRITIC
% critic backpropagation
function [NN,temporal_diff,p] = RBF_TD_0(NN,State_1,State_2,State_3,State_4,State_5,State_6,nxtState_1,nxtState_2,nxtState_3,nxtState_4,nxtState_5,nxtState_6,p,R) % 2023-01-17 (updated 2023-11-17)

    [temporal_diff,yrnj,p,z_rnj_hidden,z_rnj_output] = calculateTemporalDiff(NN,p,State_1,State_2,State_3,State_4,State_5,State_6,nxtState_1,nxtState_2,nxtState_3,nxtState_4,nxtState_5,nxtState_6,R); % calculate the current temporal difference (2023-05-24)

    % BACKWARD PASS BEGIN
    % loop though the neurons in the output layer
    for j_output_bw = 1:1:NN.k_output
        %% add backward computation
        % calculate the derivative of the activation function
        % evaluated at the output of the linear combiner of the
        % final network layer
        dfdz_outer_mu_z = ( sech(z_rnj_output(j_output_bw)) )^2; % tanh activation of the output node

        %% NOTE: Standard deviation squared purposefully not included in denominator
        % calculate the partial derivative of the loss with respect
        % to zLnj for the last network layer (L)
        % calculate delta_Lnj = dprobdmu*dfdz_outer_mu_z by
        delta_Lnj = 1*dfdz_outer_mu_z; % temporal difference error included at update step (2023-05-24)

    end % end backward pass through output layer
    
    [NN,p] = Update_critic_parameters_ReLU_tanh_EL_traces(NN,State_1,State_2,State_3,State_4,State_5,State_6,delta_Lnj,temporal_diff,yrnj,z_rnj_hidden,p); % update critic parameters (updated 2023-11-17)

end % end backpropagation function

function [temporal_diff,yrnj,p,z_rnj_hidden,z_rnj_output] = calculateTemporalDiff(NN,p,State_1,State_2,State_3,State_4,State_5,State_6,nxtState_1,nxtState_2,nxtState_3,nxtState_4,nxtState_5,nxtState_6,R) % updated 2023-06-04
    [V_S_crnt,z_rnj_hidden,yrnj,z_rnj_output] = evaluate_ReLU_tanh_six_states_one_output(NN,State_1,State_2,State_3,State_4,State_5,State_6); % evaluate critic network at the current state
    [V_S_nxt,~,~,~] = evaluate_ReLU_tanh_six_states_one_output(NN,nxtState_1,nxtState_2,nxtState_3,nxtState_4,nxtState_5,nxtState_6); % evaluate critic network at the next state

    [V_S_crnt,~] = mapminmax('reverse',V_S_crnt,p.PS_Value_targets);
    [V_S_nxt,~] = mapminmax('reverse',V_S_nxt,p.PS_Value_targets);

    temporal_diff = R - p.avgR + V_S_nxt - V_S_crnt; % calculate the temporal difference in the average reward setting (2023-06-29)
    p = updateAverageReward(p,temporal_diff); % update the 
end

% function used to update the average reward (2023-06-29)
function p = updateAverageReward(p,temporal_diff)
    p.avgR = p.avgR + p.avgRAlpha*temporal_diff; % update for average reward (2023-06-29)
end