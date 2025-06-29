%% Backpropagation function for critic with ReLU hidden layer activation and tanh
%% output layer activation.  Eligibility traces can be used to
%% incorporate previous gradient information, but were not used when generating
%% the results reported in our paper, i.e. lambda and eligibility traces were set
%% equal to zero.
function [NN,p] = Update_critic_parameters_ReLU_tanh_EL_traces(NN,State_1,State_2,State_3,State_4,State_5,State_6,delta_Lnj,temporal_diff,yrnj,z_rnj_hidden,p)

    % loop through the neurons in the hidden layer
    for j_hidden_bw = 1:1:NN.k_hidden
        %% add backward computation
        % calculate the derivative of the activation function
        % evaluated at the output of the linear combiner of the
        % hidden network layer
        if z_rnj_hidden(j_hidden_bw) >= 0 % Linear portion of ReLU node active
            dfdz_hidden(j_hidden_bw) = 1;
        elseif z_rnj_hidden(j_hidden_bw) < 0 % Linear portion of ReLU node inactive
            dfdz_hidden(j_hidden_bw) = 0;
        end
        % calculate the error obtained by subtracting the measured
        % output from the hidden layer's output
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % added on 2023-11-26
        e_hidden_nj(j_hidden_bw) =  delta_Lnj*NN.output_layer_parameters(j_hidden_bw+1,1); % calculate error propagated backward from output layer, 2023-02-22
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % calculate the partial derivative of the loss with respect
        % to z_hidden_nj for the hidden network layer (hidden)
        delta_hidden_nj(j_hidden_bw) = e_hidden_nj(j_hidden_bw)*...
            dfdz_hidden(j_hidden_bw);
        
    end % end backward pass through the hidden layer
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % added on 2023-11-26
    delta_Lnj_overall = delta_Lnj; % create a vector containing the "deltas" of the two output layer nodes, 2023-02-22
    dThetaLnj = zeros(size([1;( yrnj(1:1:NN.k_hidden) )'],1),NN.k_output); % initialize dThetaLnj to have one column
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % update all network parameters for each new data point
    % (pattern-by-pattern mode of network training)
    % update output layer parameters
    for j_output_update = 1:1:NN.k_output

        %% update eligibility trace vector
        output_gradient_without_TD_diff = delta_Lnj_overall(j_output_update).*...
            [1;( yrnj(1:1:NN.k_hidden) )'];
        p.trace_critic_output(:,j_output_update) = update_EL_trace_vector(p.trace_critic_output(:,j_output_update),p.critic_lambda,output_gradient_without_TD_diff); % update eligibility trace using output layer gradient

        %% One-Step Actor-Critic update rule
        outerLayerGradient(:,j_output_update) = ... % (2023-11-06)
            temporal_diff*p.trace_critic_output(:,j_output_update); % vector of outer layer gradients, 2023-02-20
        dThetaLnj(:,j_output_update) = +1*NN.alpha*outerLayerGradient(:,j_output_update); % calculate adjustments to layer parameters, 2023-02-20
    
        % update all output layer parameters
        NN.output_layer_parameters(:,j_output_update) = NN.output_layer_parameters(:,j_output_update) + ...
            dThetaLnj(:,j_output_update);
    
    end % end update of output layer parameters
    
    
    % update hidden layer parameters
    for j_hidden_update = 1:1:NN.k_hidden

        %% update eligibility trace vector
        hidden_gradient_without_TD_diff = delta_hidden_nj(j_hidden_update).*...
            [1;State_1;State_2;State_3;State_4;State_5;State_6];
        p.trace_critic_hidden(:,j_hidden_update) = update_EL_trace_vector(p.trace_critic_hidden(:,j_hidden_update),p.critic_lambda,hidden_gradient_without_TD_diff); % update eligibility trace using hidden layer gradient

        %% update hidden layer parameters
        % compute adjustment to previous parameter estimates
        % (negative sign is a result of gradient descent being
        % applied)
        dTheta_hidden_nj = ...
            +1*NN.alpha*temporal_diff*p.trace_critic_hidden(:,j_hidden_update);
        
        % update all hidden layer parameters (each column
        % corresponds to a node)
         NN.hidden_layer_parameters(:,j_hidden_update) = ...
             NN.hidden_layer_parameters(:,j_hidden_update) + ...
             dTheta_hidden_nj;
        
    end % end update of hidden layer parameters

end % end backpropagation function