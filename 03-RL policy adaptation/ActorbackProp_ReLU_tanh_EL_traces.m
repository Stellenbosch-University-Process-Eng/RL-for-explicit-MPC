% backpropagation function for actor.  Eligibility traces used to
% incorporate previous gradient information.
function [NN,p] = ActorbackProp_ReLU_tanh_EL_traces(NN,State_1,State_2,State_3,State_4,State_5,State_6,temporal_diff,sampled_u_1,sampled_u_2,p) % 2023-01-17

    % evaluate the actor network (2023-11-06)
    [yrnj_output_1_BEFORE_SAMPLING,yrnj_output_2_BEFORE_SAMPLING,z_rnj_hidden,yrnj,z_rnj_outer_output_1,z_rnj_output_2] = evaluate_ReLU_tanh_six_states(NN,State_1,State_2,State_3,State_4,State_5,State_6); % (2023-11-27) %evaluateActor(NN,State_1,State_2,State_3,State_4,State_5,State_6); % (updated 2023-11-17)

    yrnj_output_1 = sampled_u_1; % sampled u_1 (2023-11-06)
    yrnj_output_2 = sampled_u_2; % sampled u_2 (2023-11-06)


    % BACKWARD PASS BEGIN
    % loop though the neurons in the output layer
    for j_output_bw = 1:1:1
        %% add backward computation
        % calculate the derivative of the activation function
        % evaluated at the output of the linear combiner of the
        % final network layer
        dfdz_outer_mu_z = ( sech(z_rnj_outer_output_1(j_output_bw)) )^2; % tanh activation of first output node
        

        %% NOTE: Standard deviation squared purposefully not included in denominator
        dprobdmu_1 = +1*( ( yrnj_output_1 - yrnj_output_1_BEFORE_SAMPLING ) );

        % calculate the partial derivative of the loss with respect
        % to zLnj for the last network layer (L)
        % calculate delta_Lnj = dprobdmu*dfdz_outer_mu_z by
        delta_Lnj_1 = dprobdmu_1*dfdz_outer_mu_z;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Repeat calculation for the second node
        dfdz_outer_NODE_TWO = ( sech(z_rnj_output_2(j_output_bw)) )^2; % tanh activation of second output node
        dprobdmu_2 = +1*( ( yrnj_output_2 - yrnj_output_2_BEFORE_SAMPLING ) ); % error for node two, 2023-02-22
        delta_Lnj_2 = dprobdmu_2*dfdz_outer_NODE_TWO; % partial derivative of loss with respect to zLnj for the second output node, 2023-02-22
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end % end backward pass through output layer 

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
                node_ONE_contribution(j_hidden_bw) = delta_Lnj_1*NN.output_layer_parameters(j_hidden_bw+1,1); % contribution of first output node, 2023-02-22
                node_TWO_contribution(j_hidden_bw) = delta_Lnj_2*NN.output_layer_parameters(j_hidden_bw+1,2); % contribution of second output node, 2023-02-22
                e_hidden_nj(j_hidden_bw) = node_ONE_contribution(j_hidden_bw) + node_TWO_contribution(j_hidden_bw); % calculate error propagated backward from output layer, 2023-02-22
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                % calculate the partial derivative of the loss with respect
                % to z_hidden_nj for the hidden network layer (hidden)
                delta_hidden_nj(j_hidden_bw) = e_hidden_nj(j_hidden_bw)*...
                    dfdz_hidden(j_hidden_bw);
                
            end % end backward pass through the hidden layer

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % added on 2023-11-26
            delta_Lnj_overall = [delta_Lnj_1,delta_Lnj_2]; % create a vector containing the "deltas" of the two output layer nodes, 2023-02-22
            dThetaLnj = zeros(size([1;( yrnj(1:1:NN.k_hidden) )'],1),NN.k_output); % initialize dThetaLnj to have two columns
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % update all network parameters for each new data point
            % (pattern-by-pattern mode of network training)
            % update output layer parameters
            for j_output_update = 1:1:NN.k_output

                %% update eligibility trace vector
                output_gradient_without_TD_diff = delta_Lnj_overall(j_output_update).*...
                    [1;( yrnj(1:1:NN.k_hidden) )'];
                p.trace_actor_output(:,j_output_update) = update_EL_trace_vector(p.trace_actor_output(:,j_output_update),p.actor_lambda,output_gradient_without_TD_diff); % update eligibility trace using output layer gradient

                %% One-Step Actor-Critic update rule
                outerLayerGradient(:,j_output_update) = ... % (2023-11-06)
                    temporal_diff*p.trace_actor_output(:,j_output_update); % vector of outer layer gradients, 2023-02-20
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
                p.trace_actor_hidden(:,j_hidden_update) = update_EL_trace_vector(p.trace_actor_hidden(:,j_hidden_update),p.actor_lambda,hidden_gradient_without_TD_diff); % update eligibility trace using hidden layer gradient

                %% update hidden layer parameters
                % compute adjustment to previous parameter estimates
                % (negative sign is a result of gradient descent being
                % applied)
                dTheta_hidden_nj = ...
                    +1*NN.alpha*temporal_diff*p.trace_actor_hidden(:,j_hidden_update);
                
                % update all hidden layer parameters (each column
                % corresponds to a node)
                 NN.hidden_layer_parameters(:,j_hidden_update) = ...
                     NN.hidden_layer_parameters(:,j_hidden_update) + ...
                     dTheta_hidden_nj;
                
            end % end update of hidden layer parameters

end % end backpropagation function