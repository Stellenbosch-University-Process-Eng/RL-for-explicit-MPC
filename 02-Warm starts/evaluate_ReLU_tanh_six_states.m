%% Script that evaluates a neural network with ReLU hidden layer activation, 
%% tanh output activation, and two output layer nodes.
%% Name: Edward Bras
%% Date: 2023-02-15
function [yrnj_output,yrnj_output_NODE_TWO,z_rnj_hidden,yrnj,z_rnj_outer,z_rnj_outer_NODE_TWO,div_Flag] = evaluate_ReLU_tanh_six_states(NN,X,Y,Z,A,B,C)
    div_Flag = 0; % flag indicating whether divergence is occurring

    % FORWARD PASS BEGIN
    % loop through the network layers
    for r = 1:1:NN.L
    
    % loop through the neurons in the hidden layer
    for j_hidden_fw = 1:1:NN.k_hidden
        %% add forward computations
        % compute the output of the linear combiner of the jth
        % neuron in layer r at time instant (training sample) n
        z_rnj_hidden(j_hidden_fw) = ...        
            NN.hidden_layer_parameters(:,j_hidden_fw)'*...
            [1;X;Y;Z;A;B;C];
        
        % compute the output after passing the result through
        % the activation functions 
        if z_rnj_hidden(j_hidden_fw) >= 0
            yrnj(j_hidden_fw) = z_rnj_hidden(j_hidden_fw); % linear section of ReLU activation
        elseif z_rnj_hidden(j_hidden_fw) < 0 
            yrnj(j_hidden_fw) = 0;                         % flat section of ReLU activation
        else
                
            div_Flag = 1; % set flag to indicate divergence

            % remove the effect of "NaN" artificially
            z_rnj_hidden(j_hidden_fw) = 0;
            yrnj(j_hidden_fw) = 0;
        end
        
    end % end loop through the neurons in the hidden layer
    
        % loop through the neurons in the output layer
        for j_output_fw = 1:1:1%NN.k_output
            %% add forward computations
            % compute the output of the linear combiner at the jth
            % neuron in layer r at time instant (training sample) n
            z_rnj_outer(j_output_fw) = ...
                NN.output_layer_parameters(:,1)'*...
                [1;( yrnj(1:1:NN.k_hidden) )'];
            
            % compute the output after passing the result though
            % the activation function 
            yrnj_output = tanh(z_rnj_outer); % tanh activation of node 1

            % repeat calculations for the second output node,
            % 2023-02-22
            z_rnj_outer_NODE_TWO(j_output_fw) = ...
                NN.output_layer_parameters(:,2)'*...
                [1;( yrnj(1:1:NN.k_hidden) )'];
            
            yrnj_output_NODE_TWO = tanh(z_rnj_outer_NODE_TWO); % tanh activation of node 2

    
        end % end loop through the neurons in the output layer
    
    end % end loop through the number of network layers
    % FORWARD PASS END
end