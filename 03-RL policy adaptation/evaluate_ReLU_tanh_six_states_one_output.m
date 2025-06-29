%% Function used to evaluate neural network with ReLU hidden layer activation
%% and tanh output layer activation.
%% Edward Bras
%% Date: 2024-04-26
function [yrnj_output,z_rnj_hidden,yrnj,z_rnj_outer] = evaluate_ReLU_tanh_six_states_one_output(NN,X,Y,Z,A,B,C)
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
            fprintf('\n yrnj not assigned\n');
            display(z_rnj_hidden(j_hidden_fw))
        end
        
    end % end loop through the neurons in the hidden layer
    
        % loop through the neurons in the output layer
        for j_output_fw = 1:1:1
            %% add forward computations
            % compute the output of the linear combiner at the jth
            % neuron in layer r at time instant (training sample) n
            z_rnj_outer(j_output_fw) = ...
                NN.output_layer_parameters(:,1)'*...
                [1;( yrnj(1:1:NN.k_hidden) )'];
            
            % compute the output after passing the result though
            % the activation function 
            yrnj_output = tanh(z_rnj_outer); % tanh activation of node 1
    
        end % end loop through the neurons in the output layer
    
    end % end loop through the number of network layers
    % FORWARD PASS END
end