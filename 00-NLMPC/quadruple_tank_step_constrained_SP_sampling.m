function spSample = quadruple_tank_step_constrained_SP_sampling(spSample,SP_SS,SP_index)
    
    % Initialize the set points array
    spSample.SP_samples = zeros(1, spSample.nmberTimes); % Preallocate for efficiency
    
    % Generate the first set point
    spSample.SP_samples(1) = SP_SS(SP_index);
    
    % Generate subsequent set points with step size limit
    for i = 2:spSample.nmberTimes
        % Generate a random step within the step limit
        step = spSample.stepLim * (- 1 + 2 * rand); % Step in range [-stepLim, stepLim]
        % Calculate the next set point with step constraint
        next_SP_before_sat = spSample.SP_samples(i-1) + step;
        % Ensure the next set point stays within the bounds
        next_SP = max(spSample.setPoint_low, min(spSample.setPoint_high, next_SP_before_sat));
        
        % generate a new set point within range if the set point was
        % saturated
        if next_SP_before_sat == next_SP
            spSample.SP_samples(i) = next_SP;
        elseif next_SP ==  spSample.setPoint_low
            next_SP = next_SP + spSample.stepLim*rand; % produce a new step within the required range
        elseif next_SP == spSample.setPoint_high
            next_SP = next_SP - spSample.stepLim*rand; % produce a new step within the required range
            
        end
        
        % Assign the next set point
        spSample.SP_samples(i) = next_SP;

    end

end