% function that updates eligibility trace vectors.
function trace_updated = update_EL_trace_vector(trace,lambda,crntGradient)
    trace_updated = lambda*trace + crntGradient;
end