function _safe_log(x::Float64)
    if x <= 0.0
        return -1.0e10; # a large negative number
    else
        return log(x);
    end
end

function _objective_function(w::Array{Float64,1}, ḡ::Array{Float64,1}, 
    Σ̂::Array{Float64,2}, R::Float64, μ::Float64, ρ::Float64)


    # TODO: This version of the objective function includes the barrier term, and the penalty terms -
    f = w'*(Σ̂*w) + (1/(2*ρ))*((sum(w) - 1.0)^2 + (transpose(ḡ)*w - R)^2) - (1/μ)*sum(_safe_log.(w));

    # TODO: This version of the objective function does NOT have the barrier term
    # f = w'*(Σ̂*w) + (1/(2*ρ))*((sum(w) - 1.0)^2 + (transpose(ḡ)*w - R)^2);


    return f;
end

"""
    function solve(model::MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem; 
        verbose::Bool = true, K::Int = 10000, T₀::Float64 = 1.0, T₁::Float64 = 0.1, 
        α::Float64 = 0.99, β::Float64 = 0.01, τ::Float64 = 0.99,
        μ::Float64 = 1.0, ρ::Float64 = 1.0) -> MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem

The `solve` function solves the minimum variance portfolio allocation problem using a simulated annealing approach for a given instance 
    of the [`MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem`](@ref) problem type.

### Arguments
- `model::MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem`: An instance of the [`MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem`](@ref) that defines the problem parameters.
- `verbose::Bool = true`: A boolean flag to control verbosity of output during optimization.
- `K::Int = 10000`: The initial number of iterations at each temperature level.
- `T₀::Float64 = 1.0`: The initial temperature for the simulated annealing process.
- `T₁::Float64 = 0.1`: The final temperature for the simulated annealing process.
- `α::Float64 = 0.99`: The cooling rate for the temperature.
- `β::Float64 = 0.01`: The step size for generating new candidate solutions.
- `τ::Float64 = 0.99`: The penalty parameter update factor.
- `μ::Float64 = 1.0`: The initial penalty parameter for the logarithmic barrier term.
- `ρ::Float64 = 1.0`: The initial penalty parameter for the equality constraints.

### Returns
- `MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem`: The input model instance updated with the optimal portfolio weights.

"""
function solve(model::MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem; 
    verbose::Bool = true, K::Int = 10000, T₀::Float64 = 1.0, T₁::Float64 = 0.1, 
    α::Float64 = 0.99, β::Float64 = 0.01, τ::Float64 = 0.99,
    μ::Float64 = 1.0, ρ::Float64 = 1.0)

    # initialize -
    has_converged = false;

    # unpack the model parameters -
    w = model.w;
    ḡ = model.ḡ;
    Σ̂ = model.Σ̂;
    R = model.R;

    # initialize parameters for simulated annealing -
    T = T₀; # initial T -
    current_w = copy(w);
    current_f = _objective_function(current_w, ḡ, Σ̂, R, μ, ρ);
    
    # best solution found so far -
    w_best = copy(current_w);
    f_best = current_f;
    KL = K;
    candidate_w = similar(current_w);
    perturb = similar(current_w);

    while has_converged == false
    
        accepted_counter = 0; 
    
    #TODO: simulated annealing main loop

        for _ in 1:KL
            randn!(perturb);
            total = 0.0;
            @inbounds @simd for i in eachindex(candidate_w)
                val = current_w[i] + β*perturb[i];
                if val < 1.0e-6
                    val = 1.0e-6;
                end
                candidate_w[i] = val;
                total += val;
            end
            inv_total = inv(total);
            @inbounds @simd for i in eachindex(candidate_w)
                candidate_w[i] *= 1.0/total;
            end

            new_f = _objective_function(candidate_w, ḡ, Σ̂, R, μ, ρ);
            Δf = new_f - current_f;

            if Δf <= 0.0
                copyto!(current_w, candidate_w);
                current_f = new_f;
                accepted_counter += 1;
            elseif T > 0.0
                prob = exp(-Δf / T);
                if rand() < prob
                    copyto!(current_w, candidate_w);
                    current_f = new_f;
                    accepted_counter += 1;
                end
            end

            if new_f < f_best
                copyto!(w_best, candidate_w);
                f_best = new_f;
            end
        end

        # update KL -
        fraction_accepted = accepted_counter/KL; # what is the fraction of accepted moves
        
        # Case 1: we are accepting alot, so decrease the number of iterations
        if (fraction_accepted > 0.8)
            KL = ceil(Int, 0.75*KL);
        end

        # Case 2: not accepting many moves, so increase the number of iteratons
        if (fraction_accepted < 0.2)
            KL = ceil(Int, 1.5*KL);
        end

        # update penalty parameters and T -
        # μ *= τ*μ;
        # ρ *= τ*ρ;

        # if (T ≤ T₁)
        #     has_converged = true;
        # else
        #     T *= (α*T); # Not done yet, so decrease the T -
        # end
       
        # correctly decay penalty parameters and temperature
        μ *= τ*μ;
        ρ *= τ*ρ;

        if (T ≤ T₁)
            has_converged = true;
        else
            T *= (α*T); # Not done yet, so decrease the T -
        end
        μ *= τ
        ρ *= τ

        if T <= T₁
            has_converged = true
        else
            T *= α    # multiplicative cooling: T = α * T
        end

        if verbose
            println("T=$(round(T,digits=6)) μ=$(round(μ,digits=6)) ρ=$(round(ρ,digits=6)) accepted_frac=$(round(fraction_accepted,digits=4))")
        end
# ...existing code...
    end

    # update the model with the optimal weights -
    model.w = w_best;

    # return the model -
    return model;
end

"""
    function solve(problem::MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem) -> Dict{String,Any}

The `solve` function solves the Markowitz risky asset-only portfolio choice problem for a given instance of the [`MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem`](@ref) problem type.
The `solve` method checks for the optimization's status using an assertion. Thus, the optimization must be successful for the function to return.
Wrap the function call in a `try` block to handle exceptions.


### Arguments
- `problem::MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem`: An instance of the [`MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem`](@ref) that defines the problem parameters.

### Returns
- `Dict{String, Any}`: A dictionary with optimization results.

The results dictionary has the following keys:
- `"reward"`: The reward associated with the optimal portfolio.
- `"argmax"`: The optimal portfolio weights.
- `"objective_value"`: The value of the objective function at the optimal solution.
- `"status"`: The status of the optimization.
"""
function solve(problem::MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem)::Dict{String,Any}

    # initialize -
    results = Dict{String,Any}()
    Σ = problem.Σ;
    μ = problem.μ;
    R = problem.R;
    bounds = problem.bounds;
    wₒ = problem.initial

    # setup the problem -
    d = length(μ)
    model = Model(()->MadNLP.Optimizer(print_level=MadNLP.ERROR, max_iter=500))
    @variable(model, bounds[i,1] <= w[i=1:d] <= bounds[i,2], start=wₒ[i])

    # set objective function -
    @objective(model, Min, transpose(w)*Σ*w);

    # setup the constraints -
    @constraints(model, 
        begin
            # my turn constraint
            transpose(μ)*w >= R
            sum(w) == 1.0
        end
    );

    # run the optimization -
    optimize!(model)

    # check: was the optimization successful?
    @assert is_solved_and_feasible(model)

    # populate -
    w_opt = value.(w);
    results["argmax"] = w_opt
    results["reward"] = transpose(μ)*w_opt; 
    results["objective_value"] = objective_value(model);
    results["status"] = termination_status(model);

    # return -
    return results
end
