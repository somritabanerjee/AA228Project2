# using LightGraphs
using Printf
using DataFrames
using CSV
# using GraphPlot
using SpecialFunctions
# using BayesNets # only for testing my scoring function against bayesian_score

function write_policy(policy, filename)
    open(filename, "w") do io
        for i=1:length(policy)
            @printf(io, "%s\n", policy[i])
        end
    end
end

function compute(infile::String, outfile::String)
    println("Computing infile to outfile")
    # Read the data available
    if (infile=="large.csv")
        # CSV.File read takes too long. Using readtable (deprecated) instead.
        data = readtable("large.csv", separator=',', header=true);
    else
        data=CSV.File(infile) |> DataFrame;
    end
    # distinctStates=by(data,:s,nrow);
    # numStates=maximum(data.s,data.sp);
    # numStatesMentionedInData=size(distinctStates,1);
    # policy=Dict{Int, Int}();
    # for i=1:numStates
    #     policy[distinctStates[i,1]]=0;
    # end
    if (infile=="small.csv")
        policy=findPolicyForSmall(data);
    elseif (infile=="medium.csv")
        policy=findPolicyForMedium(data);
    elseif (infile=="large.csv")
        policy=findPolicyForLarge(data);
    end
    write_policy(policy,outfile);
end

struct TransitionAndRewardFunctions
    T::Array{Float64}
    R::Array{Float64}
end

function computeTandR(numStates, numActions, data)
    T=zeros(numStates,numActions,numStates);
    R=zeros(numStates,numActions);
    rho=zeros(numStates,numActions);
    N=zeros(numStates, numActions, numStates);
    for i in 1:size(data, 1)
        s=data[i,1];
        a=data[i,2];
        r=data[i,3];
        sp=data[i,4];
        N[s,a,sp] += 1;
        rho[s,a]+=r;
    end
    for s in collect(1:numStates)
        for a in collect(1:numActions)
            sumOfNsa=sum(N[s,a,:]);
            if (sumOfNsa !=0)
                R[s,a]=rho[s,a]/sum(N[s,a,:]);
                for sp in collect(1:numStates)
                    T[s,a,sp]=N[s,a,sp]/sum(N[s,a,:]);
                end
            end
        end
    end
    return TransitionAndRewardFunctions(T,R);
end

function doValueIteration(T,R,numStates,numActions, discount, delta, maxIter)
    k=0;
    Uold=zeros(numStates);
    Unew=zeros(numStates);
    converged=false;
    while (!converged)
        Uold=copy(Unew);
        for s=1:numStates
            UtilityFunc=zeros(numActions);
            for a=1:numActions
                rollingSum=0;
                for sp=1:numStates
                    rollingSum+=T[s,a,sp]*Uold[sp];
                end
                UtilityFunc[a]=R[s,a]+ discount*rollingSum;
            end
            (maxValue,ind)=findmax(UtilityFunc);
            Unew[s]=maxValue;
        end
        k+=1;
        if (maximum(abs.(Unew-Uold))<=delta || k>maxIter)
            converged=true;
            @show(maximum(abs.(Unew-Uold)))
            @show(k);
        end
    end
    Ustar=Unew;
    optimalPolicy=ones(numStates,1);
    for s=1:numStates
        UtilityFunc=zeros(numActions);
        for a=1:numActions
            rollingSum=0;
            for sp=1:numStates
                rollingSum+=T[s,a,sp]*Ustar[sp];
            end
            UtilityFunc[a]=R[s,a]+ discount*rollingSum;
        end
        (maxValue,ind_max)=findmax(UtilityFunc);
        optimalPolicy[s]=ind_max;
    end
    return optimalPolicy;
end

function doQLearning(numStates,numActions,data,discount,alpha,maxNumIter; Qinit=zeros(numStates,numActions), threshold=0.01)
    Q=Qinit;
    numIter=1;
    update=threshold+1; # some non-zero initialization
    while (numIter<=maxNumIter && update > threshold)
        update=0;
        for i in 1:size(data, 1)
            s=data[i,1];
            a=data[i,2];
            r=data[i,3];
            sp=data[i,4];
            (maxValue,ind_max)=findmax(Q[sp,:]);
            Q[s,a] = Q[s,a] + alpha*(r + discount*maxValue - Q[s,a]);
            # Find the biggest change in Q values for this episode
            update=max(update, alpha*(r + discount*maxValue - Q[s,a]));
        end
        @show(Q[150413,1])
        numIter+=1;
    end
    @show(numIter)
    @show(update)
    return Q;
end

function findOptimalPolicyFromQ(Q,numStates)
    optimalPolicy=ones(numStates,1);
    for s=1:numStates
        (maxValue,ind_max)=findmax(Q[s,:]);
        optimalPolicy[s]=ind_max;
    end
    return optimalPolicy;
end

function findPolicyForSmall(data)
    numStates=100;
    numActions=4;
    uniformPolicy=ones(numStates,1);
    # Maximum likelihood model-based reinforcement learning
    TR=computeTandR(numStates, numActions, data);
    discount=0.9;
    delta=0.0;
    maxIter=1000;
    optimalPolicy=doValueIteration(TR.T,TR.R,numStates,numActions, discount, delta, maxIter);
    return optimalPolicy;
    # Q-learning - For small, this gives a lower score
    # discount_Q=0.9;
    # alpha=1;
    # numIter=10;
    # Qvalues=doQLearning(numStates,numActions,data,discount_Q,alpha, numIter);
    # optimalPolicy=findOptimalPolicyFromQ(Qvalues,numStates);
    # return optimalPolicy;
end

function findPolicyForMedium(data)
    numStates=50000;
    numActions=7;
    uniformPolicy=ones(numStates,1);
    # Q-learning
    discount_Q=0.9;
    alpha=1;
    numIter=1;
    Qvalues=doQLearning(numStates,numActions,data,discount_Q,alpha,numIter);
    optimalPolicy=findOptimalPolicyFromQ(Qvalues,numStates);
    return optimalPolicy;
end

function findPolicyForLarge(data)
    numStates=312020;
    numActions=9;
    uniformPolicy=ones(numStates,1);
    # # Q-learning
    # discount_Q=0.95; # given to us!
    # uniqueStates=unique(data[:,1]);
    # numUniqueStates=size(uniqueStates, 1);
    # alpha=1/numUniqueStates;
    # @show numUniqueStates
    # Qinit=ones(numStates,numActions);
    # for si=1:numStates
    #     if (si==150413 || si==151203 || si==150211)
    #         for ai=1:9
    #             if (ai>=5)
    #                 Qinit[si,ai]=-100;
    #             else
    #                 Qinit[si,ai]=100;
    #             end
    #         end
    #     else
    #         for ai=1:9
    #             if (ai>=5)
    #                 Qinit[si,ai]=100;
    #             else
    #                 Qinit[si,ai]=-100;
    #             end
    #         end
    #     end
    # end
    # numIter=500;
    # Qvalues=doQLearning(numStates,numActions,data,discount_Q,alpha,numIter,Qinit=Qinit);
    # optimalPolicy=findOptimalPolicyFromQ(Qvalues,numStates);
    # return optimalPolicy;

    # Value iteration by approximating T and R
    s=data[1];
    a=data[2];
    r=data[3];
    sp=data[4];
    numRows=size(data,1)
    uniqueInitialStates=unique(s);
    uniqueFinalStates=unique(sp);
    # We've found that these are the same 500 states in both cases
    numUniqueStates=size(uniqueInitialStates, 1);
    s_new_numbering=zeros(Int64,numRows);
    sp_new_numbering=zeros(Int64,numRows);
    @show numRows
    for i=1:numRows
        sp_new_numbering[i] = findfirst(fs -> fs==sp[i], uniqueFinalStates);
        s_new_numbering[i] = findfirst(is -> is==s[i], uniqueInitialStates);
    end
    @show sp_new_numbering[1:10]
    T=zeros(numUniqueStates,numActions,numUniqueStates);
    R=zeros(numUniqueStates,numActions);
    # Can write R(s,a) which is mostly zeros except at a few states
    G0Indices=findall(ri -> ri>0, r);
    @show G0Indices[1:10]
    G0Rewards=r[G0Indices];
    G0Actions=a[G0Indices];
    G0States=s_new_numbering[G0Indices];
    # @show a[1:10]
    # @show G0States
    for k=1:length(G0Indices)
        R[G0States[k],G0Actions[k]]=G0Rewards[k];
    end
    # T(sp|s,a) can be approximated
    # for a=1-4, T(s|s,a)=1 and T(s'|s,a) where s' not equal to s is 0
    for a=1:4
        for s=1:numUniqueStates
            T[s,a,s]=1;
        end
    end
    # for a=5-9 T(s'|s,a) is evenly distributed
    # Can improve on this because for some sk T(s'|sk,a) is only 1 if s'=sk
    # But we would have to identify those sk
    for a=5:9
        for s=1:numUniqueStates
            for sp=1:numUniqueStates
                T[s,a,sp]=1/numUniqueStates;
            end
        end
    end
    # @show T
    # With these R and T estimates, try value iteration
    discount=0.95;
    delta=0.1;
    maxIter=1000;
    optimalPolicyForUniqueStates=doValueIteration(T,R,numUniqueStates,numActions, discount, delta, maxIter);
    # Need to convert to an optimal policy for all numStates
    optimalPolicy=zeros(numStates,1);
    for i=1:numStates
        if (i in uniqueInitialStates)
            idxInUniqueStates = findfirst(is -> is==i, uniqueInitialStates);
            optimalPolicy[i]=optimalPolicyForUniqueStates[idxInUniqueStates];
        else
            # Pick an action that is not stationary, i.e. not 1-4
            optimalPolicy[i]=5;
        end
    end
    return optimalPolicy

end


# Create the files for submission
inputfilename = ["small.csv", "medium.csv", "large.csv"]
outputfilename= ["small.policy", "medium.policy", "large.policy"]
for i=3:3
    compute(inputfilename[i], outputfilename[i])
end

# Congrats. You've reached the end! Go google puppy pictures now.
 # include("project2.jl")
