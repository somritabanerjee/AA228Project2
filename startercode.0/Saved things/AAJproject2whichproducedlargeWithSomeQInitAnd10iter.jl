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

function doQLearning(numStates,numActions,data,discount,alpha,numIter; Qinit=zeros(numStates,numActions))
    Q=Qinit;
    for k=1:numIter
        for i in 1:size(data, 1)
            s=data[i,1];
            a=data[i,2];
            r=data[i,3];
            sp=data[i,4];
            (maxValue,ind_max)=findmax(Q[sp,:]);
            Q[s,a] = Q[s,a] + alpha*(r + discount*maxValue - Q[s,a])
        end
    end
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
    # Q-learning
    discount_Q=0.95; # given to us!
    uniqueStates=unique(data[:,1]);
    numUniqueStates=size(uniqueStates, 1);
    alpha=1/numUniqueStates;
    @show numUniqueStates
    Qinit=ones(numStates,numActions);
    Qinit[150413,1]=10000000;
    Qinit[151203,1]=10000000;
    for ai=2:9
        Qinit[150413,ai]=0;
        Qinit[151203,ai]=0;
    end
    for si=1:numStates
        if (si==150413 || si==151203 || si==150211)
            break;
        end
        for ai=1:9
            if (ai>=5)
                Qinit[si,ai]=100;
            else
                Qinit[si,ai]=0;
            end
        end
    end
    numIter=10;
    Qvalues=doQLearning(numStates,numActions,data,discount_Q,alpha,numIter,Qinit=Qinit);
    optimalPolicy=findOptimalPolicyFromQ(Qvalues,numStates);
    return optimalPolicy;
end


# Create the files for submission
inputfilename = ["small.csv", "medium.csv", "large.csv"]
outputfilename= ["small.policy", "medium.policy", "large.policy"]
for i=3:3
    compute(inputfilename[i], outputfilename[i])
end

# Congrats. You've reached the end! Go google puppy pictures now.
 # include("project2.jl")
