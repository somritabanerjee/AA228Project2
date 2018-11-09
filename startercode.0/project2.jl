# using LightGraphs
using Printf
using DataFrames
using CSV
# using GraphPlot
using SpecialFunctions
using HDF5, JLD
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
            q=copy(Q[s,a]);
            Q[s,a] = q + alpha*(r + discount*maxValue - q);
            # Find the biggest change in Q values for this episode
            update=max(update, alpha*(r + discount*maxValue - q));
        end
        @show(update)
        numIter+=1;
    end
    @show(numIter)
    @show(update)
    return Q;
end

function doQLearningOnUniqueStates(numStates,numActions,svals,avals,rvals,spvals,discount,alpha,maxNumIter; Qinit=zeros(numStates,numActions), threshold=0.01)
    Q=Qinit;
    for k=1:maxNumIter
        for i in 1:size(svals, 1)
            s=svals[i];
            a=avals[i];
            r=rvals[i];
            sp=spvals[i];
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
    numIter=20;
    Qvalues=doQLearning(numStates,numActions,data,discount_Q,alpha,numIter);
    optimalPolicy=findOptimalPolicyFromQ(Qvalues,numStates);
    write_policy(optimalPolicy,"mediumNaiveQ20Iter.policy")

    # Let's try Q learning only on the states that are uniquely known
    # For the others let's find the nearest position with the same velocity and
    # assign the same Q value

    # Manipulating data
    println("Manipulating data");
    s=data[1];
    a=data[2];
    r=data[3];
    sp=data[4];
    numRows=size(data,1)

    # Computed and stored in file. Reading from file.
    uniqueStates=load("medium_uniqueStates.jld")["uniqueStates"]
    s_new_numbering=load("medium_s_new_numbering.jld")["s_new_numbering"]
    sp_new_numbering=load("medium_sp_new_numbering.jld")["sp_new_numbering"]
    # uniqueInitialStates=unique(s);
    # uniqueFinalStates=unique(sp);
    # uniqueStates=sort(union(uniqueInitialStates,uniqueFinalStates));
    # numUniqueStates=size(uniqueStates, 1);
    # s_new_numbering=zeros(Int64,numRows);
    # sp_new_numbering=zeros(Int64,numRows);
    # for i=1:numRows
    #     sp_new_numbering[i] = findfirst(fs -> fs==sp[i], uniqueStates);
    #     s_new_numbering[i] = findfirst(is -> is==s[i], uniqueStates);
    # end

    discount_Q=0.9;
    alpha=1;
    numIter=10;
    println("Starting Q Learning on unique states");
    Qvalues=doQLearningOnUniqueStates(numUniqueStates,numActions,s_new_numbering,a,r,sp_new_numbering,discount_Q,alpha,numIter);
    println("Completed Q Learning on unique states")
    QFull=zeros(numStates,numActions);
    QvaluesNonZeroIdx=findall(q -> q!=0, Qvalues);
    println("Finding Q for all states")
    for si=1:numStates
        for ai=1:numActions
            if si in uniqueStates
                idxInUniqueStates = findfirst(i -> i==si, uniqueStates);
                QFull[si,ai]=Qvalues[idxInUniqueStates,ai];
            else
                # Find nearest state with same action
                QvaluesWithSameAction=Qvalues[:,ai];
                # idx over uniqueStates
                QvaluesNonZeroIdx=findall(q -> q!=0, QvaluesWithSameAction);
                QvaluesNonZero=QvaluesWithSameAction[QvaluesNonZeroIdx];
                statesNonZero=uniqueStates[QvaluesNonZeroIdx];
                QValueOfStateClosest=findNearestQvalue(si,statesNonZero,QvaluesNonZero);
                QFull[si,ai]=QValueOfStateClosest;
            end
        end
        if (si>numStates/2)
            println("halfway done!");
        end
    end
    println("Found Q for all states")
    println("Computing optimal policy")
    optimalPolicy=findOptimalPolicyFromQ(QFull,numStates);
    write_policy(optimalPolicy, "mediumQUnique10AndClosest.policy")
    return optimalPolicy;
end

function findNearestQvalue(si,statesNonZero,QvaluesNonZero)
    # println("Finding nearest Q value for ")
    # @show si
    # The position bins can take on a value between 0 and 499
    # and the velocity bins can take on a value between 0 and 99.
    # 1+pos+500*vel gives the integer corresponding to a state with position pos and velocity vel
    posForThisState=mod(si,500)-1;
    velForThisState=floor(si/500);
    statesWithSameVel=collect(1+500*velForThisState:1+499+500*velForThisState);
    sameVelStatesNonZero=filter(m->in(m,statesWithSameVel),statesNonZero);
    statesWithSamePos=collect((1+posForThisState):500:(1+posForThisState+99*500));
    samePosStatesNonZero=filter(m->in(m,statesWithSamePos),statesNonZero);
    stateClosest=0;
    if (!isempty(sameVelStatesNonZero) && !isempty(samePosStatesNonZero))
        # find the state with same vel and closest position
        idx=findmin(abs.(sameVelStatesNonZero.-si))[2];
        stateClosestPos=sameVelStatesNonZero[idx];
        idx=findmin(abs.(samePosStatesNonZero.-si))[2];
        stateClosestVel=samePosStatesNonZero[idx];
        statesClosest=[stateClosestPos stateClosestVel];
        idx=findmin(abs.(statesClosest.-si))[2];
        stateClosest=statesClosest[idx];
    elseif  (!isempty(sameVelStatesNonZero))
        idx=findmin(abs.(sameVelStatesNonZero.-si))[2];
        stateClosestPos=sameVelStatesNonZero[idx];
        stateClosest=stateClosestPos;
    elseif (!isempty(samePosStatesNonZero))
        idx=findmin(abs.(samePosStatesNonZero.-si))[2];
        stateClosestVel=samePosStatesNonZero[idx];
        stateClosest=stateClosestVel;
    end
    # @show stateClosest
    # If no closest state found, return 0
    if (stateClosest==0)
        return 0;
    end
    idxOfStateClosest=findfirst(si -> si==stateClosest,statesNonZero);
    QValueOfStateClosest=QvaluesNonZero[idxOfStateClosest];
    return QValueOfStateClosest;
end

function constructFullOptimalPolicyFromSubset(optimalPolicyForUniqueStates, numStates, uniqueInitialStates)
    optimalPolicy=zeros(numStates,1);
    for i=1:numStates
        if (i in uniqueInitialStates)
            idxInUniqueStates = findfirst(is -> is==i, uniqueInitialStates);
            optimalPolicy[i]=optimalPolicyForUniqueStates[idxInUniqueStates];
        else
            # Pick an action that is not stationary, i.e. not 1-4
            randomNonStationaryAction=round(4*rand()+5);
            optimalPolicy[i]=randomNonStationaryAction;
        end
    end
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
    # For some reason, 30 gives us the best policy at convergence
    numIter=30;
    Qvalues=doQLearning(numStates,numActions,data,discount_Q,alpha,numIter,Qinit=Qinit);
    optimalPolicy=findOptimalPolicyFromQ(Qvalues,numStates);
    write_policy(optimalPolicy,"largeQ30IterUsingOldCode.policy")
    return optimalPolicy;
end

# Create the files for submission
inputfilename = ["small.csv", "medium.csv", "large.csv"]
outputfilename= ["small.policy", "medium.policy", "large.policy"]
for i=2:2
    compute(inputfilename[i], outputfilename[i])
end

# Congrats. You've reached the end! Go google puppy pictures now.
 # include("project2.jl")
