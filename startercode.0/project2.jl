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

function findPolicyForSmall(data)
    numStates=100;
    numActions=4;
    uniformPolicy=ones(numStates,1);
    return uniformPolicy;
end

function findPolicyForMedium(data)
    numStates=50000;
    numActions=7;
    uniformPolicy=ones(numStates,1);
    return uniformPolicy;
end

function findPolicyForLarge(data)
    numStates=312020;
    numActions=9;
    uniformPolicy=ones(numStates,1);
    return uniformPolicy;
end


# Create the files for submission
inputfilename = ["small.csv", "medium.csv", "large.csv"]
outputfilename= ["small.policy", "medium.policy", "large.policy"]
for i=1:3
    compute(inputfilename[i], outputfilename[i])
end

# Congrats. You've reached the end! Go google puppy pictures now.
 # include("project2.jl")
