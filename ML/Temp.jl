using JuMP
using StatsBase
using DataFrames
using CSV
using JLD
using Plots
include("SupportFunction.jl")
println("Leeeeroooy Jenkins")

industry = "NoDur"
folder = "12"
folderArr = [12, 24, 36, 48, 120, 240]
industryArr = ["Durbl", "Enrgy", "HiTec", "Hlth", "Manuf", "NoDur", "Other", "Shops", "Telcm", "Utils"]
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"*industry*"/"*folder*"-1/"

# Common parameters within 1 industry over time but from different datasets. Loop over datasets
# Hypothesis is that some paraemeters are always important no matter what the expansions of the dataset we are dealing with.
j=1
trainingSize=240
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
#Fetch summary file
println("Predicting industry ",industryArr[j])
summary = CSV.read(path*industryArr[j]*"/"*"Summary "*string(trainingSize)*".csv", nullable=false)

testMat = Array{Array{Float64}}(24)

for i = 1:24
    #Fetch dataset from summary
    fileName = summary[i,1]

    #Fetch best gamma from summary, based on R^2
    bestGamma = summary[i,7]

    nPredictionTerms = [10, 18, 40, 48, 130, 138, 520, 528, 27, 35, 108, 116, 351, 359, 1404, 1412, 28, 36, 112, 120, 364, 372, 1456, 1464]

    #Fetch bSolved matrix
    testMat[i] = bSolveMat = CSV.read(path*industryArr[j]*"/"*string(trainingSize)*"-1/240_"*fileName*"_bMatrix240_"*string(bestGamma)*".csv",
        nullable=false, header=false, datarow=1, types=fill(Float64,nPredictionTerms[i]))
end

#=Filenames:
1,     2     3      4       5       6       7        8
Clean, TA,   Exp,   ExpTA,  Time,   TimeTA, TimeExp, TimeExpTA
=#

### Raw datasets ###
# Check for all datasets if the raw input are used
# Are any of the raw inputs always used?
# How many percent of the 8 datasets uses the prediction term in period t
rawInputUsage()
#NoDur is clearly the strongest predictor and also shows the strongest prediction power in recent terms
function rawInputUsage()
    println("Predicting dataset ",industryArr[j])
    industryCounter = zeros(845,10)
    for t = 1:845
        for curIndustry = 1:10
            for dataset = 1:10
                if testMat[dataset][t,curIndustry] != 0
                    industryCounter[t,curIndustry] += 1
                end
            end
        end
    end
    industryCounter = industryCounter/10

    avgUsage = mean(industryCounter,1)
    for i=1:10
        println(industryArr[i], " is used \t", round(avgUsage[i],3),"% of the time across all datasets")
    end
    println()

    avgUsageIfActive = zeros(1,10)
    for curIndustry = 1:10
        iter = 0
        for t = 1:845
            if industryCounter[t,curIndustry] != 0
                iter += 1
                avgUsageIfActive[1,curIndustry] += industryCounter[t,curIndustry]
            end
        end
        avgUsageIfActive[1,curIndustry] = avgUsageIfActive[1,curIndustry]/iter
    end

    for i=1:10
        println(industryArr[i], " is used \t", round(avgUsageIfActive[i],3),"% of the time across all datasets when it's active")
    end

    #Create label rowvector
    industryArr2 = Array{String}(1,10)
    for i = 1:10
        industryArr2[1,i] = industryArr[i]
    end

    # Graph it
    plotlyjs()
    plot(industryCounter, title="Industry Overview", linewidth=2, label=industryArr2, yaxis=("% used in period t",(0,1),0:0.1:1),xaxis=("Months",0:100:845))
end

rawNonLinearUsage()
# Are non linear transformation used?
# Check how many and which non linear transformation is used.
# Only x^2, log(x), sqrt(x)?
function rawNonLinearUsage()
    avgUsage = zeros(1,40)

    #Load Exp dataset
    for i=1:40
        avgUsage[1,i] = countnz(testMat[3][:,i])/845
    end

    rawAvgUse = mean(avgUsage[1:10])
    SquaredAvgUse = mean(avgUsage[11:20])
    LogAvgUse = mean(avgUsage[21:30])
    SqrtAvgUse = mean(avgUsage[31:40])

    println("Raw usage average usage ",round(rawAvgUse,3))
    for i=1:10
        println(industryArr[i],"is used ", round(avgUsage[i],3),"% on average")
    end

    println("\nSquared usage average usage ", round(SquaredAvgUse,3))
    for i=11:20
        println(industryArr[i-10],"is used ", round(avgUsage[i],3),"% on average")
    end

    println("\nLog usage average usage ", round(LogAvgUse,3))
    for i=21:30
        println(industryArr[i-20],"is used ", round(avgUsage[i],3),"% on average")
    end

    println("\nSqrt usage average usage ", round(SqrtAvgUse,3))
    for i=31:40
        println(industryArr[i-30],"is used ", round(avgUsage[i],3),"% on average")
    end
end

# Are time series used?
    # Are the time series usage based on a single period (t-2, t-3... ) across all industries
    # or a certain industry with multiple timeseries.
function rawTimeSeriesUsage()
    avgUsage = zeros(10,13)

    #Load Exp dataset
    for industryIndex=1:10
        for i=1:13
            avgUsage[industryIndex,i] = countnz(testMat[5][:,industryIndex+(i-1)*10])/845
        end
    end

    for i=0:12
        println("Time series ", i, " is on average used ", round(mean(avgUsage[:,i+1]),3))
    end

    for industryIndex=1:10
        for i=0:12
            println("Time series ", i, " is on average used ", round(avgUsage[industryIndex, i+1],3), " in industry ",industryArr[industryIndex])
        end
        println()
    end
end

# Are TA used?
    # Are the TA input used?
function rawTAUsage()
    avgUsage = zeros(8)

    for dataset=[2,6,8]
        #Load TA dataset
        for i=1:8
            avgUsage[i] = countnz(testMat[dataset][:,end-8+i:end-8+i])/845
        end

        for i=1:8
            println("TA  ", i, " is on average used ", round(avgUsage[i],3), " in dataset ",dataset)
        end
        println("")
    end
end

# Are non linear transformation of times series used?
    # Important!
function rawTimeExpUsage()
    avgUsage = zeros(520)

    for i=1:520
        avgUsage[i] = countnz(testMat[7][:,i])
    end

    realmean = mean(avgUsage)
    degOfFree = 519

    stdev = std(avgUsage)

    for i=1:520
        zscore = (avgUsage[i]-realmean)/stdev
        if zscore > 1.965 || zscore < -1.965
            println("Paremeter ", i, " is significant")
        end
    end

end


#Are the combination of parameters used within the past 10 years seen earlier?

# Macro dataset
    # Are non linear transformation used?
    # Are time series used?
    # Are TA used?
    # Are non linear transformation of times series used?
    # Are any of the raw inputs always used?
    # Is VIX actually used?


# Common parameters within 1 dataset across industries over time. Loop over industries
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
#Fetch summary file
summary = CSV.read(path*industryArr[i]*"/"*"Summary "*string(trainingSize)*".csv", nullable=false)

#Fetch dataset from summary
fileName = summary[i,1]

#Fetch best gamma from summary, based on R^2
bestGamma = summary[i,7]

#Fetch bSolved matrix
CSV.read(path*industryArr[i]*"/"*string(trainingSize)*"-1/240_"*fileName*"_bMatrix240_"*bestGamma*".csv", nullable=false, header=false, datarow=1)
