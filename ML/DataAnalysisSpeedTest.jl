using StatsBase
using DataFrames
using CSV
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/Speed Benchmark Proffs/New/"
cd(path)

minValArr = zeros(6,3)
minIndexArr = zeros(6,3)

for i = 1:6
    lookUpVal = [1,101,201,301,401,501]
    normData = CSV.read(string(lookUpVal[i])*"bestNorm.csv",delim=",", nullable = false)
    warmstartData = CSV.read(string(lookUpVal[i])*"bestWarm.csv",delim=",", nullable = false)
    mathData = CSV.read(string(lookUpVal[i])*"bestMath.csv",delim=",", nullable = false)

    normMin = findmin(Array(normData))
    warmMin = findmin(Array(warmstartData))
    mathMin = findmin(Array(mathData))

    if normMin[2] > 13
        normIndex = normMin[2]%13
        normMinVal = normMin[1]
    else
        normIndex = normMin[2]
        normMinVal = normMin[1]
    end

    if warmMin[2] > 13
        warmIndex = warmMin[2]%13
        warmMinVal = warmMin[1]
    else
        warmIndex = warmMin[2]
        warmMinVal = warmMin[1]
    end

    if mathMin[2] > 13
        mathIndex = mathMin[2]%13
        mathMinVal = mathMin[1]
    else
        mathIndex = mathMin[2]
        mathMinVal = mathMin[1]
    end

    minValArr[i,1] = normMinVal
    minValArr[i,2] = warmMinVal
    minValArr[i,3] = mathMinVal

    minIndexArr[i,1] = normIndex
    minIndexArr[i,2] = warmIndex
    minIndexArr[i,3] = warmIndex
end
