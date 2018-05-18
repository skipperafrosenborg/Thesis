using StatsBase
using DataFrames
using CSV
using JuMP
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

dataSet = "VIXTimeTA2.4/"

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/RFR/"*dataSet
cd(path)

totalMatrix = zeros(844,12)

for i = 0:83
    tempMatrix = CSV.read(path*string(i)*"_returnPvalueOutcome1to200.csv",nullable=false, header=false, datarow=1)
    if i == 0
        totalMatrix = Array(tempMatrix)
    else
        totalMatrix[6:end,2:end] += Array(tempMatrix[6:end,2:end])
    end
end

mean(totalMatrix[:,end-4:end],1)

writedlm(path*"VIXTimeTA2_4.csv",totalMatrix,",")
