using StatsBase
using DataFrames
using CSV
using JuMP
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

dataSet = "VIXTimeTA4DynamicBigM/"

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/RFR/"*dataSet
cd(path)

totalMatrix = zeros(844,12)
forecastMatrix = zeros(844,22)

for i = 0:84
    #println(i)
    tempMatrix = CSV.read(path*string(i)*"_returnPvalueOutcome1to200.csv",nullable=false, header=false, datarow=1, types=fill(Float64,4))
    tempForecast = CSV.read(path*string(i)*"_forecastsAndErrors1to200.csv",nullable=false, header=false, datarow=1, types=fill(Float64,22))
    if i == 0
        totalMatrix = Array(tempMatrix)
        forecastMatrix = Array(tempForecast)
    else
        totalMatrix[6:end,2:end] += Array(tempMatrix[6:end,2:end])
        forecastMatrix[6:end,1:end] += Array(tempForecast[6:end,1:end])
    end
end

#mean(totalMatrix[:,end-4:end],1)

writedlm(path*"VIXTimeTA4DynamicBigMForecast.csv", forecastMatrix, ",")
writedlm(path*"VIXTimeTA4DynamicBigM.csv",totalMatrix,",")

weightMatrix = zeros(846,11)
for i = 1:1
    weightMatrix = zeros(846,11)
    for j = 0:84
        tempMatrix = CSV.read(path*"wightsCEO_Model"*string(i)*"_"*string(j)*".csv",nullable=false, header=false, datarow=1, types=fill(Float64,11))
        if j == 0
            weightMatrix = Array(tempMatrix)
        else
            weightMatrix += Array(tempMatrix)
        end
    end
    writedlm(path*"weightsModel"*string(i)*".csv",weightMatrix[1:844,:],",")
end
