using StatsBase
using DataFrames
using CSV
using JuMP
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

for i = 1:16
    input = i
    gammaArr = linspace(0,4,16)

    dataSet = "VIXBestRegulation740-790/"*string(input)*"/"

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

    writedlm(path*"VIXTimeTABestRegForecast"*string(round(gammaArr[input],3))*".csv", forecastMatrix, ",")
    writedlm(path*"VIXTimeTABestReg"*string(round(gammaArr[input],3))*".csv",totalMatrix,",")

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
        writedlm(path*"weightsModel"*string(round(gammaArr[input],3))*".csv",weightMatrix[1:844,:],",")
    end
end


totalArray = zeros(844,34)
tempString = "Perfect,1/N"
for i = 1:16
    gammaArr = linspace(0,4,16)
    dataSet = "VIXBestRegulation740-790/"*string(i)*"/"
    path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/RFR/"*dataSet

    tempArr = CSV.read(path*"VIXTimeTABestReg"*string(round(gammaArr[i],3))*".csv", types = fill(Float64,4),
        datarow = 1, header=false, nullable=false)

    if i == 1
        totalArray[:,1:2] = Array{Float64}(tempArr)[:,1:2]
    end

    totalArray[:, 1+i*2:2+i*2] = Array{Float64}(tempArr)[:,3:4]

    tempString = tempString*",Return "*string(round(gammaArr[i],3))*",Pvalue "*string(round(gammaArr[i],3))
end
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/RFR/"*"VIXBestRegulation740-790/"


f = open(path*"TotalReturnArray.csv", "w")
write(f, tempString*"\n")
writecsv(f, totalArray)
close(f)
