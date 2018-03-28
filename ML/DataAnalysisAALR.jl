using StatsBase
using DataFrames
using CSV
using Bootstrap #External packages, must be added
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Results/IndexData/AALR Test 1/"
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/ML"

stringArr = Array{String}(13*5)
for i = 1:13
    for j = 1:5
        stringArr[(i-1)*5+j] = "k =$i, gamma =$j"
    end
end

cd(path)
y_real = CSV.read("IndexData_realArray.CSV",
    delim = ',', nullable=false, header=["y_real"])
y_hat = CSV.read("IndexData_solution.CSV",
    delim = ',', nullable=false, header=stringArr)

y_real = y_real[1:246,:]
y_hat = y_hat[1:246,:]

nRows = size(y_hat)[1]
nCols = size(y_hat)[2]

classificationRate = Array{Float64}(nCols)
meanErr = Array{Float64}(nCols)
RMSE = Array{Float64}(nCols)
Rsquare = Array{Float64}(nCols)
SSTO = sum((y_real[i,1]-mean(y_real[:,1]))^2 for i=1:nRows)

for k = 1:13
    for g = 1:5
        currentCol = (k-1)*5+g
        # Calculate classifaction rate
        trueClass = 0
        for row = 1:nRows
            if sign(y_real[row,1]) == sign(y_hat[row,currentCol])
                trueClass += 1
            end
        end
        classificationRate[currentCol] = trueClass/nRows

        # Mean Error rate
        # Calculate RMSE
        # Calculate R^2
        errSum = 0
        errSquaredSum = 0
        for row = 1:nRows
            errSum += abs(y_real[row,1] - y_hat[row,currentCol])
            errSquaredSum += (y_real[row,1] - y_hat[row,currentCol])^2
        end
        meanErr[currentCol] = errSum/nRows
        RMSE[currentCol] = sqrt(errSquaredSum/nRows)
        Rsquare[currentCol] = 1- errSquaredSum/SSTO

        # Calculate insample error

        # Process data
    end
end

println("Classification: ",findmax(classificationRate))
println("OoS R²: ",findmax(Rsquare))
println("Mean Error: ",findmin(meanErr))
println("RMSE: ",findmin(RMSE))
