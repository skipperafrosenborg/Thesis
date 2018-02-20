using CSV
using DataFrames
using Plots
#using Plotly
include("DataLoad.jl")

#Esben's path
cd("$(homedir())/Documents/GitHub/Thesis/Data")
path = "$(homedir())/Documents/GitHub/Thesis/Data"


#Skipper's path
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"

mainData = loadHousingData(path)
#mainData = loadCPUData(path)
#mainData = loadElevatorData(path)

dataSize = size(mainData)
colNames = names(mainData)

#Converting it into a datamatrix instead of a dataframe
combinedData = Array(mainData)
nRows = size(combinedData)[1]
nCols = size(combinedData)[2]

"""
Function that creates histogram of specified column ranges
input has to be a dataframe
"""
function createHistograms(inputData, startRange, endRange)
    mainData = copy(inputData)
    headers = names(mainData)
    for i = startRange:endRange
        display(plot(histogram(mainData[:,i], title = headers[i])))
    end
end

gr()
createHistograms(mainData, 1, 5)


#histogram(mainData[:,2], title = names(mainData)[2])
println("DONE")
