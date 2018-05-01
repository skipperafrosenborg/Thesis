using StatsBase
using DataFrames
using CSV
include("SupportFunction.jl")
println("Leeeeroooy Jenkins")

industryArr = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
folderArr = [12, 24, 36, 48, 120, 240]

totalLog = zeros(6,10)

#Getting best data within a time period with all predictos
for a = 1:6
    folder = folderArr[a]
    for b = 1:10
        industry = industryArr[b]
        localBestClassRate = zeros(1,10)
        path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"*industry*"/"
        #path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
        cd(path)
        prediction = CSV.read(string(folder)*"-1/"*string(folder)*"_VIX_MacroTimeExpTA_Predicted.CSV", delim = ',', nullable=false)
        real = CSV.read(string(folder)*"-1/"*string(folder)*"_VIX_MacroTimeExpTA_real.CSV", delim = ',', nullable=false)
        predictionLength = 800
        for i = 1:10
            trueClassCounter = 0
            for j = 1:predictionLength
                predSign = sign(prediction[j,3+i])
                realSign = sign(real[j,4])
                if predSign == realSign
                    trueClassCounter += 1
                end
            end
            localBestClassRate[1,i] = trueClassCounter/800
        end
        totalLog[a,b] = maximum(localBestClassRate)
    end
end

#end

folder = 12
industry = "Durbl"
function logMaxes(industry, folder)
    path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"*industry*"/"
    cd(path)
    summary = CSV.read("Summary "*string(folder)*".CSV", delim = ',', nullable=false)
    classificationRate, index = findmax(summary[:,2])
    bestGamma = summary[index,6]
    ModelName = summary[index,1]
    path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"*industry*"/"*string(folder)*"-1/"
    cd(path)
    indicators = CSV.read(string(folder)*"_"*ModelName*"_Indi.CSV", delim = ',', nullable=false)
    indicatorArray = indicators[:,3+bestGamma]
    #=bestGamma = string(names(indicators)[3+bestGamma])
    if bestGamma == "0_1"
        bestGamma = 0
    else
        bestGamma = parse(Float64, bestGamma)
    end
    =#
    return bestGamma, classificationRate, ModelName, indicatorArray
end

bestGammaArr = zeros(6,10)
classificationRateArr = zeros(6,10)
ModelNameArr = Array{String}(6,10)
indicatorValues12 = zeros(1073, 10)
indicatorValues24 = zeros(1061, 10)
indicatorValues36 = zeros(1049, 10)
indicatorValues48 = zeros(1037, 10)
indicatorValues120= zeros( 965, 10)
indicatorValues240= zeros( 845, 10)

for i = 1:6
    for j = 1:10
        bestGammaArr[i,j], classificationRateArr[i,j], ModelNameArr[i,j], indicatorVals = logMaxes(industryArr[j],folderArr[i])
        if i == 1
            indicatorValues12[:,j] = indicatorVals
        end

        if i == 2
            indicatorValues24[:,j] = indicatorVals
        end

        if i == 3
            indicatorValues36[:,j] = indicatorVals
        end

        if i == 4
            indicatorValues48[:,j] = indicatorVals
        end

        if i == 5
            indicatorValues120[:,j] = indicatorVals
        end

        if i == 6
            indicatorValues240[:,j] = indicatorVals
        end
    end
end

industryArr = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
industryArrFile = Array{String}(1,10)
for i=1:10
    industryArrFile[1,i] = industryArr[i]
end

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
f = open(path*"GammaInfo.csv", "w")
writecsv(f, industryArrFile)
writecsv(f,bestGammaArr)
close(f)

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
f = open(path*"ClassificationInfo.csv", "w")
writecsv(f, industryArrFile)
writecsv(f,classificationRateArr)
close(f)

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
f = open(path*"ModelNameInfo.csv", "w")
writecsv(f, industryArrFile)
writecsv(f, ModelNameArr)
close(f)

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
f = open(path*"Indicators12.csv", "w")
writecsv(f, industryArrFile)
writecsv(f,indicatorValues12)
close(f)

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
f = open(path*"Indicators24.csv", "w")
writecsv(f, industryArrFile)
writecsv(f,indicatorValues24)
close(f)

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
f = open(path*"Indicators36.csv", "w")
writecsv(f, industryArrFile)
writecsv(f,indicatorValues36)
close(f)

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
f = open(path*"Indicators48.csv", "w")
writecsv(f, industryArrFile)
writecsv(f,indicatorValues48)
close(f)

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
f = open(path*"Indicators120.csv", "w")
writecsv(f, industryArrFile)
writecsv(f,indicatorValues120)
close(f)

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
f = open(path*"Indicators240.csv", "w")
writecsv(f, industryArrFile)
writecsv(f,indicatorValues240)
close(f)
