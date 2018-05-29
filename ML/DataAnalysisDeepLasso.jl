using JuMP
using StatsBase
using DataFrames
using CSV
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
j=2
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
getRaw()
function getRaw()
    avgUsage = zeros(10,10)
    signifMat = Array{String}(10,10)
    for j = 1:10
        println("Predicting industry ", industryArr[j])
        summary = CSV.read(path*industryArr[j]*"/"*"Summary "*string(trainingSize)*".csv", nullable=false)
        testMat = zeros(845,10)
        for i = 1:1


            #Fetch dataset from summary
            fileName = summary[i,1]

            #Fetch best gamma from summary, based on R^2
            bestGamma = summary[i,7]

            nPredictionTerms = [10, 18, 40, 48, 130, 138, 520, 528, 27, 35, 108, 116, 351, 359, 1404, 1412, 28, 36, 112, 120, 364, 372, 1456, 1464]

            #Fetch bSolved matrix
            tempMat = bSolveMat = CSV.read(path*industryArr[j]*"/"*string(trainingSize)*"-1/240_"*fileName*"_bMatrix240_"*string(bestGamma)*".csv",
                nullable=false, header=false, datarow=1, types=fill(Float64,nPredictionTerms[i]))

            testMat += Array(tempMat)[:,1:10]
            # 10 industries predicting 10 industries
        end
        testMat = testMat/1
        testMat[find(testMat)] = 1

        for k = 1:10
            avgUsage[j,k] = countnz(testMat[:,k])/845
        end
    end

    realmean = mean(avgUsage)
    realStd = std(avgUsage)
    for j = 1:10
        for k = 1:10
            z_score = (avgUsage[j,k]-realmean)/(realStd)
            if z_score > 2.326347874
                signifMat[j,k] = "***"
            elseif z_score > 1.644853627
                signifMat[j,k] = "**"
            elseif z_score > 1.281551566
                signifMat[j,k] = "*"
            else
                signifMat[j,k] = " "
            end
        end
    end

    println(signifMat)
    println(avgUsage)

    # 1.959964 --> 0.95
    # 2.575829 --> 0.99
    # 3.290527 --> 0.999
    writedlm(path*"ParametersAnalysis/industryRawAverageSignif.csv",signifMat,",")
    writedlm(path*"ParametersAnalysis/industryRawAverageUsage.csv",avgUsage,",")
end

getRawCorrectPredictions()
function getRawCorrectPredictions()
    avgUsage = zeros(10,10)
    avgCorrectUsage = zeros(10,10)
    signifMat = Array{String}(10,10)
    for j = 1:10
        println("Predicting industry ", industryArr[j])
        summary = CSV.read(path*industryArr[j]*"/"*"Summary "*string(trainingSize)*".csv", nullable=false)
        testMat = zeros(845,10)
        for i = 1:1
            #Fetch dataset from summary
            fileName = summary[i,1]

            #Fetch best gamma from summary, based on R^2
            bestGamma = summary[i,7]

            nPredictionTerms = [10, 18, 40, 48, 130, 138, 520, 528, 27, 35, 108, 116, 351, 359, 1404, 1412, 28, 36, 112, 120, 364, 372, 1456, 1464]

            #Fetch bSolved matrix
            tempMat = CSV.read(path*industryArr[j]*"/"*string(trainingSize)*"-1/240_"*fileName*"_bMatrix240_"*string(bestGamma)*".csv",
                nullable=false, header=false, datarow=1, types=fill(Float64,nPredictionTerms[i]))

            realValues = CSV.read(path*industryArr[j]*"/"*string(trainingSize)*"-1/240_"*fileName*"_real.csv",
                nullable=false, types=[Int64, Int64, Int64, Float64])

            predictedValues = CSV.read(path*industryArr[j]*"/"*string(trainingSize)*"-1/240_"*fileName*"_predicted.csv",
                nullable=false, types=vcat([Int64, Int64, Int64],fill(Float64,10)))


            for t = 1:845
                if sign(predictedValues[t,3+bestGamma]) == sign(realValues[t,4])
                    for pt = 1:10
                        if tempMat[t,pt] != 0
                            avgCorrectUsage[j,pt] += 1
                        end
                    end
                end
            end

            testMat += Array(tempMat)[:,1:10]
        end

        testMat = testMat/1
        testMat[find(testMat)] = 1

        for k = 1:10
            avgUsage[j,k] = countnz(testMat[:,k])/845
        end
    end

    avgCorrectUsage = avgCorrectUsage/845

    avgCorrectUsagePercent = zeros(10,10)
    for i=1:10
        for j=1:10
            avgCorrectUsagePercent[i,j] = avgCorrectUsage[i,j]/avgUsage[i,j]
        end
    end
    # 1.959964 --> 0.95
    # 2.575829 --> 0.99
    # 3.290527 --> 0.999
    writedlm(path*"ParametersAnalysis/RawIndustryCorrectUsage.csv",avgCorrectUsagePercent,",")
end

getMacroTimeTA()
function getMacroTimeTA()
    path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
    avgUsage = zeros(10,35)
    signifMat = Array{String}(10,35)
    for j = 1:10
        println("Predicting industry ", industryArr[j])
        summary = CSV.read(path*industryArr[j]*"/"*"Summary "*string(trainingSize)*".csv", nullable=false)
        testMat = zeros(845,359)
        for i = 14
            #Fetch dataset from summary
            fileName = summary[i,1]

            #Fetch best gamma from summary, based on R^2
            bestGamma = summary[i,7]

            nPredictionTerms = [10, 18, 40, 48, 130, 138, 520, 528, 27, 35, 108, 116, 351, 359, 1404, 1412, 28, 36, 112, 120, 364, 372, 1456, 1464]

            #Fetch bSolved matrix
            tempMat = CSV.read(path*industryArr[j]*"/"*string(trainingSize)*"-1/240_"*fileName*"_bMatrix240_"*string(bestGamma)*".csv",
                nullable=false, header=false, datarow=1, types=fill(Float64,nPredictionTerms[i]))

            testMat += Array(tempMat)
            # 10 industries predicting 10 industries
        end
        testMat[find(testMat)] = 1

        for k = 1:27
            for t = 0:12
                avgUsage[j,k] += countnz(testMat[:,k+27*t])/845
            end
            avgUsage[j,k] = avgUsage[j,k]/13
        end

        for k = 28:35
            avgUsage[j,k] = countnz(testMat[:, end-35+k])/845
        end
    end

    realmean = mean(avgUsage)
    realStd = std(avgUsage)
    for j = 1:10
        for k = 1:35
            z_score = (avgUsage[j,k]-realmean)/(realStd)
            if z_score > 2.326347874
                signifMat[j,k] = "***"
            elseif z_score > 1.644853627
                signifMat[j,k] = "**"
            elseif z_score > 1.281551566
                signifMat[j,k] = "*"
            else
                signifMat[j,k] = " "
            end
        end
    end

    println(signifMat)
    println(avgUsage)

    writedlm(path*"ParametersAnalysis/MacroTimeTAAverageSignif.csv",signifMat,",")
    writedlm(path*"ParametersAnalysis/MacroTimeTAAverageUsage.csv",avgUsage,",")
end

getMacroTimeTAMostUsedVariables()
function getMacroTimeTAMostUsedVariables()
    path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
    avgUsage = zeros(13,27)
    signifMat = Array{String}(13,27)

    allMat = zeros(845,359)
    for j = 1:10
        println("Predicting industry ", industryArr[j])
        summary = CSV.read(path*industryArr[j]*"/"*"Summary "*string(trainingSize)*".csv", nullable=false)
        testMat = zeros(845,359)
        for i = 14
            #Fetch dataset from summary
            fileName = summary[i,1]

            #Fetch best gamma from summary, based on R^2
            bestGamma = summary[i,7]

            nPredictionTerms = [10, 18, 40, 48, 130, 138, 520, 528, 27, 35, 108, 116, 351, 359, 1404, 1412, 28, 36, 112, 120, 364, 372, 1456, 1464]

            #Fetch bSolved matrix
            tempMat = CSV.read(path*industryArr[j]*"/"*string(trainingSize)*"-1/240_"*fileName*"_bMatrix240_"*string(bestGamma)*".csv",
                nullable=false, header=false, datarow=1, types=fill(Float64,nPredictionTerms[i]))

            testMat += Array(tempMat)
            # 10 industries predicting 10 industries
        end
        testMat[find(testMat)] = 1

        allMat += testMat

        #find(x -> x>0.9, mean(testMat,1))

        for k = 1:27
            for t = 0:12
                avgUsage[t+1,k] += countnz(testMat[:,k+27*t])/845
            end
        end
    end

    allMat = allMat/10
    allMatMean = mean(allMat,1)
    mean(allMatMean)
    std(allMatMean)

    n = 500

    stepVector = zeros(n+1,3)
    for i = 0:n
        stepVector[i+1,1] = i/n
        x = length(find(x-> x <= i/n, mean(allMat,1)))
        stepVector[i+1,3] = x
        y = length(find(x-> x >= i/n && x < (i+1)/n, mean(allMat,1)))
        stepVector[i+1,2] = y
    end
    writedlm(path*"ParametersAnalysis/MacroTimeTAParameterDistribution.CSV",stepVector,",")

    x = find(x-> x< 0.05 && x>0.01, mean(allMat,1))
    x = find(x-> x> 0.5, mean(allMat,1))

    for i=x
        if floor(i/27) < 13
            println("T",Int64(floor(i/27)),", ",i%27)
        else
            println("TA term ",i%27)
        end
    end
    mean(allMat,1)[x]
end

getVixMacroTimeTA()
function getVixMacroTimeTA()
    path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
    avgUsage = zeros(10,36)
    signifMat = Array{String}(10,36)
    for j = 1:10
        println("Predicting industry ", industryArr[j])
        summary = CSV.read(path*industryArr[j]*"/"*"Summary "*string(trainingSize)*".csv", nullable=false)
        testMat = zeros(845,372)
        for i = 22
            #Fetch dataset from summary
            fileName = summary[i,1]

            #Fetch best gamma from summary, based on R^2
            bestGamma = summary[i,7]

            nPredictionTerms = [10, 18, 40, 48, 130, 138, 520, 528, 27, 35, 108, 116, 351, 359, 1404, 1412, 28, 36, 112, 120, 364, 372, 1456, 1464]

            #Fetch bSolved matrix
            tempMat = CSV.read(path*industryArr[j]*"/"*string(trainingSize)*"-1/240_"*fileName*"_bMatrix240_"*string(bestGamma)*".csv",
                nullable=false, header=false, datarow=1, types=fill(Float64,nPredictionTerms[i]))

            testMat += Array(tempMat)
            # 10 industries predicting 10 industries
        end
        testMat[find(testMat)] = 1

        for k = 1:28
            for t = 0:12
                avgUsage[j,k] += countnz(testMat[:,k+27*t])/845
            end
        end

        for k = 29:36
            avgUsage[j,k] = countnz(testMat[:, end-36+k])/845
        end
    end

    realmean = mean(avgUsage)
    realStd = std(avgUsage)
    for j = 1:10
        for k = 1:10
            z_score = (avgUsage[j,k]-realmean)/(realStd)
            if z_score > 2.326347874
                signifMat[j,k] = "***"
            elseif z_score > 1.644853627
                signifMat[j,k] = "**"
            elseif z_score > 1.281551566
                signifMat[j,k] = "*"
            else
                signifMat[j,k] = " "
            end
        end
    end

    println(signifMat)
    println(avgUsage)

    writedlm(path*"ParametersAnalysis/VIXMacroTimeTAAverageSignif.csv",signifMat,",")
    writedlm(path*"ParametersAnalysis/VIXMacroTimeTAAverageUsage.csv",avgUsage,",")
end

getVixPeriodMacroTimeTA()
function getVixPeriodMacroTimeTA()
    path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
    avgUsage = zeros(10,36)
    signifMat = Array{String}(10,36)
    for j = 1:10
        println("Predicting industry ", industryArr[j])
        summary = CSV.read(path*industryArr[j]*"/"*"Summary "*string(trainingSize)*".csv", nullable=false)
        testMat = zeros(323,372)
        for i = 22
            #Fetch dataset from summary
            fileName = summary[i,1]

            #Fetch best gamma from summary, based on R^2
            bestGamma = summary[i,7]

            nPredictionTerms = [10, 18, 40, 48, 130, 138, 520, 528, 27, 35, 108, 116, 351, 359, 1404, 1412, 28, 36, 112, 120, 364, 372, 1456, 1464]

            #Fetch bSolved matrix
            tempMat = CSV.read(path*industryArr[j]*"/"*string(trainingSize)*"-1/240_"*fileName*"_bMatrix240_"*string(bestGamma)*".csv",
                nullable=false, header=false, datarow=1, types=fill(Float64,nPredictionTerms[i]))

            testMat += Array(tempMat)[end-322:end,:]
            # 10 industries predicting 10 industries
        end
        testMat[find(testMat)] = 1

        for k = 1:28
            for t = 0:12
                avgUsage[j,k] += countnz(testMat[:,k+27*t])/323
            end
        end

        for k = 29:36
            avgUsage[j,k] = countnz(testMat[:, end-36+k])/323
        end
    end

    realmean = mean(avgUsage)
    realStd = std(avgUsage)
    for j = 1:10
        for k = 1:36
            z_score = (avgUsage[j,k]-realmean)/(realStd)
            if z_score > 3.090232306
                signifMat[j,k] = "***"
            elseif z_score > 2.326347874
                signifMat[j,k] = "**"
            elseif z_score > 1.644853627
                signifMat[j,k] = "*"
            else
                signifMat[j,k] = " "
            end
        end
    end

    println(signifMat)
    println(avgUsage)

    writedlm(path*"ParametersAnalysis/VIXPeriodMacroTimeTAAverageSignif.csv",signifMat,",")
    writedlm(path*"ParametersAnalysis/VIXPeriodMacroTimeTAAverageUsage.csv",avgUsage,",")
end

getVixMacroTimeTAMostUsedVariables()
function getVixMacroTimeTAMostUsedVariables()
    path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
    avgUsage = zeros(13,28)
    signifMat = Array{String}(13,28)

    allMat = zeros(845,372)
    for j = 1:10
        println("Predicting industry ", industryArr[j])
        summary = CSV.read(path*industryArr[j]*"/"*"Summary "*string(trainingSize)*".csv", nullable=false)
        testMat = zeros(845,372)
        for i = 22
            #Fetch dataset from summary
            fileName = summary[i,1]

            #Fetch best gamma from summary, based on R^2
            bestGamma = summary[i,7]

            nPredictionTerms = [10, 18, 40, 48, 130, 138, 520, 528, 27, 35, 108, 116, 351, 359, 1404, 1412, 28, 36, 112, 120, 364, 372, 1456, 1464]

            #Fetch bSolved matrix
            tempMat = CSV.read(path*industryArr[j]*"/"*string(trainingSize)*"-1/240_"*fileName*"_bMatrix240_"*string(bestGamma)*".csv",
                nullable=false, header=false, datarow=1, types=fill(Float64,nPredictionTerms[i]))

            testMat += Array(tempMat)
            # 10 industries predicting 10 industries
        end
        testMat[find(testMat)] = 1

        allMat += testMat

        #find(x -> x>0.9, mean(testMat,1))

        for k = 1:28
            for t = 0:12
                avgUsage[t+1,k] += countnz(testMat[:,k+27*t])/845
            end
        end
    end

    allMat = allMat/10
    allMatMean = mean(allMat,1)
    mean(allMatMean)
    std(allMatMean)

    stepVector = zeros(10001,3)
    for i = 0:10000
        stepVector[i+1,1] = i/10000
        x = length(find(x-> x <= i/10000, mean(allMat,1)))
        stepVector[i+1,3] = x
        y = length(find(x-> x >= i/10000 && x < (i+1)/10000, mean(allMat,1)))
        stepVector[i+1,2] = y
    end
    writedlm(path*"ParametersAnalysis/VIXMacroTimeTAParameterDistribution.CSV",stepVector,",")

    x = find(x-> x< 0.05 && x>0.01, mean(allMat,1))
    x = find(x-> x> 0.25, mean(allMat,1))

    for i=x
        if floor(i/28) < 13
            println("T",Int64(floor(i/28)),", ",i%28)
        else
            println("TA term ",i%28)
        end
    end
    mean(allMat,1)[x]
end











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

getRawTime()
function getRawTime()
    avgUsage = zeros(10,10)
    signifMat = Array{String}(10,10)
    for j = 1:10
        println("Predicting industry ", industryArr[j])
        summary = CSV.read(path*industryArr[j]*"/"*"Summary "*string(trainingSize)*".csv", nullable=false)
        testMat = zeros(845,10)
        for i = 5:5
            #Fetch dataset from summary
            fileName = summary[i,1]

            #Fetch best gamma from summary, based on R^2
            bestGamma = summary[i,7]

            nPredictionTerms = [10, 18, 40, 48, 130, 138, 520, 528, 27, 35, 108, 116, 351, 359, 1404, 1412, 28, 36, 112, 120, 364, 372, 1456, 1464]

            #Fetch bSolved matrix
            tempMat = bSolveMat = CSV.read(path*industryArr[j]*"/"*string(trainingSize)*"-1/240_"*fileName*"_bMatrix240_"*string(bestGamma)*".csv",
                nullable=false, header=false, datarow=1, types=fill(Float64,nPredictionTerms[i]))

            testMat = Array(tempMat)
            # 10 industries predicting 10 industries
        end
        countnz(testMat[8,:])
        testMat[find(testMat)] = 1

        for k = 1:10
            avgUsage[j,k] = countnz(testMat[:,[k,k+10,k+20,k+30,k+40,k+50,k+60,k+70,k+80,k+90,k+100,k+110,k+120]])/845
        end
    end

    realmean = mean(avgUsage)
    realStd = std(avgUsage)
    for j = 1:10
        for k = 1:10
            z_score = (avgUsage[j,k]-realmean)/(realStd)
            if z_score > 3.090232306
                signifMat[j,k] = "***"
            elseif z_score > 2.326347874
                signifMat[j,k] = "**"
            elseif z_score > 1.644853627
                signifMat[j,k] = "*"
            else
                signifMat[j,k] = " "
            end
        end
    end

    println(signifMat)
    println(avgUsage)

    # 1.959964 --> 0.95
    # 2.575829 --> 0.99
    # 3.290527 --> 0.999
    writedlm(path*"ParametersAnalysis/RawTimeIndustryAverageSignif.csv",signifMat,",")
    writedlm(path*"ParametersAnalysis/RawTimeIndustryAverageUsage.csv",avgUsage,",")
end

getRawExp()
function getRawExp()
    avgUsage = zeros(10,10)
    signifMat = Array{String}(10,10)
    for j = 1:10
        println("Predicting industry ", industryArr[j])
        summary = CSV.read(path*industryArr[j]*"/"*"Summary "*string(trainingSize)*".csv", nullable=false)
        testMat = zeros(845,10)
        for i = 3
            #Fetch dataset from summary
            fileName = summary[i,1]

            #Fetch best gamma from summary, based on R^2
            bestGamma = summary[i,7]

            nPredictionTerms = [10, 18, 40, 48, 130, 138, 520, 528, 27, 35, 108, 116, 351, 359, 1404, 1412, 28, 36, 112, 120, 364, 372, 1456, 1464]

            #Fetch bSolved matrix
            tempMat = bSolveMat = CSV.read(path*industryArr[j]*"/"*string(trainingSize)*"-1/240_"*fileName*"_bMatrix240_"*string(bestGamma)*".csv",
                nullable=false, header=false, datarow=1, types=fill(Float64,nPredictionTerms[i]))

            testMat = Array(tempMat)
            # 10 industries predicting 10 industries
        end
        countnz(testMat[8,:])
        testMat[find(testMat)] = 1

        for k = 1:10
            avgUsage[j,k] = countnz(testMat[:,[k,k+10,k+20,k+30]])/845
        end
    end

    realmean = mean(avgUsage)
    realStd = std(avgUsage)
    for j = 1:10
        for k = 1:10
            z_score = (avgUsage[j,k]-realmean)/(realStd)
            if z_score > 3.090232306
                signifMat[j,k] = "***"
            elseif z_score > 2.326347874
                signifMat[j,k] = "**"
            elseif z_score > 1.644853627
                signifMat[j,k] = "*"
            else
                signifMat[j,k] = " "
            end
        end
    end

    println(signifMat)
    println(avgUsage)

    # 1.959964 --> 0.95
    # 2.575829 --> 0.99
    # 3.290527 --> 0.999
    writedlm(path*"ParametersAnalysis/RawExpIndustryAverageSignif.csv",signifMat,",")
    writedlm(path*"ParametersAnalysis/RawExpIndustryAverageUsage.csv",avgUsage,",")
end

getRawTimeExp()
function getRawExp()
    avgUsage = zeros(10,10)
    signifMat = Array{String}(10,10)
    for j = 1:10
        println("Predicting industry ", industryArr[j])
        summary = CSV.read(path*industryArr[j]*"/"*"Summary "*string(trainingSize)*".csv", nullable=false)
        testMat = zeros(845,10)
        for i = 7
            #Fetch dataset from summary
            fileName = summary[i,1]

            #Fetch best gamma from summary, based on R^2
            bestGamma = summary[i,7]

            nPredictionTerms = [10, 18, 40, 48, 130, 138, 520, 528, 27, 35, 108, 116, 351, 359, 1404, 1412, 28, 36, 112, 120, 364, 372, 1456, 1464]

            #Fetch bSolved matrix
            tempMat = bSolveMat = CSV.read(path*industryArr[j]*"/"*string(trainingSize)*"-1/240_"*fileName*"_bMatrix240_"*string(bestGamma)*".csv",
                nullable=false, header=false, datarow=1, types=fill(Float64,nPredictionTerms[i]))

            testMat = Array(tempMat)[:,:]
            # 10 industries predicting 10 industries
        end
        countnz(testMat[8,:])
        testMat[find(testMat)] = 1

        lRange = vcat(Array{Int64}(15:15+11),Array{Int64}(28:28+11),Array{Int64}(41:41+11))


        for k = 1:10
            for l = lRange
                avgUsage[j,k] += countnz(testMat[:,[k+10*(l-1)]])/845
            end
        end
    end

    realmean = mean(avgUsage)
    realStd = std(avgUsage)
    for j = 1:10
        for k = 1:10
            z_score = (avgUsage[j,k]-realmean)/(realStd)
            if z_score > 3.090232306
                signifMat[j,k] = "***"
            elseif z_score > 2.326347874
                signifMat[j,k] = "**"
            elseif z_score > 1.644853627
                signifMat[j,k] = "*"
            else
                signifMat[j,k] = " "
            end
        end
    end

    println(signifMat)
    println(avgUsage)

    # 1.959964 --> 0.95
    # 2.575829 --> 0.99
    # 3.290527 --> 0.999
    writedlm(path*"ParametersAnalysis/RawTimeExpRawTimeExpIndustryAverageSignif.csv",signifMat,",")
    writedlm(path*"ParametersAnalysis/RawTimeExpRawTimeExpIndustryAverageUsage.csv",avgUsage,",")
end

getMacro()
function getMacro()
    avgUsage = zeros(10,17)
    signifMat = Array{String}(10,17)
    for j = 1:10
        println("Predicting industry ", industryArr[j])
        summary = CSV.read(path*industryArr[j]*"/"*"Summary "*string(trainingSize)*".csv", nullable=false)
        testMat = zeros(845,17)
        for i = 17-8:24-8


            #Fetch dataset from summary
            fileName = summary[i,1]

            #Fetch best gamma from summary, based on R^2
            bestGamma = summary[i,7]

            nPredictionTerms = [10, 18, 40, 48, 130, 138, 520, 528, 27, 35, 108, 116, 351, 359, 1404, 1412, 28, 36, 112, 120, 364, 372, 1456, 1464]

            #Fetch bSolved matrix
            tempMat = bSolveMat = CSV.read(path*industryArr[j]*"/"*string(trainingSize)*"-1/240_"*fileName*"_bMatrix240_"*string(bestGamma)*".csv",
                nullable=false, header=false, datarow=1, types=fill(Float64,nPredictionTerms[i]))

            testMat += Array(tempMat)[:,11:27]
            # 10 industries predicting 10 industries
        end
        testMat = testMat/8
        testMat[find(testMat)] = 1

        for k = 1:17
            avgUsage[j,k] = countnz(testMat[:,k])/845
        end
    end

    realmean = mean(avgUsage)
    realStd = std(avgUsage)
    for j = 1:10
        for k = 1:17
            z_score = (avgUsage[j,k]-realmean)/(realStd)
            if z_score > 3.090232306
                signifMat[j,k] = "***"
            elseif z_score > 2.326347874
                signifMat[j,k] = "**"
            elseif z_score > 1.644853627
                signifMat[j,k] = "*"
            else
                signifMat[j,k] = " "
            end
        end
    end

    println(signifMat)
    println(avgUsage)

    writedlm(path*"ParametersAnalysis/MacroAverageSignif.csv",signifMat,",")
    writedlm(path*"ParametersAnalysis/MacroAverageUsage.csv",avgUsage,",")
end

getTA()
function getTA()
    avgUsage = zeros(10,8)
    signifMat = Array{String}(10,8)
    for j = 1:10
        println("Predicting industry ", industryArr[j])
        summary = CSV.read(path*industryArr[j]*"/"*"Summary "*string(trainingSize)*".csv", nullable=false)
        testMat = zeros(845,8)
        for i = [2,6,8,2+8,6+8,8+8,2+16,6+16,8+16]


            #Fetch dataset from summary
            fileName = summary[i,1]

            #Fetch best gamma from summary, based on R^2
            bestGamma = summary[i,7]

            nPredictionTerms = [10, 18, 40, 48, 130, 138, 520, 528, 27, 35, 108, 116, 351, 359, 1404, 1412, 28, 36, 112, 120, 364, 372, 1456, 1464]

            #Fetch bSolved matrix
            tempMat = bSolveMat = CSV.read(path*industryArr[j]*"/"*string(trainingSize)*"-1/240_"*fileName*"_bMatrix240_"*string(bestGamma)*".csv",
                nullable=false, header=false, datarow=1, types=fill(Float64,nPredictionTerms[i]))

            testMat += Array(tempMat)[:,end-7:end]
            # 10 industries predicting 10 industries
        end
        testMat = testMat/8
        testMat[find(testMat)] = 1

        for k = 1:8
            avgUsage[j,k] = countnz(testMat[:,k])/845
        end
    end

    realmean = mean(avgUsage)
    realStd = std(avgUsage)
    for j = 1:10
        for k = 1:8
            z_score = (avgUsage[j,k]-realmean)/(realStd)
            if z_score > 3.090232306
                signifMat[j,k] = "***"
            elseif z_score > 2.326347874
                signifMat[j,k] = "**"
            elseif z_score > 1.644853627
                signifMat[j,k] = "*"
            else
                signifMat[j,k] = " "
            end
        end
    end

    println(signifMat)
    println(avgUsage)

    writedlm(path*"ParametersAnalysis/TAAverageSignif.csv",signifMat,",")
    writedlm(path*"ParametersAnalysis/TAAverageUsage.csv",avgUsage,",")
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
    avgUsage = zeros(10)

    for i=1:10
        avgUsage[i] = countnz(testMat[1][:,i])
    end

    realmean = mean(avgUsage)
    degOfFree = 519

    stdev = std(avgUsage)

    for i=1:10
        zscore = (avgUsage[i]-realmean)/stdev
        if zscore > 1.965 || zscore < -1.965
            println("Paremeter ", i, " is significant, and had an average of ", avgUsage[i])
        end
    end
end


#Are the combination of parameters used within the past 10 years seen earlier?

for i = 9:9+8
    macroUsage(i)
end
# Macro dataset
function macroUsage(datIndx)
    dataset = testMat[datIndx]

    nCols = size(dataset)[2]
    nCols = 27

    avgUsage = zeros(nCols)

    for i=1:nCols
        avgUsage[i] = countnz(dataset[:,i])
    end

    realmean = mean(avgUsage)
    degOfFree = nCols-1

    stdev = std(avgUsage)

    for i=1:nCols
        zscore = (avgUsage[i]-realmean)/stdev
        if zscore > 1.965 || zscore < -1.965
            #println("Paremeter ", i, " is significant, and had an average of ", avgUsage[i])
        end
    end

    println(find(x -> x < 845*0.05, avgUsage))
    #println(avgUsage[11:27])
end

function macroUsage()
    nCols = 27

    avgUsage = zeros(8,nCols)

    for j=9:9+7
        for i=1:nCols
            avgUsage[j-8,i] = countnz(testMat[j][:,i])
        end
    end

    println(find(x -> x < 845*0.05, mean(avgUsage,1)))
    #println(avgUsage[11:27])
end
    # Are non linear transformation used
    # Are time series used?
    # Are TA used?
    # Are non linear transformation of times series used?
    # Are any of the raw inputs always used?
    # Is VIX actually used?
