#Code written by Skipper af Rosenborg and Esben Bager
using StatsBase
using DataFrames
using CSV

#Load training size
trainingSizeInput = parse(Int64, ARGS[1])
#trainingSizeInput = 12
println(typeof(trainingSizeInput))

#set path
path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML/Lasso_Test"
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/ML"
cd(path)
@everywhere include("ParallelModelGeneration.jl")
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Function to run LASSO and output result files
function runLassos(VIX, raw, expTrans, timeTrans, TA, trainingSize)
	#Set path and load data

	#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/IndexDataDiff/"
	path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data/IndexDataDiff/"

	#= industry must be one of the following
	NoDur
	Durbl
	Manuf
	Enrgy
	HiTec
	Telcm
	Shops
	Hlth
	Utils
	Other
	=#
	industry = ARGS[2]
	#industry = "NoDur"
	mainData = loadIndexDataLOGReturn(industry, path)
	mainDataArr = Array(mainData[:,1:end-2])

	path = "/zhome/9f/d/88706/SpecialeCode/"
	#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/"
	dateAndReseccion = Array(mainData[:,end-1:end])
	mainDataArr = Array(mainData[:,1:end-2])

	#Extract names
	colNames = names(mainData)
	if VIX == 1
		if timeTrans == 1
			featureNames = expandColNamesTimeToString(colNames[1:end-3])
		else
			featureNames = colNames[1:end-3]
		end
	else
		if timeTrans == 1
			featureNames = expandColNamesTimeToString(colNames[1:end-4])
		else
			featureNames = colNames[1:end-4]
		end
	end
	s = split(expandedColNamesToString(featureNames,expTrans, TA),",")
	s = Array{String}(s)

	nRows = size(mainDataArr)[1]
	nCols = size(mainDataArr)[2]

	#Pipeline data and include different transformations
	fileName = path*"Results/IndexData/LassoTest/"*industry*"/"*string(trainingSize)*"-1/"*string(trainingSize)*"_"

	mainYarr = mainDataArr[:, nCols:nCols]

	mainXarr = mainDataArr[:,1:nCols-1]
	if raw == 1
		fileName = fileName*"Raw"
		mainXarr = mainXarr[:,1:10]
	else
		if VIX == 1
			fileName = fileName*"VIX_Macro"
		else
			fileName = fileName*"Macro"
			mainXarr = mainXarr[:,1:end-1]
		end
	end

	# Transform with time elements
	if timeTrans == 1
		fileName = fileName*"Time"
		mainXarr = expandWithTime3612(mainXarr)
	end

	# Transform #
	if expTrans == 1
		fileName = fileName*"Exp"
		mainXarr = expandWithTransformations(mainXarr)
	end

	if TA == 1
		fileName = fileName*"TA"
		mainXarr = expandWithMAandMomentum(mainXarr, mainYarr, (nCols-1))
	end

	# Standardize
	standX = zScoreByColumn(mainXarr)
	standY = mainYarr

	allData = hcat(standX, standY)
	bCols = size(standX)[2]
	nRows = size(standX)[1]
	println(" \nSolving Convex\n")

	fileName = fileName*"_"

	println(fileName)

	#Set gamma and initilise arrays
	nGammas = 10
	predictions = 1
	testRuns = nRows-trainingSize-predictions

	ISR = SharedArray{Float64}(testRuns, nGammas)
	Indi = SharedArray{Float64}(testRuns, nGammas)
	#gammaArray = log.(logspace(0, SSTO/2, nGammas))
	predictedArr = SharedArray{Float64}(testRuns, nGammas)
	realArr = SharedArray{Float64}(testRuns, 1)
	bSolvedArr = SharedArray{Float64}(testRuns, bCols)
	bSolMatrix = SharedArray{Float64}(testRuns, bCols, nGammas)

	xTrainInput = [allData[r:(trainingSize+r-1), 1:bCols] for r = 1:testRuns]
	YtrainInput = [allData[r:(trainingSize+r-1), bCols+1] for r = 1:testRuns]
	XpredInput  = [allData[(trainingSize+r+1-1), 1:bCols] for r = 1:testRuns]
	YpredInput  = [allData[(trainingSize+r+1-1), bCols+1] for r = 1:testRuns]

	gammaArray = zeros(testRuns,nGammas)
	for r = 1:testRuns
		maxVal = findmax(abs.(xTrainInput[r]'*YtrainInput[r]))[1]
		gammaArray[r,:] = Array(linspace(0,maxVal,nGammas))
 	end

	for i=1:nGammas
		gammaInput = gammaArray[:,i]

		#Loop over gamma and solve LASSO in parallel
		results = pmap(generatSolveAndProcess, xTrainInput, YtrainInput, XpredInput, YpredInput, gammaInput)
		results
		#Log results
		for j in 1:testRuns
			ISR[j,i] = results[j][1]
			Indi[j,i] = results[j][2]
			predictedArr[j,i] = results[j][3]
			realArr[j,1] = YpredInput[j][1]
			bSolMatrix[j,:,i] = results[j][4]
		end
	end

	#Output bmatrix files
	for i=1:nGammas
		writedlm(fileName*"bMatrix"*string(trainingSizeInput)*"_"*string(i)*".csv",bSolMatrix[:,:,i],",")
	end

	#Output files
	dateAndReseccionOutput = dateAndReseccion[trainingSize+1:trainingSize+testRuns,:]
	toInput = Array{String}(1,2)
	toInput[1,1] = "Date"
	toInput[1,2] = "Recession"
	dateAndReseccionOutput = vcat(toInput,dateAndReseccionOutput)
	runCounter = collect(0:testRuns)

	combinedArray = vcat((round.(gammaArray[1,:],3))', ISR)
	ISRcomb = hcat(runCounter,dateAndReseccionOutput, combinedArray)
	writedlm(fileName*"ISRsquared.CSV", ISRcomb,",")

	combinedArray = vcat((round.(gammaArray[1,:],3))', Indi)
	Indicomb = hcat(runCounter, dateAndReseccionOutput, combinedArray)
	writedlm(fileName*"Indi.CSV", Indicomb,",")

	combinedArray = vcat("Real value", realArr)
	realArr = hcat(runCounter, dateAndReseccionOutput, combinedArray)
	writedlm(fileName*"real.CSV", realArr,",")

	combinedArray = vcat((round.(gammaArray[1,:],3))', predictedArr)
	predictedArr = hcat(runCounter, dateAndReseccionOutput, combinedArray)
	writedlm(fileName*"predicted.CSV", predictedArr,",")

	#writedlm(fileName*"bSolved.CSV", bSolvedArr,",")
end


#Loops to process all dataset transformations
#Testing raw dataset
for timeTrans = 0:1
	for expTrans = 0:1
		for TA = 0:1
			runLassos(0, 1, timeTrans, expTrans, TA, trainingSizeInput)
		end
	end
end

#Testing macro dataset, including and excluding VIX
for VIX = 0:1
	for timeTrans = 0:1
		for expTrans = 0:1
			for TA = 0:1
				runLassos(VIX, 0, timeTrans, expTrans, TA, trainingSizeInput)
			end
		end
	end
end

#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/ML"
#cd(path)
println("Finished")
