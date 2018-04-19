using JuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
using Bootstrap #External packages, must be added

include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Skipper's path
#inputArg=106
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"

#HPC path
inputArg = parse(Int64, ARGS[1]) #Should range from 0 to 106
path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data"

mainData = loadIndexDataNoDurLOGReturn(path)
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results"

path = "/zhome/9f/d/88706/SpecialeCode/Results"
#### MUST CHANGE ####
fileName = path*"/IndexData/AALRTest/"

#Reset HPC path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML"
#cd(path)

dateAndReseccion = Array(mainData[:,end-1:end])
mainDataArr = Array(mainData[:,1:end-2])

colNames = names(mainData)

nRows = size(mainDataArr)[1]
nCols = size(mainDataArr)[2]

mainXarr = mainDataArr[:,1:nCols-1]
mainYarr = mainDataArr[:, nCols:nCols]

# Transform with time elements
mainXarr = expandWithTime3612(mainXarr)

# Transform #
mainXarr = expandWithTransformations(mainXarr)
mainXarr = expandWithMAandMomentum(mainXarr, mainYarr, (nCols-1))

# Standardize
standX = zScoreByColumn(mainXarr)
standY = zScoreByColumn(mainDataArr[:, nCols:nCols])
allData = hcat(standX, standY)
bCols = size(standX)[2]
nRows = size(standX)[1]

trainingSize = 120
predictions = 1
testRuns = nRows-trainingSize-predictions

Xtrain = allData[:, 1:bCols]
trainingData = Xtrain[:,1:nCols-1]

function solveStage1(trainingData)
	### STAGE 1 ###
	println("STAGE 1 INITIATED")
	#Define solve
	m = JuMP.Model(solver = GurobiSolver(OutputFlag = 0, Threads = 2))

	#Add binary variables variables
	@variable(m, 0 <= z[1:nCols-1] <= 1, Bin )

	#Calculate highly correlated matix
	HC = cor(trainingData)

	#Define objective function
	@objective(m, Max, sum(z[i] for i=1:length(z)))

	#Define max correlation and constraints
	rho = 0.8
	for i=1:length(z)
		for j=1:length(z)
			if i != j
				if HC[i,j] >= rho
					@constraint(m,z[i]+z[j] <= 1)
				end
			end
		end
	end

	#Get solution status
	status = solve(m);

	#Get objective value
	println("Solved stage 1 with kMax = ", getobjectivevalue(m))
	return kmax = getobjectivevalue(m)
end

kmax = solveStage1(trainingData)
#Get solution value
#zSolved = getvalue(z)
println("STAGE 1 DONE")

### STAGE 2 ###
println("STAGE 2 INITIATED")
bigM = 5
tau = 2
amountOfGammas = 3

kValue = []
RMSEValue = []
bSolved = []
bestBeta = []
prevSolutions = zeros(Int64(kmax*amountOfGammas), bCols)

#="""
For each of the three beta sets produced, we will test for significance
and condition number of model to see if cuts are necessary
High or low condition number doesn't mean that one correlation matrix is "better"
than the other. All it means is that variables are more correlated or less.
Whether it's good or not depends on the application.
"""=#
function stageThree(best3Beta, X, Y, allCuts)
	#=Condition Number
	  A high condition number indicates a multicollinearity problem. A condition
	  number greater than 15 is usually taken as evidence of
	  multicollinearity and a condition number greater than 30 is
	  usually an instance of severe multicollinearity
	=#

	bCols = size(X)[2]
	nRows = size(X)[1]
	cuts = Matrix(0, bCols+1)
	rowsPerSample = nRows #All of rows in training data to generate beta estimates, but selected with replacement
	totalSamples = 25 #25 different times we will get a beta estimate
	nBoot = 1000

	#For loop start
	for i = 1:size(best3Beta)[1]
		if signifBoolean[i] == 1 #if the previous solution was already completely significant, skip the bootstrapping
			continue
		end
		bestBeta = best3Beta[i,4:bCols+3]
		bestK = best3Beta[i,2]
		bestGamma = best3Beta[i,3]

		xColumns = []
		bSample = Matrix(totalSamples, bCols)

		bZeros = zeros(bCols)
		for j = 1:bCols
			if !isequal(bestBeta[j],0)
				push!(xColumns, j)
			end
		end

		selectedX = X[:,xColumns]
		condNumber = cond(selectedX)
		if condNumber >= 15
			bZeros[xColumns] = 1
			subsetSize = size(xColumns)[1]
			newCut = [bZeros' subsetSize]
			cuts = [cuts; newCut]
			println("A cut based on Condition number = $condNumber has been created from Beta$i")
		end

		# test significance
		bestZ = zeros(bestBeta)
		for l=1:size(bestBeta)[1]
			if bestBeta[l] != 0
				bestZ[l] = 1
			end
		end

		bZeros = zeros(bCols)
		bSample = createBetaDistribution(bSample, X, Y, bestK, totalSamples, rowsPerSample,  bestGamma, allCuts, bestZ) #standX, standY, k, sampleSize, rowsPerSample

		confArray99 = createConfidenceIntervalArray(bSample, nBoot, 0.99)
		confArray95 = createConfidenceIntervalArray(bSample, nBoot, 0.95)
		confArray90 = createConfidenceIntervalArray(bSample, nBoot, 0.90)

		significanceResult = testSignificance(confArray99, confArray95, confArray90, bestBeta)
		significanceResultNONSI = [] # = significanceResult[xColumns]
		subsetSize = size(xColumns)[1]
		for n = 1:size(significanceResult)[1]
			for s = 1:subsetSize
				if significanceResult[n] == 0 && xColumns[s] == n
					push!(significanceResultNONSI,xColumns[s])
					println("Parameter $n is selected, but NOT significant")
				elseif significanceResult[n] > 0 && xColumns[s] == n
					println("Parameter $n is significant with ", significanceResult[n])
				end
			end
		end

		if !isempty(significanceResultNONSI)
			bZeros[significanceResultNONSI] = 1
			subsetSize = size(significanceResultNONSI)[1]
			newCut = [bZeros' subsetSize]
			cuts = [cuts; newCut]
			println("A cut based on parameters being non-significant in Beta$i has been created")
		end
		if isempty(significanceResultNONSI)
			signifBoolean[i] = 1
		end
	end
	return cuts
end

function buildAndSolveStage2(standX, standY, curKmax, gamma, warmstartBool, warmStartSol, cuts, fixedZ, HC)
	HCPairCounter = 0

	#Define parameters and model
	stage2Model = JuMP.Model(solver = GurobiSolver(TimeLimit = 30, OutputFlag = 0, Threads = 2));

	#Define variables
	@variable(stage2Model, b[1:bCols]) #Beta values
	@variable(stage2Model, 0 <= z[1:bCols] <= 1, Bin ) #binary z variable
	@variable(stage2Model, v[1:bCols]) # auxiliary variables for abs
	@variable(stage2Model, T) #First objective term
	@variable(stage2Model, G) #Second objective term

	if warmstartBool == true
		for j in find(warmStartSol)
			setvalue(b[j], warmStartSol[j])
			setvalue(v[j], abs(warmStartSol[j]))
			setvalue(z[j], 1)
		end
	end

	#Define objective function (5a)
	@objective(stage2Model, Min, T+G)
	@constraint(stage2Model, soc, norm( [1-T;2*(standX*b-standY)] ) <= 1+T)
	@constraint(stage2Model, gammaConstr, gamma*ones(bCols)'*v <= G) #4bCols

	#Define constraints (5c)
	@constraint(stage2Model, conBigMN, -1*b .<= bigM*z) #from 0 to bCols -1
	@constraint(stage2Model, conBigMP,  1*b .<= bigM*z) #from bCols to 2bCols-1

	#Second objective term
	@constraint(stage2Model, 1*b .<= v) #from 2bCols to 3bCols-1
	@constraint(stage2Model, -1*b .<= v) #from 3bCols to 4bCols-1

	#Define kmax constraint (5d)
	@constraint(stage2Model, kMaxConstr, sum(z[j] for j=1:bCols) <= curKmax) #4bCols+1

	#Constraint 5f - can only select one of a pair of highly correlated features
	rho = 0.8
	for k=1:bCols
		for j=1:bCols
			if k != j
				if HC[k,j] >= rho
					HCPairCounter += 1
					@constraint(stage2Model,z[k]+z[j] <= 1) #from 4bCols+1 to (4bCols+1+HCPairCounter)
				end
			end
		end
	end

	#Constraint (5g) - only one transformation allowed (x, x^2, log(x) or sqrt(x))
	for j=1:(nCols-1)
		@constraint(stage2Model, z[j]+z[j+(nCols-1)]+z[j+2*(nCols-1)]+z[j+3*(nCols-1)] <= 1) #from (4bCols+1+HCPairCounter) to (4bCols+1+HCPairCounter+nCols)
	end

	#Implement cuts
	if !isempty(cuts)
		#nEmptyCuts = 19
		#zOnes = ones(bCols)
		for c = 1:size(cuts)[1]
			@constraint(stage2Model, sum(z[i] for i=find(cuts[c,1:end])) <= countnz(cuts[c,1:end])-1)
			#@constraint(stage2Model, sum(zOnes[i]*z[i] for i=1:bCols) <= bCols) #(4bCols+1+HCPairCounter+nCols) to (4bCols+1+HCPairCounter+nCols+18)
		end
	end

	if !isempty(fixedZ)
		@constraint(stage2Model, sum(z[i] for i=find(fixedZ)) == countnz(fixedZ))
	end


	status = solve(stage2Model)

	if status == :InfeasibleOrUnbounded
		bSolved = zeros(bCols)
        obj = 9999
        objBound = 9999
	else
		bSolved = getvalue(b)
		obj = getobjectivevalue(stage2Model)
		objBound = getobjectivebound(stage2Model)
	end
	return bSolved, obj, objBound
end

function checkRMSE(standX, standY, bSolved, best3Beta, k, gamma)
	RMSE = getRMSE(standX,standY,bSolved)

	if RMSE < best3Beta[1,1] #largest than 3rd largest RMSE
		if RMSE < best3Beta[2,1] #largest than 2nd largest RMSE
			if RMSE < best3Beta[3,1] #largest than largest RMSE
				best3Beta[1,:] = best3Beta[2,:]
				best3Beta[2,:] = best3Beta[3,:]
				best3Beta[3,1] = RMSE
				best3Beta[3,2] = k
				best3Beta[3,3] = gamma #potentially store actual gamma value
				best3Beta[3,4:bCols+3] = bSolved
			else
				best3Beta[1,:] = best3Beta[2,:]
				best3Beta[2,1] = RMSE
				best3Beta[2,2] = k
				best3Beta[2,3] = gamma #potentially store actual gamma value
				best3Beta[2,4:bCols+3] = bSolved
			end
		else
			best3Beta[1,1] = RMSE
			best3Beta[1,2] = k
			best3Beta[1,3] = gamma #potentially store actual gamma value
			best3Beta[1,4:bCols+3] = bSolved
		end
	end

	return best3Beta
end

stage2Model = 0
best3Beta = zeros(3,bCols+3)
best3Beta[:,1] = 1e3
signifBoolean = zeros(3)

@time(
for r = 1+inputArg*10:10+inputArg*10#(nRows-trainingSize-predictions)

	#SPLIT DATA
	trainingData = Xtrain[:,1:nCols-1]
	standX = allData[r:(trainingSize+r-1), 1:bCols]
	standY = allData[r:(trainingSize+r-1), bCols+1]
	standXTest  = allData[(trainingSize+r):(trainingSize+r+predictions-1), 1:bCols]
	standYTest  = allData[(trainingSize+r):(trainingSize+r+predictions-1), bCols+1]
	dateAndReseccionTest = dateAndReseccion[(trainingSize+r):(trainingSize+r+predictions-1), :]
	nRows = size(standX)[1]

	SSTO = sum((standY[i]-mean(standY))^2 for i=1:length(standY))
	amountOfGammas = 3

	#Spaced between 0 and half the SSTO since this would then get SSTO*absSumOfBeta which would force everything to 0
	gammaArray = log10.(logspace(0, SSTO/2, amountOfGammas))

	#INITIALISE STORING ARRAYS
	bSample = []
	allCuts = []
	signifBoolean = zeros(3)
	best3Beta = zeros(3,bCols+3)
	best3Beta[:,1] = 1e3
	runCount = 1
	curRMSE = 1e3
	HC = cor(standX)

	#BUILD AND SOLVE MODEL
	bSolved = []
	warmstart = false
	for i = 1:kmax
		for g = 1:amountOfGammas
			println("\nSolving for k = $i and gamma index = $g\n")
			bSolved, obj, objBound = buildAndSolveStage2(standX, standY, i, gammaArray[g], warmstart, bSolved, [], [], HC)

			#println("Cur index is: ",Int64(g+(i-1)*amountOfGammas))
			prevSolutions[Int64(g+(i-1)*amountOfGammas),:] = bSolved

			best3Beta = checkRMSE(standX, standY, bSolved, best3Beta, i, g)

			warmstart = true
		 end
	end

	#CHECK FOR SIGNIFICANCE (THIS INCLUDES SOLVING THE MODEL WITH FIXED z[] VARIABLES)
	cuts = stageThree(best3Beta, standX, standY, allCuts)

	#Check for cuts
	while !isempty(cuts)
		if runCount > 4
			break
		end
		best3Beta[:,1] = 1e3
		allCuts = vcat(allCuts,cuts[:,1:end-1])

		# Based on the cuts, ONLY resolve the problems that the cut affects.
		# Find the minimum number of variables in over all cuts
		minNumberOfVar = kmax
		for j=1:size(allCuts)[1]
			if countnz(allCuts[1,:]) < kmax
				minNumberOfVar = countnz(allCuts[1,:])
			end
		end

		#Rebuild and solve model with cuts
		for i = 1:kmax
			for g = 1:amountOfGammas
				bSolved = prevSolutions[Int64(g+(i-1)*amountOfGammas),:]
				if i >= minNumberOfVar
					#println("\nSolving for k = $i and gamma index = $g\n")
					bSolved, obj, objBound = buildAndSolveStage2(standX, standY, i, gammaArray[g], warmstart, prevSolutions[Int64(g+(i-1)*amountOfGammas),:], allCuts, [], HC)
				end

				#prevSolutions[Int64(i+(i-1)*amountOfGammas),:] = bSolved

				best3Beta = checkRMSE(standX, standY, bSolved, best3Beta, i, g)

				warmstart = true
			 end
		end

		#Check significance
		cuts = stageThree(best3Beta, standX, standY, allCuts)

		runCount += 1
	end

	best3Beta = cat(2,signifBoolean,best3Beta)

	prediction = Array{Float64}(3,1)
	prediction[1,1] = (standXTest*best3Beta[1,5:end])[1]
	prediction[2,1] = (standXTest*best3Beta[2,5:end])[1]
	prediction[3,1] = (standXTest*best3Beta[3,5:end])[1]

	best3Beta = cat(2,prediction,best3Beta)

	real = Array{Float64}(3,1)
	real[1,1] = standYTest[1]
	real[2,1] = standYTest[1]
	real[3,1] = standYTest[1]

	best3Beta = cat(2,real,best3Beta)

	dateAndReseccionOutput = Array{Int64}(3,2)
	dateAndReseccionOutput[1,:] = dateAndReseccionTest
	dateAndReseccionOutput[2,:] = dateAndReseccionTest
	dateAndReseccionOutput[3,:] = dateAndReseccionTest

	best3Beta = cat(2,dateAndReseccionOutput,best3Beta)

	### EXTRACT PREDICTION AND ISRS AND DATE###
	f = open(fileName*string(r)*"AALRBestK.csv", "w")
	write(f, "Date,Reseccion,Real,Prediciton,Significant,RMSE,k,gamma,"*expandedColNamesToString(colNames)*"\n")
	writecsv(f,best3Beta)
	close(f)
end
)
println("Complete")
