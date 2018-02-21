using JuMP
#using CPLEX
#using ConditionalJuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
#using Bootstrap
@everywhere include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
cd("$(homedir())/Documents/GitHub/Thesis/Data")
path = "$(homedir())/Documents/GitHub/Thesis/Data"

#Skipper's path
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"

#=
#mainData = loadHousingData(path)
mainData = loadCPUData(path)
mainDataShuf = mainData
#mainDataShuf = shuffleobs(mainData)
train, test = splitobs(getobs(mainDataShuf), at = 0.5)
trainingData = Array(train)
testData = Array(test)
nRows = size(trainingData)[1]
nCols = size(trainingData)[2]
=#

mainData, testData = loadElevatorData(path)
dataSize = size(mainData)
colNames = names(mainData)
trainingData = Array(mainData)
testData = Array(testData)
nRows = size(trainingData)[1]
nCols = size(trainingData)[2]

#=
mainDataArray = Array(mainData)
mainRows = size(mainDataArray)[1]
splitRows = Int64(ceil(mainRows/2))
train, test = splitDataIn2(mainDataArray, splitRows, mainRows)
=#
### STAGE 1 ###
println("STAGE 1 INITIATED")
#Define solve
m = Model(solver = GurobiSolver())

#Add binary variables variables
@variable(m, 0 <= z[1:nCols] <= 1, Bin )

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

#Print model
#print(m)

#Get solution status
status = solve(m)

#Get objective value
println("Objective value kMax: ", getobjectivevalue(m))
kmax = getobjectivevalue(m)

#Get solution value
#zSolved = getvalue(z)
println("STAGE 1 DONE")

### STAGE 2 ###
println("STAGE 2 INITIATED")
println("Standardizing data")

#Split data by X and y
y = trainingData[:,nCols]
X = trainingData[:,1:nCols-1]
yTest = testData[:,nCols]
XTest = testData[:,1:nCols-1]

# TRANSFORMATIONS ###
X = expandWithTransformations(X)
XTest = expandWithTransformations(XTest)

# Standardize
@everywhere standX = zScoreByColumn(X)
@everywhere standY = zscore(y)
standXTest = zScoreByColumn(XTest)
standYTest = zscore(yTest)

# Output for Lasso regression in R
#=
writedlm("Elevators/elevatorXTrain.CSV",standX,",")
writedlm("Elevators/elevatorYTrain.CSV",standY,",")
writedlm("Elevators/elevatorXTest.CSV",standXTest,",")
writedlm("Elevators/elevatorYTest.CSV",standYTest,",")
=#

#Initialise values for check later
bCols = size(X)[2]
kValue = []
RsquaredValue = []
bSolved = []
bestBeta = []
warmStartBeta = []
warmstartZ = []
gValue = []

startIter = 1
stage2Model = Model(solver = GurobiSolver(Presolve=-1, TimeLimit = 200))
totalExpr = @expression(stage2Model, 0)
@everywhere HC = cor(X)
@everywhere bigM = 4
@everywhere tau = 2


function getBigM(i)
	warmStartBetaTemp = gradDecent(standX, standY, 11350, 1e-5, i, HC, 0)
	bigMTemp = tau*norm(warmStartBetaTemp, Inf)
	println("Calculated big M iteration $i/$kmax")
	return i, bigMTemp
end

@time(test = pmap(getBigM, Array(1:1)))

@time(for i in 150:150
	getBigM(i)
end)

println("Setup model")
#Define parameters and model
stage2Model = Model(solver = GurobiSolver(Presolve=-1, TimeLimit = 200))
gamma = 10

#Define variables
@variable(stage2Model, b[1:bCols])
@variable(stage2Model, T)

#Define binary variable (5b)
@variable(stage2Model, 0 <= z[1:bCols] <= 1, Bin )

#Define objective function (5a)
@objective(stage2Model, Min, T)

println("Trying to implement new constraint")
xSquareExpr = @expression(stage2Model, 0*b[1]^2)
for l = 1:bCols
	coef = 0
	for j = 1:nRows
	   coef += standX[j,l]^(2)
	end
	append!(xSquareExpr, @expression(stage2Model, coef*b[l]^2))
end
println("Implemented x^2")

ySquaredExpr = @expression(stage2Model, 0*b[1])
for j = 1:nRows
	append!(ySquaredExpr,@expression(stage2Model, standY[j,1]^2))
end
print(ySquaredExpr)
println("Implemented y^2")

simpleBetaExpr = @expression(stage2Model, 0*b[1])
for l = 1:bCols
	coef = 0
	for j = 1:nRows
		coef += -1*2*standX[j, l]*standY[j]
	end
	append!(simpleBetaExpr, @expression(stage2Model, coef*b[l]))
end
println("Implemented simpleBetaExpr")

crossBetaExpr = @expression(stage2Model, 0*b[1]*b[2])
iter = 1
for l = 1:bCols
	for k = (l + 1):bCols
		coef = 0
		for j = 1:nRows
			coef += 2*standX[j,l]*standX[j,k]
		end
		append!(crossBetaExpr, @expression(stage2Model, coef*b[l,1]*b[k,1]))
	end
	println("Finished out loop of $l/$bCols")
end
println("Implemented crossBetaExpr")
totalExpr = @expression(stage2Model, crossBetaExpr+simpleBetaExpr+xSquareExpr+ySquaredExpr)
@constraint(stage2Model, totalExpr <= T)
println("Successfully added quadratic constraints")

#Define constraints (5c)
@constraint(stage2Model, conBigMN, -1*b .<= bigM*z)
@constraint(stage2Model, conBigMP,  1*b .<= bigM*z)

#Define kmax constraint (5d)
@constraint(stage2Model, kMaxConstr, sum(z[j] for j=1:bCols) <= kmax)

#Constraint 5f - can only select one of a pair of highly correlated features
rho = 0.8
for k=1:bCols
	for j=1:bCols
		if k != j
			if HC[k,j] >= rho
				@constraint(stage2Model,z[k]+z[j] <= 1)

				#=
				#Check for errors in warmstart
				if warmstartZ[j] + warmstartZ[k] > 1
					println("Error with index $j and $k in constraingt 5f")
				end
				=#
			end
		end
	end
end

#Constraint (5g) - only one transformation allowed (x, x^2, log(x) or sqrt(x))
for j=1:(nCols-1)
	@constraint(stage2Model, z[j]+z[j+(nCols-1)]+z[j+2*(nCols-1)]+z[j+3*(nCols-1)] <= 1)

	#=
	#Check for errors in warmstart
	if warmstartZ[j]+warmstartZ[j+(nCols-1)]+warmstartZ[j+2*(nCols-1)]+warmstartZ[j+3*(nCols-1)] > 1
		println("Error with index $j in constraingt 5g")
	end
	=#
end
bbdata = NodeData[]
addinfocallback(stage2Model, infocallback, when = :MIPNode)
#MIPNode: when we are at a node in the branch-and-cut tree
#MIPSol: when we have found a new MIP incumbent
#Other: when we are still in the process of MIP
#or during iterations of a continuous solver.
#For MIPs, this is generally be used for keeping track
#of pessimistic and optimistic bounds
#https://github.com/JuliaOpt/JuMP.jl/pull/814

JuMP.build(stage2Model)

m2 = internalmodel(stage2Model)



for i in startIter:1:kmax
	println("Calculating warmstart solution")
	#Calculate warmstart
	#=
	warmStartError = 1e6
	for j in 1:5
		warmStartBetaTemp = gradDecent(standX, standY, 30000, 1e-3, i, HC, bSolved)
		tempError = norm(standY- standX*warmStartBetaTemp)^2
		println("Iteration $j of 5: Warmstart error is:", tempError)
		if tempError < warmStartError
			warmStartError = copy(tempError)
			warmStartBeta = copy(warmStartBetaTemp)
			println("Iteration $j of 5: Changed warmstartBeta")
		end
	end
	println("Warmstart error is:", warmStartError)
	#Set warmstart values
	warmstartZ = zeros(warmStartBeta)
	for j in 1:bCols
		setvalue(b[j], warmStartBeta[j])
		if isequal(warmStartBeta[j],0)
			setvalue(z[j], 0)
		else
			setvalue(z[j], 1)
			warmstartZ[j] = 1
		end
	end
	=#

	#Set kMax rhs constraint
	JuMP.setRHS(kMaxConstr, i)
	println("Starting to solve stage 2 model with kMax = $i")

	#print(stage2Model)

	#Solve Stage 2 model
	status = solve(stage2Model)
	println("Objective value: ", getobjectivevalue(stage2Model))

	#Get solution and calculate R^2
	bSolved = getvalue(b)
	zSolved = getvalue(z)

	#nameString = string("SolutionWithK",i,".csv")
	#printSolutionToCSV(nameString, bbdata)

	#Out of sample test
	Rsquared = getRSquared(standXTest,standYTest,bSolved)
	#Rsquared = getRSquared(standX,standY,bSolved)
	if any(Rsquared .> RsquaredValue) || isempty(RsquaredValue)
		bestBeta = bSolved
	end
	push!(kValue, i)
	#push!(gValue, g)
	push!(RsquaredValue, Rsquared)
	println("Rsquared value is: $Rsquared for kMax = $i")

	#printNonZeroValues(bSolved)
end


println([kValue RsquaredValue])
println("STAGE 2 DONE")
bestRsquared = maximum(RsquaredValue)
kBestSol = kValue[indmax(RsquaredValue)]
#gBestSol = gValue[indmax(RsquaredValue)]
println("Bets solution found is: R^2 = $bestRsquared, k = $kBestSol")# Gamma = $gBestSol")



"""
For each of the three beta sets produced, we will test for significance
and condition number of model to see if cuts are necessary
High or low condition number doesn't mean that one correlation matrix is "better"
than the other. All it means is that variables are more correlated or less.
Whether it's good or not depends on the application.
"""
using Bootstrap
function stageThree(bestBeta1, bestK1, bestBeta2, bestK2, bestBeta3, bestK3, X, Y)
	#Condition Number
	#A high condition number indicates a multicollinearity problem. A condition
	# number greater than 15 is usually taken as evidence of
	# multicollinearity and a condition number greater than 30 is
	# usually an instance of severe multicollinearity
	bCols = size(X)[2]
	nRows = size(X)[1]
	cuts = Matrix(0, bCols+1)
	bZeros = zeros(bCols)
	rowsPerSample = nRows #All of rows in training data to generate beta estimates, but selected with replacement
	totalSamples = 25 #25 different times we will get a beta estimate
	nBoot = 1000
	for i=1:3
		xColumns = []
		bSample = Matrix(totalSamples, bCols)
		if i == 1
			for j = 1:bCols
				if !isequal(bestBeta1[j],0)
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
			createBetaDistribution(bSample, X, Y, bestK1, totalSamples, rowsPerSample) #standX, standY, k, sampleSize, rowsPerSample
			confArray99 = createConfidenceIntervalArray(bSample, nBoot, 0.99)
			confArray95 = createConfidenceIntervalArray(bSample, nBoot, 0.95)
			confArray90 = createConfidenceIntervalArray(bSample, nBoot, 0.90)

			significanceResult = testSignificance(confArray99, confArray95, confArray90, bestBeta1)
			significanceResult = significanceResult[xColumns]
			subsetSize = size(xColumns)[1]
			for s=1:subsetSize
				if significanceResult[s] > 0
					println("Parameter $s is significant with ", significanceResult[s])
				else
					println("Parameter $s is NOT significant")
				end
			end


			if count(k->(k==0), significanceResult) > 0
				bZeros[xColumns] = 1
				subsetSize = size(xColumns)[1]
				newCut = [bZeros' subsetSize]
				cuts = [cuts; newCut]
				println("A cut based on parameters being non-significant in Beta$i has been created")
			end
		elseif i == 2
			for j = 1:bCols
				if !isequal(bestBeta1[j],0)
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
			createBetaDistribution(bSample, X, Y, bestK1, totalSamples, rowsPerSample) #standX, standY, k, sampleSize, rowsPerSample
			confArray99 = createConfidenceIntervalArray(bSample, nBoot, 0.99)
			confArray95 = createConfidenceIntervalArray(bSample, nBoot, 0.95)
			confArray90 = createConfidenceIntervalArray(bSample, nBoot, 0.90)

			significanceResult = testSignificance(confArray99, confArray95, confArray90, bestBeta1)
			significanceResult = significanceResult[xColumns]
			subsetSize = size(xColumns)[1]
			for s=1:subsetSize
				if significanceResult[s] > 0
					println("Parameter $s is significant with ", significanceResult[s])
				else
					println("Parameter $s is NOT significant")
				end
			end
			if count(k->(k==0), significanceResult) > 0
				bZeros[xColumns] = 1
				subsetSize = size(xColumns)[1]
				newCut = [bZeros' subsetSize]
				cuts = [cuts; newCut]
				println("A cut based on parameters being non-significant in Beta$i has been created")
			end

		else
			for j = 1:bCols
				if !isequal(bestBeta3[j],0)
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
			createBetaDistribution(bSample, X, Y, bestK3, totalSamples, rowsPerSample) #standX, standY, k, sampleSize, rowsPerSample
			confArray99 = createConfidenceIntervalArray(bSample, nBoot, 0.99)
			confArray95 = createConfidenceIntervalArray(bSample, nBoot, 0.95)
			confArray90 = createConfidenceIntervalArray(bSample, nBoot, 0.90)

			significanceResult = testSignificance(confArray99, confArray95, confArray90, bestBeta1)
			significanceResult = significanceResult[xColumns]
			subsetSize = size(xColumns)[1]
			for s=1:subsetSize
				if significanceResult[s] > 0
					println("Parameter $s is significant with ", significanceResult[s])
				else
					println("Parameter $s is NOT significant")
				end
			end

			if count(k->(k==0), significanceResult) > 0
				bZeros[xColumns] = 1
				subsetSize = size(xColumns)[1]
				newCut = [bZeros' subsetSize]
				cuts = [cuts; newCut]
				println("A cut based on parameters being non-significant in Beta$i has been created")
			end

		end
	end
	return cuts
end

function solveForBeta(X, Y, K)
	#Initialise values for check later
	bCols = size(X)[2]
	standX = copy(X)
	standY = copy(Y)
	bCols = size(standX)[2]
	nRows = size(standX)[1]
	kValue = []
	RsquaredValue = []
	bSolved = []
	bestBeta = []
	warmStartBeta = []
	warmstartZ = []
	gValue = []

	startIter = K
	stage2Model = Model(solver = GurobiSolver(Presolve=-1, TimeLimit = 200))
	totalExpr = @expression(stage2Model, 0)
	@everywhere HC = cor(X)
	@everywhere bigM = 4
	@everywhere tau = 2


	function getBigM(i)
		warmStartBetaTemp = gradDecent(standX, standY, 11350, 1e-5, i, HC, 0)
		bigMTemp = tau*norm(warmStartBetaTemp, Inf)
		#println("Calculated big M iteration $i/$kmax")
		return i, bigMTemp
	end

	@time(test = pmap(getBigM, Array(1:1)))

	@time(for i in 150:150
		getBigM(i)
	end)

	#println("Setup model")
	#Define parameters and model
	stage2Model = Model(solver = GurobiSolver(Presolve=-1, TimeLimit = 200))
	gamma = 10

	#Define variables
	@variable(stage2Model, b[1:bCols])
	@variable(stage2Model, T)

	#Define binary variable (5b)
	@variable(stage2Model, 0 <= z[1:bCols] <= 1, Bin )

	#Define objective function (5a)
	@objective(stage2Model, Min, T)

	#println("Trying to implement new constraint")
	xSquareExpr = @expression(stage2Model, 0*b[1]^2)
	for l = 1:bCols
		coef = 0
		for j = 1:nRows
		   coef += standX[j,l]^(2)
		end
		append!(xSquareExpr, @expression(stage2Model, coef*b[l]^2))
	end
	#println("Implemented x^2")

	ySquaredExpr = @expression(stage2Model, 0*b[1])
	for j = 1:nRows
		append!(ySquaredExpr,@expression(stage2Model, standY[j,1]^2))
	end
	print(ySquaredExpr)
	#println("Implemented y^2")

	simpleBetaExpr = @expression(stage2Model, 0*b[1])
	for l = 1:bCols
		coef = 0
		for j = 1:nRows
			coef += -1*2*standX[j, l]*standY[j]
		end
		append!(simpleBetaExpr, @expression(stage2Model, coef*b[l]))
	end
	#println("Implemented simpleBetaExpr")

	crossBetaExpr = @expression(stage2Model, 0*b[1]*b[2])
	iter = 1
	for l = 1:bCols
		for k = (l + 1):bCols
			coef = 0
			for j = 1:nRows
				coef += 2*standX[j,l]*standX[j,k]
			end
			append!(crossBetaExpr, @expression(stage2Model, coef*b[l,1]*b[k,1]))
		end
		#println("Finished out loop of $l/$bCols")
	end
	#println("Implemented crossBetaExpr")
	totalExpr = @expression(stage2Model, crossBetaExpr+simpleBetaExpr+xSquareExpr+ySquaredExpr)
	@constraint(stage2Model, totalExpr <= T)
	#println("Successfully added quadratic constraints")

	#Define constraints (5c)
	@constraint(stage2Model, conBigMN, -1*b .<= bigM*z)
	@constraint(stage2Model, conBigMP,  1*b .<= bigM*z)

	#Define kmax constraint (5d)
	@constraint(stage2Model, kMaxConstr, sum(z[j] for j=1:bCols) <= K)

	#Constraint 5f - can only select one of a pair of highly correlated features
	rho = 0.8
	for k=1:bCols
		for j=1:bCols
			if k != j
				if HC[k,j] >= rho
					@constraint(stage2Model,z[k]+z[j] <= 1)
				end
			end
		end
	end

	#Constraint (5g) - only one transformation allowed (x, x^2, log(x) or sqrt(x))
	for j=1:(nCols-1)
		@constraint(stage2Model, z[j]+z[j+(nCols-1)]+z[j+2*(nCols-1)]+z[j+3*(nCols-1)] <= 1)
	end


	JuMP.build(stage2Model)

	m2 = internalmodel(stage2Model)


	#println("Calculating warmstart solution")
	#Solve Stage 2 model
	status = solve(stage2Model)
	#println("Objective value: ", getobjectivevalue(stage2Model))

	#Get solution and calculate R^2
	bSolved = getvalue(b)
	zSolved = getvalue(z)

	return bSolved
end

stageThree(bestBeta, 14, bestBeta, 14, bestBeta, 14, standX, standY)
