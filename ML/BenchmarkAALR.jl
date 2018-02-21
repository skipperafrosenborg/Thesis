using JuMP
#using CPLEX
#using ConditionalJuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
#cd("$(homedir())/Documents/GitHub/Thesis/Data")
#path = "$(homedir())/Documents/GitHub/Thesis/Data"

#Skipper's path
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"

mainData = loadHousingData(path)
#mainData = loadCPUData(path)

mainDataArr = Array(mainData)
halfRows = Int64(floor(size(mainDataArr)[1]/2))
train, test = splitDataIn2(mainDataArr, halfRows, size(mainData)[1])

trainingData = Array(train)
testData = Array(test)
nRows = size(trainingData)[1]
nCols = size(trainingData)[2]

#=
mainData, testData = loadElevatorData(path)
dataSize = size(mainData)
colNames = names(mainData)
trainingData = Array(mainData)
testData = Array(testData)
nRows = size(trainingData)[1]
nCols = size(trainingData)[2]
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

### STAGE 2 ###
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
standX = zScoreByColumn(X)
standY = zscore(y)
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

startIter = 1
HC = cor(X)
bigM = 100
tau = 2

println("Setup model")
#Define parameters and model
stage2Model = Model(solver = GurobiSolver(TimeLimit = 200))
gamma = 10

#Define variables
@variable(stage2Model, b[1:bCols]) #Beta values

#Define binary variable (5b)
@variable(stage2Model, 0 <= z[1:bCols] <= 1, Bin )

@variable(stage2Model, v[1:bCols]) # auxiliary variables for abs

@variable(stage2Model, T) #First objective term
@variable(stage2Model, G) #Second objective term


#Define objective function (5a)
@objective(stage2Model, Min, T+G)

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
	println("Finished loop $l/$bCols")
end
println("Implemented crossBetaExpr")
totalExpr = @expression(stage2Model, crossBetaExpr+simpleBetaExpr+xSquareExpr+ySquaredExpr)
@constraint(stage2Model, quadConst, totalExpr <= T)
println("Successfully added quadratic constraints")

#Define constraints (5c)
@constraint(stage2Model, conBigMN, -1*b .<= bigM*z)
@constraint(stage2Model, conBigMP,  1*b .<= bigM*z)


SSTO = sum((standY[i]-mean(standY))^2 for i=1:length(standY))
amountOfGammas = 5
#Spaced between 0 and half the SSTO since this would then get SSTO*absSumOfBeta which would force everything to 0
gammaArray = logspace(0, log10(SSTO/5000), amountOfGammas)
#Second objective term
@constraint(stage2Model, 1*b .<= v)
@constraint(stage2Model, -1*b .<= v)
oneNorm = sum(v[i] for i=1:bCols)

#gamma[g]*oneNorm <= G ---> -G <= -gamma[g]*oneNorm --> G >= gamma[g]*oneNorm
g=5
@constraint(stage2Model, gammaConstr, gammaArray[g]*oneNorm <= G)

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

JuMP.build(stage2Model)

m2 = internalmodel(stage2Model)
stage2Model = Model(solver = GurobiSolver(TimeLimit = 200))

function changeBigM(model, newBigM)
	startIndx = (nCols-1)*4
	for i in 1:((nCols-1)*4)
		Gurobi.changecoeffs!(m2,[i],[startIndx+i],[-newBigM])
		Gurobi.changecoeffs!(m2,[i+(nCols-1)*4],[startIndx+i],[-newBigM])
		Gurobi.updatemodel!(m2)
	end
end

function changeGamma(model, newGamma)
	startRow  = (nCols-1)*4*4+1
	startIndx = (nCols-1)*4*2
	for i in 1:(nCols-1)*4
		Gurobi.changecoeffs!(m2, [startRow], [startIndx+i], [newGamma])
		Gurobi.updatemodel!(m2)
	end
end

Gurobi.writeproblem(m2, "testproblem.lp")

solArr = zeros(kmax*length(gammaArray),3)

i=10
for i in startIter:1:kmax
	println("Calculating warmstart solution")
	#Calculate warmstart

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
	startVar = zeros(length(warmStartBeta)*3+2)
	warmstartZ = zeros(warmStartBeta)
	for j in 1:bCols
		startVar[j] = warmStartBeta[j]
		startVar[j+2*bCols] = abs(warmStartBeta[j])
		if isequal(warmStartBeta[j],0)
			startVar[j+bCols] = 0
		else
			startVar[j+bCols] = 1
			warmstartZ[j] = 1
		end
	end
	startVar[bCols*3+1] = 10000
	startVar[bCols*3+2] = 10000

	Gurobi.setwarmstart!(m2,startVar)

	#Set kMax rhs constraint
	curLB = Gurobi.getconstrUB(m2) #Get current UBbounds
	curLB[bCols*4+2] = i #Change upperbound in current bound vector
	Gurobi.setconstrUB!(m2, curLB) #Push bound vector to model
	Gurobi.updatemodel!(m2)

	#Set new Big M
	newBigM = tau*norm(warmStartBeta, Inf)
	changeBigM(m2,newBigM)

	for j in 1:length(gammaArray)
		changeGamma(m2, gammaArray[j])

		println("Starting to solve stage 2 model with kMax = $i and gamma = $j")

		Gurobi.writeproblem(m2, "testproblem.lp")

		#Solve Stage 2 model
		status = Gurobi.optimize!(m2)
		println("Objective value: ", Gurobi.getobjval(m2))

		sol = Gurobi.getsolution(m2)

		#Get solution and calculate R^2
		bSolved = sol[1:bCols]
		zSolved = sol[1+bCols:2*bCols]

		#Out of sample test
		Rsquared = getRSquared(standXTest,standYTest,bSolved)
		#Rsquared = getRSquared(standX,standY,bSolved)
		if any(Rsquared .> RsquaredValue) || isempty(RsquaredValue)
			bestBeta = bSolved
		end
		solArr[Int64((i-1)*length(gammaArray)+j),1] = i
		solArr[Int64((i-1)*length(gammaArray)+j),2] = j
		solArr[Int64((i-1)*length(gammaArray)+j),3] = Rsquared
		push!(kValue, (i-1)*length(gammaArray)+j)
		push!(RsquaredValue, Rsquared)
		println("Rsquared value is: $Rsquared for kMax = $i")
	end
	#printNonZeroValues(bSolved)
end
indmax(solArr[:,3])
println("STAGE 2 DONE")
bestRsquared = maximum(RsquaredValue)
kBestSol = kValue[indmax(RsquaredValue)]
println("Bets solution found is: R^2 = $bestRsquared, k = $kBestSol")
