using JuMP
#using CPLEX
#using ConditionalJuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
using MLDataUtils
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
#cd("$(homedir())/Documents/GitHub/Thesis/Data")
#path = "$(homedir())/Documents/GitHub/Thesis/Data"

#Skipper's path
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"

#mainData = loadHousingData(path)
#mainData = loadCPUData(path)
mainData, testData = loadElevatorData(path)

dataSize = size(mainData)
colNames = names(mainData)

#mainDataShuf = shuffleobs(mainData)
#train, test = splitobs(getobs(mainDataShuf), at = 0.5)

train = Array(mainData)

trainingData = Array(train)
testData = Array(testData)
nRows = size(trainingData)[1]
nCols = size(trainingData)[2]

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
kValue = []
RsquaredValue = []
bSolved = []
bestBeta = []
warmStartBeta = []
warmstartZ = []
HC = cor(X)
for i in 2:1:kmax
	println("Setup model")
	#Define parameters and model
	bCols = size(X)[2]

	stage2Model = Model(solver = GurobiSolver(Presolve=-1, TimeLimit = 200))
	gamma = 10


	#Define variables
	@variable(stage2Model, b[1:bCols])

	@variable(stage2Model, T)
	#@variable(stage2Model, T[1:consplit])
	#Define binary variable (5b)
	@variable(stage2Model, 0 <= z[1:bCols] <= 1, Bin )
	#@variable(stage2Model, O)

	println("Calculating warmstart solution")
	#Calculate warmstart
	warmStartError = 1e6
	for j in 1:5
		warmStartBetaTemp = gradDecent(standX, standY, 30000, 1e-6, i, HC)
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

	#Calculating bigM
	tau = 2
	bigM = tau*norm(warmStartBeta, Inf)

	#Define objective function (5a)
	@objective(stage2Model, Min, T)#sum(T[j] for j= 1:consplit))

	println("Trying to implement new constraint")
	#@constraint(stage2Model, norm(standY - standX*b) <= T)

 	#Working for p<100
	expr = @expression(stage2Model, (standY[1]-standX[1,:]'*b)^2)
	for l=2:nRows
		tempExpr = @expression(stage2Model, (standY[l]-standX[l,:]'*b)^2)
		append!(expr, tempExpr)
		#println("Constraint $l added")
	end
	println("Successfully added quadratic constraints")
	@constraint(stage2Model, expr <= T)

	#@constraint(stage2Model, sum((standY[j] - standX[j,:]'*b)^2 for j=1:nRows) <= T)

	#=
	for k = 1:(consplit-1)
			@constraint(stage2Model, sum((standY[j] - standX[j,:]'*b)^2 for j = Int64(1+(k-1)*floor(nRows/consplit)):Int64(floor(nRows/consplit)+(k-1)*floor(nRows/consplit))) <=   T[k])
			println("Constraint $k added")
	end
	=#
	#@constraint(stage2Model, sum((standY[j] - standX[j,:]'*b)^2 for j = Int64(1+floor(nRows/consplit)+(consplit-1)*floor(nRows/consplit)):Int64(nRows)) <=   T[consplit])

	println("Implemented new constraints")

	#@constraint(stage2Model, 1 <=T)
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

					#Check for errors in warmstart
					if warmstartZ[j] + warmstartZ[k] > 1
						println("Error with index $j and $k in constraingt 5f")
					end
				end
			end
		end
	end

	#Constraint (5g) - only one transformation allowed (x, x^2, log(x) or sqrt(x))
	for j=1:(nCols-1)
		@constraint(stage2Model, z[j]+z[j+(nCols-1)]+z[j+2*(nCols-1)]+z[j+3*(nCols-1)] <= 1)

		#Check for errors in warmstart
		if warmstartZ[j]+warmstartZ[j+(nCols-1)]+warmstartZ[j+2*(nCols-1)]+warmstartZ[j+3*(nCols-1)] > 1
			println("Error with index $j in constraingt 5g")
		end
	end

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

	#Out of sample test
	Rsquared = getRSquared(standXTest,standYTest,bSolved)
	if any(Rsquared .> RsquaredValue) || isempty(RsquaredValue)
		bestBeta = bSolved
	end
	push!(kValue, i)
	push!(RsquaredValue, Rsquared)
	println("Rsquared value is: $Rsquared for kMax = $i")

	#printNonZeroValues(bSolved)
end
println("STAGE 2 DONE")
bestRsquared = maximum(RsquaredValue)
kBestSol = kValue[indmax(RsquaredValue)]
println("Bets solution found is: R^2 = $bestRsquared, k = $kBestSol")
