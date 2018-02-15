using JuMP
#using CPLEX
#using ConditionalJuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
using MLDataUtils
@everywhere include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
#cd("$(homedir())/Documents/GitHub/Thesis/Data")
#path = "$(homedir())/Documents/GitHub/Thesis/Data"

#Skipper's path
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"

mainData = loadHousingData(path)
#mainData = loadCPUData(path)

mainDataShuf = mainData
#mainDataShuf = shuffleobs(mainData)
train, test = splitobs(getobs(mainDataShuf), at = 0.5)

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

startIter = 1
stage2Model = Model(solver = GurobiSolver(Presolve=-1, TimeLimit = 200))
totalExpr = @expression(stage2Model, 0)
@everywhere HC = cor(X)
@everywhere bigM = 0
@everywhere tau = 2

function getBigM(i)
	warmStartBetaTemp = gradDecent(standX, standY, 30000, 1e-3, i, HC)
	bigMTemp = tau*norm(warmStartBetaTemp, Inf)
	println("Calculated big M iteration $i/$kmax")
	return i, bigMTemp
end


@time(test = pmap(getBigM, Array(1:1)))

@time(for i in 1:1
	getBigM(i)
end
)


for i in startIter:1:kmax
	if i == startIter
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
	end

	println("Calculating warmstart solution")
	#Calculate warmstart
	warmStartError = 1e6
	for j in 1:5
		warmStartBetaTemp = gradDecent(standX, standY, 30000, 1e-3, i, HC)
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

	if i == startIter
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

		#=
		#Working for p<100
		expr = @expression(stage2Model, (standY[1]-standX[1,:]'*b)^2)
		for l=2:nRows
			tempExpr = @expression(stage2Model, (standY[l]-standX[l,:]'*b)^2)
			append!(expr, tempExpr)
			#println("Constraint $l added")
		end
		@constraint(stage2Model, expr <= T)
		=#
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
	#Rsquared = getRSquared(standX,standY,bSolved)
	if any(Rsquared .> RsquaredValue) || isempty(RsquaredValue)
		bestBeta = bSolved
	end
	push!(kValue, i)
	push!(RsquaredValue, Rsquared)
	println("Rsquared value is: $Rsquared for kMax = $i")

	#printNonZeroValues(bSolved)
end
println([kValue RsquaredValue])
println("STAGE 2 DONE")
bestRsquared = maximum(RsquaredValue)
kBestSol = kValue[indmax(RsquaredValue)]
println("Bets solution found is: R^2 = $bestRsquared, k = $kBestSol")
