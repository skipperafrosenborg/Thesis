using JuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
using Bootstrap #External packages, must be added
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
#cd("$(homedir())/Documents/GitHub/Thesis/Data")
#path = "$(homedir())/Documents/GitHub/Thesis/Data"

#Skipper's path
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"
#HPC path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data"
#mainData = loadIndexDataNoDur(path)
#fileName = path*"/Results/IndexData/IndexData"
#mainData = loadConcrete(path)
#fileName = path*"/Results/Concrete/Run1/Concrete_Run"
#mainData = loadHousingData(path)
#fileName = path*"/Results/HousingData/HousingData"
mainData = loadCPUData(path)
fileName = path*"/Results/CPUData/Run5/CPUData"

#Reset HPC path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML"
#cd(path)

mainDataArr = Array(mainData)
halfRows = Int64(floor(size(mainDataArr)[1]/2))
train, test = splitDataIn2(mainDataArr, halfRows, size(mainData)[1])
halfRows = Int64(floor(size(test)[1]/2))
test, valid = splitDataIn2(mainDataArr, halfRows, size(test)[1])

colNames = names(mainData)
trainingData = Array(train)
testData = Array(test)
valiData = Array(valid)
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
m = JuMP.Model(solver = GurobiSolver(OutputFlag =0 ))

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

#Get solution status
status = solve(m);

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
yVali = valiData[:,nCols]
XVali = valiData[:,1:nCols-1]

# TRANSFORMATIONS ###
X = expandWithTransformations(X)
XTest = expandWithTransformations(XTest)
XVali = expandWithTransformations(XVali)

# Standardize
standX = zScoreByColumn(X)
standY = zscore(y)
standXTest = zScoreByColumn(XTest)
standYTest = zscore(yTest)
standXVali = zScoreByColumn(XVali)
standYVali = zscore(yVali)


# Output for Lasso regression in R
writedlm(fileName*"XTrain.CSV",standX,",")
writedlm(fileName*"YTrain.CSV",standY,",")
writedlm(fileName*"XTest.CSV",standXTest,",")
writedlm(fileName*"YTest.CSV",standYTest,",")
writedlm(fileName*"XVali.CSV",standXVali,",")
writedlm(fileName*"YVali.CSV",standYVali,",")

#Initialise values for check later
bCols = size(X)[2]
kValue = []
RsquaredValue = []
bSolved = []
bestBeta = []
warmStartBeta = []
warmstartZ = []

startIter = 1

bigM = 100
tau = 2

stage2Model = JuMP.Model(solver = GurobiSolver(TimeLimit = 30))
SSTO = sum((standY[i]-mean(standY))^2 for i=1:length(standY))
amountOfGammas = 5
#Spaced between 0 and half the SSTO since this would then get SSTO*absSumOfBeta which would force everything to 0
gammaArray = log10.(logspace(0, log10.(SSTO), amountOfGammas))

bbdata = NodeData[]

function buildStage2(standX, standY, kmax)
	HC = cor(X)
	HCPairCounter = 0
	#println("Building model")
	#Define parameters and model
	stage2Model = JuMP.Model(solver = GurobiSolver(TimeLimit = 20, OutputFlag = 0));
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
		#println("Finished loop $l/$bCols")
	end
	#println("Implemented crossBetaExpr")
	totalExpr = @expression(stage2Model, crossBetaExpr+simpleBetaExpr+xSquareExpr+ySquaredExpr)
	@constraint(stage2Model, quadConst, totalExpr <= T)
	#println("Successfully added quadratic constraints")

	#Define constraints (5c)
	@constraint(stage2Model, conBigMN, -1*b .<= bigM*z) #from 0 to bCols -1
	@constraint(stage2Model, conBigMP,  1*b .<= bigM*z) #from bCols to 2bCols-1

	#Second objective term
	@constraint(stage2Model, 1*b .<= v) #from 2bCols to 3bCols-1
	@constraint(stage2Model, -1*b .<= v) #from 3bCols to 4bCols-1
	oneNorm = sum(v[i] for i=1:bCols)

	#gamma[g]*oneNorm <= G ---> -G <= -gamma[g]*oneNorm --> G >= gamma[g]*oneNorm
	g=5
	@constraint(stage2Model, gammaConstr, 1*oneNorm <= G) #4bCols

	#Define kmax constraint (5d)
	@constraint(stage2Model, kMaxConstr, sum(z[j] for j=1:bCols) <= kmax) #4bCols+1

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

	#Adding empty cuts to possibly contain cuts. Can generate a maximum of 6 cuts for
	#each stage 3 and default value is 3 iterations between stage 2 and stage 3,
	#so 18 empty constraints are added + 1 for fixing the solution
	nEmptyCuts = 19
	zOnes = ones(bCols)
	for c = 1:nEmptyCuts
		@constraint(stage2Model, sum(zOnes[i]*z[i] for i=1:bCols) <= bCols) #(4bCols+1+HCPairCounter+nCols) to (4bCols+1+HCPairCounter+nCols+18)
	end

	#addinfocallback(stage2Model, infocallback, when = :Intermediate)

	JuMP.build(stage2Model)

	return internalmodel(stage2Model), HCPairCounter
end

function buildLasso(standX, standY)
	#Define parameters and model
	lassoModel = JuMP.Model(solver = GurobiSolver(TimeLimit = 30, OutputFlag = 0));
	gamma = 10

	#Define variables
	@variable(lassoModel, b[1:bCols]) #Beta values

	@variable(lassoModel, v[1:bCols]) # auxiliary variables for abs

	@variable(lassoModel, T) #First objective term
	@variable(lassoModel, G) #Second objective term

	#Define objective function (5a)
	@objective(lassoModel, Min, T+G)

	#println("Trying to implement new constraint")
	xSquareExpr = @expression(lassoModel, 0*b[1]^2)
	for l = 1:bCols
		coef = 0
		for j = 1:nRows
		   coef += standX[j,l]^(2)
		end
		append!(xSquareExpr, @expression(lassoModel, coef*b[l]^2))
	end
	#println("Implemented x^2")

	ySquaredExpr = @expression(lassoModel, 0*b[1])
	for j = 1:nRows
		append!(ySquaredExpr,@expression(lassoModel, standY[j,1]^2))
	end
	#println("Implemented y^2")

	simpleBetaExpr = @expression(lassoModel, 0*b[1])
	for l = 1:bCols
		coef = 0
		for j = 1:nRows
			coef += -1*2*standX[j, l]*standY[j]
		end
		append!(simpleBetaExpr, @expression(lassoModel, coef*b[l]))
	end
	#println("Implemented simpleBetaExpr")

	crossBetaExpr = @expression(lassoModel, 0*b[1]*b[2])
	iter = 1
	for l = 1:bCols
		for k = (l + 1):bCols
			coef = 0
			for j = 1:nRows
				coef += 2*standX[j,l]*standX[j,k]
			end
			append!(crossBetaExpr, @expression(lassoModel, coef*b[l,1]*b[k,1]))
		end
		#println("Finished loop $l/$bCols")
	end
	#println("Implemented crossBetaExpr")
	totalExpr = @expression(lassoModel, crossBetaExpr+simpleBetaExpr+xSquareExpr+ySquaredExpr)
	@constraint(lassoModel, quadConst, totalExpr <= T)
	#println("Successfully added quadratic constraints")

	#Second objective term
	@constraint(lassoModel, 1*b .<= v) #from 2bCols to 3bCols-1
	@constraint(lassoModel, -1*b .<= v) #from 3bCols to 4bCols-1
	oneNorm = sum(v[i] for i=1:bCols)

	#gamma[g]*oneNorm <= G ---> -G <= -gamma[g]*oneNorm --> G >= gamma[g]*oneNorm
	@constraint(lassoModel, gammaConstr, 1*oneNorm <= G) #4bCols

	JuMP.build(lassoModel)

	return internalmodel(lassoModel)
end

function solveLasso(model)
	bestNBeta = zeros(bCols+1,bCols+3)
	for i=1:bCols+1
		bestNBeta[i,1] = i-1
	end

	println("Solving Lasso for all gamma")
	gammaArray = logspace(0, 3, 100)
	gamma = 0
	tol = 1e-6

	open(fileName*"LassoLog.csv", "w") do f
		allColNames = expandedColNamesToString(colNames)
		write(f, "RsquaredValue, gamma, nNZ, $allColNames\n")
		for j in 1:length(gammaArray)
			gamma = gammaArray[j]
			changeGammaLasso(model, gamma)

			#Gurobi.writeproblem(model, "testproblem.lp")

			#solve problem
			status = Gurobi.optimize!(model)

			#Get solution
			sol = Gurobi.getsolution(model)

			#Get parameters
			bSolved = sol[1:bCols]

			#Shrink values to 0 if within tolerance "tol"
			for i=1:length(bSolved)
				if bSolved[i] < tol
					if bSolved[i] > -tol
						bSolved[i] = 0
					end
				end
			end

			#Out of sample test
			Rsquared = getRSquared(standXVali,standYVali,bSolved)

			#Count number of non zero elements
			numNZ = countnz(bSolved)

			if Rsquared > bestNBeta[numNZ+1,2]
				bestNBeta[numNZ+1,2] = getRSquared(standXTest,standYTest,bSolved)
				bestNBeta[numNZ+1,3] = gamma
				bestNBeta[numNZ+1,4:bCols+3] = bSolved
			end

			write(f, "$Rsquared, $gamma, $numNZ, $bSolved\n")
			#println("Rsquared =$Rsquared \t gamma =$j \t n NZ=$numNZ")
		end
	end
	f = open(fileName*"LassoBestK.csv", "w")
	write(f, "k,Rsquared,gamma,"*expandedColNamesToString(colNames)*"\n")
	writecsv(f,bestNBeta)
	close(f)
	println("Solve all Lasso problems")
	return bestNBeta
end

lassoM = buildLasso(standX, standY)
bestNLASSO = solveLasso(lassoM)

stage2Model, HCPairCounter = buildStage2(standX,standY, kmax)

# best3Beta[:,1] = Rsquared
# best3Beta[:,2] = kmax
# best3Beta[:,3] = gamma
# best3Beta[:,4:bCols+3] = beta

function solveForAllK(model, kmax)
	best3Beta = zeros(3,bCols+3)
	solArr = zeros(kmax*length(gammaArray),3)
	println("Solving for all k and gamma")
	for i in 1:1:kmax
		#println("Calculating warmstart solution")

		#=
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
		=#

		#Set kMax rhs constraint
		curUB = Gurobi.getconstrUB(model) #Get current UBbounds
		curUB[bCols*4+2] = i #Change upperbound in current bound vector
		Gurobi.setconstrUB!(model, curUB) #Push bound vector to model
		Gurobi.updatemodel!(model)

		#Set new Big M
		newBigM = 3#tau*norm(warmStartBeta, Inf)
		changeBigM(model,newBigM)
		for j in 1:length(gammaArray)
			changeGamma(model, gammaArray[j])

			#println("Starting to solve stage 2 model with kMax = $i and gamma = $j")

			#Gurobi.writeproblem(model, "testproblem.lp")

			#Solve Stage 2 model
			status = Gurobi.optimize!(model)
			#println("Objective value: ", Gurobi.getobjval(model))

			sol = Gurobi.getsolution(model)

			#printSolutionToCSV("lars.csv", bbdata)
			#Get solution and calculate R^2
			bSolved = sol[1:bCols]
			zSolved = sol[1+bCols:2*bCols]

			#Out of sample test
			Rsquared = getRSquared(standXVali,standYVali,bSolved)

			if Rsquared > best3Beta[1,1] #largest than 3rd largest Rsquared
				if Rsquared > best3Beta[2,1] #largest than 2nd largest Rsquared
					if Rsquared > best3Beta[3,1] #largest than largest Rsquared
						best3Beta[1,:] = best3Beta[2,:]
						best3Beta[2,:] = best3Beta[3,:]
						best3Beta[3,1] = Rsquared
						best3Beta[3,2] = i
						best3Beta[3,3] = j #potentially store actual gamma value
						best3Beta[3,4:bCols+3] = bSolved
					else
						best3Beta[1,:] = best3Beta[2,:]
						best3Beta[2,1] = Rsquared
						best3Beta[2,2] = i
						best3Beta[2,3] = j #potentially store actual gamma value
						best3Beta[2,4:bCols+3] = bSolved
					end
				else
					best3Beta[1,1] = Rsquared
					best3Beta[1,2] = i
					best3Beta[1,3] = j #potentially store actual gamma value
					best3Beta[1,4:bCols+3] = bSolved
				end
			end

			if any(Rsquared .> RsquaredValue) || isempty(RsquaredValue)
				bestBeta = bSolved
			end
			solArr[Int64((i-1)*length(gammaArray)+j),1] = i
			solArr[Int64((i-1)*length(gammaArray)+j),2] = j
			solArr[Int64((i-1)*length(gammaArray)+j),3] = Rsquared
			push!(kValue, (i-1)*length(gammaArray)+j)
			push!(RsquaredValue, Rsquared)
			#println("Rsquared =$Rsquared\t kMax =$i \t gamma =$j")
		end
		#printNonZeroValues(bSolved)
	end

	return best3Beta, solArr, model
end

function solveAndLogForAllK(model, kmax)
	best3Beta = zeros(3,bCols+3)
	solArr = zeros(kmax*length(gammaArray),3)
	println("Solving for all k and gamma")
	open(fileName*"AALRLog.csv", "w") do f
		allColNames = expandedColNamesToString(colNames)
		write(f, "RsquaredValue,gamma,k, $allColNames\n")
		for i in 1:1:kmax
			#println("Calculating warmstart solution")

			#=
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
			=#

			#Set kMax rhs constraint
			curUB = Gurobi.getconstrUB(model) #Get current UBbounds
			curUB[bCols*4+2] = i #Change upperbound in current bound vector
			Gurobi.setconstrUB!(model, curUB) #Push bound vector to model
			Gurobi.updatemodel!(model)

			#Set new Big M
			newBigM = 10#tau*norm(warmStartBeta, Inf)
			changeBigM(model,newBigM)
			for j in 1:length(gammaArray)
				gamma=gammaArray[j]
				changeGamma(model, gammaArray[j])

				#println("Starting to solve stage 2 model with kMax = $i and gamma = $j")

				#Gurobi.writeproblem(model, "testproblem.lp")

				#Solve Stage 2 model
				status = Gurobi.optimize!(model)
				#println("Objective value: ", Gurobi.getobjval(model))

				sol = Gurobi.getsolution(model)

				#printSolutionToCSV("lars.csv", bbdata)
				#Get solution and calculate R^2
				bSolved = sol[1:bCols]
				zSolved = sol[1+bCols:2*bCols]

				#Out of sample test
				Rsquared = getRSquared(standXVali,standYVali,bSolved)

				if Rsquared > best3Beta[1,1] #largest than 3rd largest Rsquared
					if Rsquared > best3Beta[2,1] #largest than 2nd largest Rsquared
						if Rsquared > best3Beta[3,1] #largest than largest Rsquared
							best3Beta[1,:] = best3Beta[2,:]
							best3Beta[2,:] = best3Beta[3,:]
							best3Beta[3,1] = Rsquared
							best3Beta[3,2] = i
							best3Beta[3,3] = j #potentially store actual gamma value
							best3Beta[3,4:bCols+3] = bSolved
						else
							best3Beta[1,:] = best3Beta[2,:]
							best3Beta[2,1] = Rsquared
							best3Beta[2,2] = i
							best3Beta[2,3] = j #potentially store actual gamma value
							best3Beta[2,4:bCols+3] = bSolved
						end
					else
						best3Beta[1,1] = Rsquared
						best3Beta[1,2] = i
						best3Beta[1,3] = j #potentially store actual gamma value
						best3Beta[1,4:bCols+3] = bSolved
					end
				end

				if any(Rsquared .> RsquaredValue) || isempty(RsquaredValue)
					bestBeta = bSolved
				end
				solArr[Int64((i-1)*length(gammaArray)+j),1] = i
				solArr[Int64((i-1)*length(gammaArray)+j),2] = j
				solArr[Int64((i-1)*length(gammaArray)+j),3] = Rsquared
				push!(kValue, (i-1)*length(gammaArray)+j)
				push!(RsquaredValue, Rsquared)
				println("Rsquared =$Rsquared\t kMax =$i \t gamma =$j")
				write(f, "$Rsquared,$gamma,$i,$bSolved\n")
			end
			#printNonZeroValues(bSolved)
		end
	end
	return best3Beta, solArr, model
end

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
	nBoot = 10000

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
		createBetaDistribution(bSample, X, Y, bestK, totalSamples, rowsPerSample,  bestGamma, allCuts, bestZ) #standX, standY, k, sampleSize, rowsPerSample

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

bSample = []
allCuts = []
signifBoolean = zeros(3)

stage2Model, HCPairCounter = buildStage2(standX,standY, kmax)

#Gurobi.writeproblem(stage2Model, "testproblem1.lp")

best3Beta, solArr, stage2Model = solveAndLogForAllK(stage2Model, kmax)

cuts = stageThree(best3Beta, standX, standY, allCuts)

#cuts = stageThree(best3Beta[3,4:bCols+3], best3Beta[3,2], best3Beta[3,3],
# 				  best3Beta[2,4:bCols+3], best3Beta[2,2], best3Beta[2,3],
#				  best3Beta[1,4:bCols+3], best3Beta[1,2], best3Beta[1,3],
#				  standX, standY, allCuts)
#writedlm("betaOut.CSV", cuts,",")

cutCounter = 0
runCount = 1
cutMatrix = []

while !isempty(cuts)
	#Function to add cuts to problem
	#addCuts(stage2Model, cuts)
	preCutCounter = copy(cutCounter)
	allCuts = vcat(allCuts,cuts)
	cutMatrix = cuts
	cutRows = size(cutMatrix)[1]
	cutCols = size(cutMatrix)[2]
	cutCounter += cutRows
	addCuts(stage2Model, cutMatrix, preCutCounter)


	#Resolve problem
	Gurobi.updatemodel!(stage2Model)
	#Gurobi.writeproblem(stage2Model, "testproblem2.lp")
	#println(stage2Model)
	best3Beta, solArr = solveAndLogForAllK(stage2Model, kmax)

	#Stage 3
	cuts = stageThree(best3Beta, standX, standY, allCuts)
	println("Finished iteration $runCount")
	runCount += 1
	if runCount == 4
		break
	end
end


best3Beta[1,1] = getRSquared(standXTest, standYTest, best3Beta[1,4:end])
best3Beta[2,1] = getRSquared(standXTest, standYTest, best3Beta[2,4:end])
best3Beta[3,1] = getRSquared(standXTest, standYTest, best3Beta[3,4:end])

best3Beta = cat(2,signifBoolean,best3Beta)



f = open(fileName*"AALRBestK.csv", "w")
write(f, "Significant,Rsquared,k,gamma,"*expandedColNamesToString(colNames)*"\n")
writecsv(f,best3Beta)
close(f)
#println(getRMSE(standXTest, standYTest, best3Beta[1,4:111]))
#println(getRMSE(standXTest, standYTest, best3Beta[2,4:111]))
#println(getRMSE(standXTest, standYTest, best3Beta[3,4:111]))
#Gurobi.writeproblem(stage2Model, "testproblem3.lp")
println("Complete")
