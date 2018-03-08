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
mainData = loadIndexDataNoDur(path)
fileName = path*"/Results/IndexData/IndexData"
#mainData = loadConcrete(path)
#fileName = path*"/Results/Concrete/Concrete"
#mainData = loadHousingData(path)
#fileName = path*"/Results/HousingData/HousingData"
#mainData = loadCPUData(path)
#fileName = path*"/Results/CPUData/CPUData"

#Reset HPC path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML"
#cd(path)

mainDataArr = Array(mainData)

colNames = names(mainData)

nRows = size(mainDataArr)[1]
nCols = size(mainDataArr)[2]

mainXarr = mainDataArr[:,1:nCols-1]

# Transform #
mainXarr = expandWithTransformations(mainXarr)

# Standardize
standX = zScoreByColumn(mainXarr)
allData = hcat(standX,mainDataArr[:,nCols])

#=


#Split total data into  train / vali / test
halfRows = Int64(floor(size(allData)[1]/2))
train, test = splitDataIn2(allData, halfRows, size(allData)[1])
halfRows = Int64(floor(size(test)[1]/2))
test, vali = splitDataIn2(test, halfRows, size(test)[1])

#Split data by X and y
standX = train[:,1:nCols-1]
standY = train[:,nCols]
standXTest = test[:,1:nCols-1]
standYTest = test[:,nCols]
standXVali = vali[:,1:nCols-1]
standYVali = vali[:,nCols]
=#

#=
mainData, testData = loadElevatorData(path)
dataSize = size(mainData)
colNames = names(mainData)
trainingData = Array(mainData)
testData = Array(testData)
nRows = size(trainingData)[1]
nCols = size(trainingData)[2]
=#

#=
# Output for Lasso regression in R
writedlm(fileName*"XTrain.CSV",standX,",")
writedlm(fileName*"YTrain.CSV",standY,",")
writedlm(fileName*"XTest.CSV",standXTest,",")
writedlm(fileName*"YTest.CSV",standYTest,",")
writedlm(fileName*"XVali.CSV",standXVali,",")
writedlm(fileName*"YVali.CSV",standYVali,",")
=#

#Initialise values for check later
bCols = size(standX)[2]
kValue = []
RsquaredValue = []
bSolved = []

function buildLasso(standX, standY)
	nRows = size(standX)[1]

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

function solveLasso(model, X_pred)
	nGammas = 50
	yHatArr = zeros(nGammas)
	gammaArray = logspace(0, 3, nGammas)
	gamma = 0
	tol = 1e-6
	bSolved = 0
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
		yHatArr[j] = (X_pred*bSolved)[1]
	end
	return yHatArr
end

windowSize = 50
testSize = 100
yHatPred = zeros(testSize,50)
for i = 1:testSize
	if i%25 == 0
		println("Iteration $i out of $testSize iniated")
	end
	X = allData[1+i+100:100+windowSize+i,1:bCols]
	y = allData[1+i+100:100+windowSize+i,bCols+1]
	lassoM = buildLasso(X,y)

	X_pred = allData[windowSize+i+1+100:100+windowSize+i+2,1:bCols]
	y_pred = allData[windowSize+i+1+100:100+windowSize+i+2,bCols+1]
	yHatPred[i,:] = solveLasso(lassoM,X_pred)
end
y_vali = allData[windowSize+2:windowSize+2+testSize-1]

RMSE = zeros(50)
RsquareArr = zeros(50)
for j = 1:50
	SSres = sum((y_vali[i] - yHatPred[i,j])^2 for i=1:length(y_vali))
	n=length(y)

	SSTO = sum((y_vali[i]-mean(y_vali))^2 for i=1:length(y_vali))
    Rsquared = 1-SSres/SSTO

	RsquareArr[j] = Rsquared
	RMSE[j] = sqrt(SSres/n)
end




using Convex
using SCS
using DataFrames
using CSV
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Skipper's path
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"
#HPC path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data"
mainData = loadIndexDataNoDur(path)
fileName = path*"/Results/IndexData/IndexData"
#mainData = loadConcrete(path)
#fileName = path*"/Results/Concrete/Concrete"
#mainData = loadHousingData(path)
#fileName = path*"/Results/HousingData/HousingData"
#mainData = loadCPUData(path)
#fileName = path*"/Results/CPUData/CPUData"

#Reset HPC path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML"
#cd(path)

mainDataArr = Array(mainData)

colNames = names(mainData)

nRows = size(mainDataArr)[1]
nCols = size(mainDataArr)[2]

mainXarr = mainDataArr[:,1:nCols-1]

# Transform #
mainXarr = expandWithTransformations(mainXarr)

# Standardize
standX = zScoreByColumn(mainXarr)
allData = hcat(standX,mainDataArr[:,nCols])

windowSize = 50
testSize = 150

nGammas = 10
gammaArray = logspace(0, 3, nGammas)
gamma = 0

yHatPred = zeros(testSize,nGammas)
# Create a (column vector) variable of size n x 1.
bCols=size(allData)[2]-1
beta = Variable(bCols)
v = Variable(bCols)
for i = 1:testSize

	if i%25 == 0
		println("Iteration $i out of $testSize iniated")
	end
	X = allData[1+i:windowSize+i,1:bCols]
	y = allData[1+i:windowSize+i,bCols+1]

	for j=1:nGammas
		gamma = gammaArray[j]
		problem = minimize(sumsquares(y - X * beta)+gamma*norm(beta,1))
		# Solve the problem by calling solve!
		solve!(problem, SCSSolver(verbose=false));
		# Check the status of the problem
		problem.status # :Optimal, :Infeasible, :Unbounded etc.
		# Get the optimal value
		problem.optval
		yHatPred[i,j] = (allData[windowSize+i+1, 1:bCols]'*beta.value)[1]
	end
end

y_vali = allData[windowSize+2:windowSize+2+testSize-1,end]

RMSE = zeros(50)
RsquareArr = zeros(50)
for j = 1:50
	SSres = sum((y_vali[i] - yHatPred[i,j])^2 for i=1:length(y_vali))
	n=length(y)

	SSTO = sum((y_vali[i]-mean(y_vali))^2 for i=1:length(y_vali))
    Rsquared = 1-SSres/SSTO

	RsquareArr[j] = Rsquared
	RMSE[j] = sqrt(SSres/n)
end

maximum(RsquareArr)
minimum(RMSE)
