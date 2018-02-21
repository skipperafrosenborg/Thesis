using JuMP
#using CPLEX
#using ConditionalJuMP
using Gurobi
using StatsBase
using DataFrames
using CSV

@everywhere include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
cd("$(homedir())/Documents/GitHub/Thesis/Data")
path = "$(homedir())/Documents/GitHub/Thesis/Data"

#Skipper's path
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"

#mainData = loadHousingData(path)
#mainData = loadCPUData(path)
mainData = loadElevatorData(path)

mainDataArray = Array(mainData)
any(isnan, cor(mainDataArray))
mainRows = size(mainDataArray)[1]
splitRows = Int64(ceil(mainRows/2))
train, test = splitDataIn2(mainDataArray, splitRows, mainRows)

trainingData = Array(train)
testData = Array(test)
nRows = size(trainingData)[1]
nCols = size(trainingData)[2]


### STAGE 1 ###
println("STAGE 1 INITIATED")
#Define solve
m = Model(solver = GurobiSolver())

#Add binary variables variables
@variable(m, 0 <= z[1:nCols] <= 1, Bin )

#Calculate highly correlated matix
any(isnan, trainingData)#isnan(trainingData)
trainingData
HC = cor(trainingData)
HC[:,17]
any(isnan, HC)
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
SSTO = sum((standY[i]-mean(standY))^2 for i=1:length(standY)) #the error if beta=0
amountOfGammas = 5
#Spaced between 0 and half the SSTO since this would then get SSTO*absSumOfBeta which would force everything to 0
gammaArray = logspace(0, log10(SSTO/5000), amountOfGammas)



function solveForBeta(x, y, k)
	standX = copy(x)
	X = copy(x)
	standY = copy(y)
	nRows = size(X)[1]
	#Initialise values for check later
	kValue = []
	RsquaredValue = []
	bSolved = []
	warmStartBeta = []
	warmstartZ = []
	HC = cor(X)
	for g in 1:1:1#amountOfGammas
		for i in startIter:1:kmax
			println("Setup model")
			bbdata = NodeData[]
			#Define parameters and model
			stage2Model = Model(solver = GurobiSolver(Presolve=2, TimeLimit = 200))
				#Define variables
			@variable(stage2Model, b[1:bCols])
			#@variable(stage2Model, v[1:bCols]) # auxiliary variables for abs
			@variable(stage2Model, T) # First objective term
			#@variable(stage2Model, G) #Second objective term
				#Define binary variable (5b)
			@variable(stage2Model, 0 <= z[1:bCols] <= 1, Bin )

			#Define objective function (5a)

			#=
			#Second objective term
			@constraint(stage2Model, 1*b .<= v)
			@constraint(stage2Model, -1*b .<= v)
			oneNorm = sum(v[i] for i=1:bCols)
			#gamma[g]*oneNorm <= G ---> -G <= -gamma[g]*oneNorm ---> G >= gamma[g]*oneNorm
			@constraint(stage2Model, gammaConstr, gammaArray[g]*oneNorm <= G)
			=#


			@objective(stage2Model, Min, T)
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

			#@constraint(stage2Model, sum((standY[i]- sum(standX[i,j]*b[j] for j=1:bCols))^2 for i=1:nRows)  <= T)

			#Define constraints (5c)
			@constraint(stage2Model, conBigMN, -1*b .<= bigM*z)
			@constraint(stage2Model, conBigMP,  1*b .<= bigM*z)

			#Define kmax constraint (5d)
			@constraint(stage2Model, kMaxConstr, sum(z[j] for j=1:bCols) <= i)

			#Constraint 5f - can only select one of a pair of highly correlated features
			rho = 0.8
			for k=1:bCols
				for j=1:bCols
					if k != j
						if HC[k,j] >= rho
							@constraint(stage2Model,z[k]+z[j] <= 1)
								#Check for errors in warmstart
							if warmstartZ[j] + warmstartZ[k] > 1
								println("Error with index $j and $k in constraint 5f")
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
					println("Error with index $j in constraint 5g")
				end
			end


			#=
			#Working for p<100
			expr = @expression(stage2Model, (standY[1]-standX[1,:]'*b)^2)
			for l=2:nRows
				tempExpr = @expression(stage2Model, (standY[l]-standX[l,:]'*b)^2)
				append!(expr, tempExpr)
				println("Constraint $l added")
			end
			@constraint(stage2Model, expr <= T)

			=#


			#Set kMax rhs constraint
			#JuMP.setRHS(kMaxConstr, i)

			#Second objective term
			#JuMP.setRHS(gammaConstr, gamma[g])

			println("Starting to solve stage 2 model with kMax = $i")

			#print(stage2Model)
			addinfocallback(stage2Model, infocallback, when = :Intermediate)
			#MIPNode: when we are at a node in the branch-and-cut tree
			#MIPSol: when we have found a new MIP incumbent
			#Intermediate: when we are still in the process of MIP
			#or during iterations of a continuous solver.
			#For MIPs, this is generally be used for keeping track
			#of pessimistic and optimistic bounds

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
				#bestGamma = g
				bestK = i
			end
			push!(kValue, i)
			push!(RsquaredValue, Rsquared)
			println("Rsquared value is: $Rsquared for kMax = $i and Gamma = $g")

			#nameString = string("SolutionWithK",i,"Gamma",g,".csv")
			#printSolutionToCSV(nameString, bbdata)
			#printNonZeroValues(bSolved)
		end
	end
	#println("STAGE 2 DONE")
	bestRsquared = maximum(RsquaredValue)
	kBestSol = kValue[indmax(RsquaredValue)]
	println("Bets solution found is: R^2 = $bestRsquared, k = $kBestSol")

	return bSolved
end

bbdata = NodeData[]
bSolution = solveForBeta(standX, standY, 9)

identifyParameters(bSolution, colPredNames)
sampleSize = 50
bCols = size(X)[2]
bSample = Matrix(sampleSize, bCols)


function createSampleX(x, inputRows)
	inputX = copy(x)
	outputX = inputX[inputRows,:]
	return outputX
end

function createSampleY(y, inputRows)
	inputY = copy(y)
	outputY = inputY[inputRows,:]
	return outputY
end

function selectSampleRows(rowsWanted, nRows)
	rowsSelected = rand(1:nRows, rowsWanted)
	return rowsSelected
end

function createBetaDistribution(bSample, standX, standY, k, sampleSize, rowsPerSample)
	for i=1:sampleSize
		sampleRows = selectSampleRowsWR(rowsPerSample, nRows)
		sampleX = createSampleX(standX, sampleRows)
		sampleY = createSampleY(standY, sampleRows)
		bSample[i,:] = solveForBeta(sampleX, sampleY, k)
	end
end

createBetaDistribution(bSample, standX, standY, 9, sampleSize, 200)

println("HEJ SKIPPER!")
using Bootstrap
nBoot = 1000
bSample = convert(Array{Float64}, bSample)
function createConfidenceIntervalArray(sampleInput, nBoot, confLevel)
	bSamples = convert(Array{Float64}, sampleInput)
	bCols = size(bSamples)[2]
	confIntArray = Matrix(2, bCols)
	for i=1:bCols
		bs = bootstrap(bSample[:,i], mean, BasicSampling(nBoot))
		cil = confLevel
		CiEst = ci(bs, BasicConfInt(cil))
		confIntArray[1,i] = CiEst[1][2] #lower
		confIntArray[2,i] = CiEst[1][3]
	end
	return confIntArray
end

confArray99 = createConfidenceIntervalArray(bSample, 1000, 0.99)
confArray95 = createConfidenceIntervalArray(bSample, 1000, 0.95)
confArray90 = createConfidenceIntervalArray(bSample, 1000, 0.90)

function testSignificance(confIntArray99, confIntArray95, confIntArray90, bResult)
	bCols = size(bResult)[1]
	significance = zeros(bCols)
	for i=1:bCols
		if bResult[i] >= confIntArray99[1,i] && bResult[i] <=confIntArray99[2,i] && confIntArray99[1,i] > 0
			significance[i] = 0.99
		elseif bResult[i] >= confIntArray99[1,i] && bResult[i] <=confIntArray99[2,i] && confIntArray99[2,i] < 0
			significance[i] = 0.99
		elseif bResult[i] >= confIntArray95[1,i] && bResult[i] <=confIntArray95[2,i] && confIntArray95[1,i] > 0
			significance[i] = 0.95
		elseif bResult[i] >= confIntArray95[1,i] && bResult[i] <=confIntArray95[2,i] && confIntArray95[2,i] < 0
			significance[i] = 0.95
		elseif bResult[i] >= confIntArray90[1,i] && bResult[i] <=confIntArray90[2,i] && confIntArray90[1,i] > 0
			significance[i] = 0.90
		elseif bResult[i] >= confIntArray90[1,i] && bResult[i] <=confIntArray90[2,i] && confIntArray90[2,i] < 0
			significance[i] = 0.90
		else
			significance[i] = 0
			println("BooHoo")
		end
	end
	return significance
end

bSolution = solveForBeta(standX, standY, 9)
signi = testSignificance(confArray99, confArray95, confArray90, bSample[1,:])
for i=1:72
	if signi[i] > 0
		println("Parameter $i is significant with ", signi[i])
	end
end

confArray90[2,52]
bresult = bSample[1,:]
bCols = size(bresult)[1]

baaa = 5
if baaa >= 3 && baaa <= 4
	println("Hej")
end


#If
CiEst = Array(ci(bs1, BasicConfInt(cil))) #return t0, lower, upper
CiEstLow = CiEst[1][2]
CiEstUp  = CiEst[1][3]
ci(bs1, PercentileConfInt(cil)) #return t0, lower, upper
ci(bs1, NormalConfInt(cil)) #return t0, lower, upper
ci(bs1, BCaConfInt(cil)) #return t0, lower, upper
#ci(bs1, sd1, StudentConfInt(cil), 4) #return t0, lower, upper

r = randn(100)
n_boot = 1000

## basic bootstrap
bs1 = bootstrap(r, std, BasicSampling(n_boot))

## balanced bootstrap
bs2 = bootstrap(r, std, BalancedSampling(n_boot))





##BOOTSTRAP FOR SIGNIFICANCE
srand(2)

using Distributions
using Optim

N=1000
K=3




function loglike(rho,y,x)
	rhoCols = length(rho)
    beta = rho[1:rhoCols-1]
    sigma2 = exp(rho[rhoCols])+eps(Float64)
    residual = y-x*beta
    dist = Normal(0, sigma2)
    contributions = logpdf.(dist,residual) #Normally distributed ERRORS
    loglikelihood = sum(contributions)
    return -loglikelihood #negative log-likelihood
end


rhoGuess = copy(bSolved)
rhoGuess = vcat(rhoGuess)

rhoCols = length(rhoGuess)
function wrapLoglike(rho)
    return loglike(rho,standY,standX)
end

optimum = optimize(wrapLoglike, rhoGuess, ConjugateGradient())
MLE = optimum.minimizer
MLE[rhoCols] = exp(MLE[rhoCols])

B=100 #amount of samples
samples = zeros(B, 72)

for b=1:20
    theIndex = sample(1:nRows,nRows)
    x = standX[theIndex,:]
    y = standY[theIndex,:]
    samples[b,:] = solveForBeta(x, y, 9)
	#println("Iteration $b/$B")
end
#samples[:, rhoCols] = exp(samples[:, rhoCols])

using Bootstrap
n_boot = 1000
bs1 = bootstrap(samples[:,3], mean, BasicSampling(n_boot))
cil = 0.95
ci(bs1, BasicConfInt(cil)) #return t0, lower, upper

#=
bootstrapSE = std(samples,1)
nullDistribution = samples
pvalues = ones(rhoCols)
for i=1:rhoCols
    nullDistribution[:,i] = nullDistribution[:,i]-mean(nullDistribution[:,i])
end
nullDistribution[:, rhoCols] = 1 + nullDistribution[:, rhoCols]
pvalues = [mean(abs(MLE[i]).<abs(nullDistribution[:,i])) for i=1:rhoCols]
for i=1:rhoCols
	if pvalues[i] >= 0.05
		println("parameter $i has a p-value of ", pvalues[i])
	end
end
=#

k = 4
i = 2
base = string("SolutionWithK",k,"I",i,".csv")
for j=1:3
	open(base,"w") do fp
		println(fp, "Hej",j)
	end
end
