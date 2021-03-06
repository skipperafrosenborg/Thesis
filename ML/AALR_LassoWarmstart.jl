using JuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
using Bootstrap #External packages, must be added
using JLD
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Skipper's path
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"

#HPC path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data"
mainData = loadIndexDataNoDurLOGReturn(path)
fileName = path*"/Results/IndexData/IndexData"

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

trainingSize = 12
predictions = 1
testRuns = nRows-trainingSize-predictions

Xtrain = allData[:, 1:bCols]
trainingData = Xtrain[:,1:nCols-1]

### STAGE 1 ###
println("STAGE 1 INITIATED")
#Define solve
m = JuMP.Model(solver = GurobiSolver(OutputFlag =0 ))

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
println("Objective value kMax: ", getobjectivevalue(m))
kmax = getobjectivevalue(m)

println("STAGE 1 DONE")

amountOfGammas = 3
solArr = zeros((nRows-trainingSize-predictions), kmax*amountOfGammas)
realArr = zeros((nRows-trainingSize-predictions), 1)
ISRSquared = zeros((nRows-trainingSize-predictions), kmax*amountOfGammas)
bSolMatrix = zeros((nRows-trainingSize-predictions), kmax*amountOfGammas, bCols)

### STAGE 2 ###
println("STAGE 2 INITIATED")

#Initialise values for check later
bSolved = []
bestBeta = []

bigM = 100
#Spaced between 0 and half the SSTO since this would then get SSTO*absSumOfBeta which would force everything to 0
#gammaArray = log10.(logspace(0.001, log10.(SSTO), amountOfGammas))
#gammaArray = log10.(logspace(0.001, 10, amountOfGammas))
amountofGammas = 3
gammaArray = log10.(logspace(0.01, 5, 3))


function solveLasso(Xtrain, Ytrain, gamma)
	bCols = size(Xtrain)[2]
	M = JuMP.Model(solver = GurobiSolver(OutputFlag = 0))
	@variables M begin
	        b[1:bCols]
	        t[1:bCols]
	        w
	end
	@objective(M,Min,0.5*w+gamma*ones(bCols)'*t)
	@constraint(M, soc, norm( [1-w;2*(Xtrain*b-Ytrain)] ) <= 1+w)
	@constraint(M,  b .<= t)
	@constraint(M, -t .<= b)

	solve(M)
	bSolved = getvalue(b)

	for i in 1:length(bSolved)
		if bSolved[i] <= 1e-6
			if bSolved[i] >= -1e-6
				bSolved[i] = 0
			end
		end
	end

	k = countnz(bSolved)
	return bSolved, k
end

function getBetaAndGamma(standX,standY, kmax)
	bSolved=zeros(bCols)
	k=1000
	iter = 250
    iterCounter = 1
	gammaArray = log10.(logspace(0, 10, 500))
	gammaArray
    distance = 125
	while (k!=kmax && iterCounter <= 25)
        if iter>500 || iter <1
            break
        end

		bSolved, k = solveLasso(standX,standY,gammaArray[iter])
        #println("k= $k \t distance = $distance, iter = $iter")
        if k > kmax
            iter += distance
        else
            iter -= distance
        end
        distance = Int64(ceil(distance/2))
        iterCounter += 1
		#println("Iteration: ",iter," gamma=",gammaArray[iter])
	end

    if k != kmax
        iter = 1
        while (k!=kmax && iter<500)
    		bSolved, k = solveLasso(standX,standY,gammaArray[iter])
            #println("k= $k \t distance = $distance, iter = $iter")
            iter += 1
    		#println("Iteration: ",iter," gamma=",gammaArray[iter])
    	end
    end

	k = countnz(bSolved)
	return bSolved
end

function solveAndLogForAllK(kmax, standXVali, standYVali, r, warmstart)
    best3Beta = zeros(3,bCols+3)
    gammaArray = log10.(logspace(0.01, 5, 3))
    bestObjtives = zeros(kmax, length(gammaArray))
	#println("Solving for all k and gamma")
    println("Warmstart = ",warmstart)
	#allColNames = expandedColNamesToString(colNames)
    bSolved = zeros(bCols,1)
    SSTO = sum((standY[i]-mean(standY[:]))^2 for i=1:length(standY))
    gammaArray = log10.(logspace(0.001, SSTO, amountOfGammas))
    HC = cor(standX)
	for i in 1:kmax
        if warmstart == true
            bSolved = getBetaAndGamma(standX,standY,i)
        end
		for g in 1:length(gammaArray)
			gamma = gammaArray[g]

        	HCPairCounter = 0

            #println("Building model")
        	#Define parameters and model
        	stage2Model = JuMP.Model(solver = GurobiSolver(TimeLimit = 30, OutputFlag = 0));

        	#Define variables
        	@variable(stage2Model, b[1:bCols], start = 0) #Beta values

        	#Define binary variable (5b)
        	@variable(stage2Model, 0 <= z[1:bCols] <= 1, Bin, start = 0)

        	@variable(stage2Model, v[1:bCols], start = 0) # auxiliary variables for abs

            if warmstart == true
                for j in find(bSolved)
    				setvalue(b[j], bSolved[j])
    			    setvalue(v[j], abs(bSolved[j]))
    				setvalue(z[j], 1)
    			end
            end

        	@variable(stage2Model, T) #First objective term
        	@variable(stage2Model, G) #Second objective term

        	#Define objective function (5a)
        	@objective(stage2Model, Min, T+G)

            @constraint(stage2Model, soc, norm( [1-T;2*(standX*b-standY)] ) <= 1+T)

        	#Define constraints (5c)
        	@constraint(stage2Model, conBigMN, -1*b .<= bigM*z) #from 0 to bCols -1
        	@constraint(stage2Model, conBigMP,  1*b .<= bigM*z) #from bCols to 2bCols-1

        	#Second objective term
        	@constraint(stage2Model, 1*b .<= v) #from 2bCols to 3bCols-1
        	@constraint(stage2Model, -1*b .<= v) #from 3bCols to 4bCols-1


        	#gamma[g]*oneNorm <= G ---> -G <= -gamma[g]*oneNorm --> G >= gamma[g]*oneNorm
            @constraint(stage2Model, gammaConstr, gamma*ones(bCols)'*v <= G) #4bCols

            #oneNorm = sum(v[i] for i=1:bCols)
            #@constraint(stage2Model, gammaConstr, gamma*oneNorm <= G) #4bCols

        	#Define kmax constraint (5d)
        	@constraint(stage2Model, kMaxConstr, sum(z[j] for j=1:bCols) <= i) #4bCols+1

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

			#Solve Stage 2 model
            solve(stage2Model)
            bSolved = getvalue(b)
            bSolMatrix[r,Int64((i-1)*amountOfGammas+g),:] = bSolved

            SSres = sum((standY[i]-(standX[i,:]'*bSolved))^2 for i=1:length(standY))
            ISRSquared[r,Int64((i-1)*amountOfGammas+g)] = 1 - SSres/SSTO
			#Get solution and calculate R^2
			#bSolved = sol[1:bCols]
			#zSolved = sol[1+bCols:2*bCols]

            obj = Gurobi.getobjval(internalmodel(stage2Model))
            bound = Gurobi.getobjbound(internalmodel(stage2Model))
            bestObjtives[Int64(i),Int64(g)] = obj
            gap = abs.(-1+bound/obj)

            prediction = standXVali*bSolved

			println("Prediction =$prediction\t Real =$standYVali\t kMax =$i \t gamma =$g \t obj = $obj, bound = $bound, gap = $gap")
            solArr[r,Int64((i-1)*amountOfGammas+g)] = prediction[1]
            realArr[r,1] = standYVali[1]

		end
		#printNonZeroValues(bSolved)
	end
    println(warmstart)
    return bestObjtives
	#return #best3Beta, solArr, model
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

function AALR_Time_Run(standX, standY, standXVali, standYVali, trainingData, r)
    nRows = size(standX)[1]

    bSample = []
    allCuts = []
    signifBoolean = zeros(3)
    #stage2Model, HCPairCounter = @time(buildStage2(standX,standY, kmax))

    warmstart = false
    bestObjtives = @time(solveAndLogForAllK(kmax, standXVali, standYVali, r, warmstart))
    print("This was time for non warmstart")
    println("")

    warmstart = true
    @time(solveAndLogForAllK(kmax, standXVali, standYVali, r, warmstart))
    print("This was time for warmstart")
    #println("")
    #Gurobi.writeproblem(stage2Model, "testproblem1.lp")

end
bestObjtives = 0
#inputArg = parse(Int64, ARGS[1])

inputArg = 0
warmstart = false
@time(
for r = 1:100:501#1+inputArg*10:10+inputArg*10#(nRows-trainingSize-predictions)
	standX = allData[r:(trainingSize+r), 1:bCols]
    nRows = size(Xtrain)[1]
    trainingData = Xtrain[:,1:nCols-1]
	standY = allData[r:(trainingSize+r), bCols+1]
	standXVali  = allData[(trainingSize+r+1):(trainingSize+r+predictions), 1:bCols]
	standYVali  = allData[(trainingSize+r+1):(trainingSize+r+predictions), bCols+1]
	bestObjtives = AALR_Time_Run(standX, standY, standXVali, standYVali, trainingData, r)
end
)

#writedlm(fileName*"_solution"*string(ARGS[1])*".CSV",solArr,",")
#writedlm(fileName*"_realArray"*string(ARGS[1])*".CSV",realArr,",")
#writedlm(fileName*"_ISRSquared"*string(ARGS[1])*".CSV",ISRSquared,",")
#save(fileName*"bSolMatrix"*string(ARGS[1])*".jld", "data", bSolMatrix)

#=
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
	Gurobi.writeproblem(stage2Model, "testproblem2.lp")
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

best3Beta

f = open(fileName*"AALRBestK.csv", "w")
write(f, "Rsquared,k,gamma,"*expandedColNamesToString(colNames)*"\n")
writecsv(f,best3Beta)
close(f)
#println(getRMSE(standXTest, standYTest, best3Beta[1,4:111]))
#println(getRMSE(standXTest, standYTest, best3Beta[2,4:111]))
#println(getRMSE(standXTest, standYTest, best3Beta[3,4:111]))
Gurobi.writeproblem(stage2Model, "testproblem3.lp")
=#
