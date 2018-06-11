#Load Packages
using JuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

### Set path ###
#Esben's path
#cd("$(homedir())/Documents/GitHub/Thesis/Data")
#path = "$(homedir())/Documents/GitHub/Thesis/Data"

#Skipper's path
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/IndexDataDiff"

#HPC path
path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data/IndexDataDiff/"

### Load data ###
#mainData = loadIndexDataNoDur(path)
mainData = loadIndexDataLOGReturn("NoDur", path)
path = "/zhome/9f/d/88706/SpecialeCode/Results/IndexData/SpeedTest/"
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/SpeedTest/"

#Reset HPC path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML"
#cd(path)

# Split data
dateAndReseccion = Array(mainData[:,end-1:end])
mainDataArr = Array(mainData[:,1:end-2])

colNames = names(mainData)

nRows = size(mainDataArr)[1]
nCols = size(mainDataArr)[2]

mainXarr = mainDataArr[:,1:nCols-1]
mainYarr = mainDataArr[:, nCols:nCols]

# Transform with time elements
mainXarr = expandWithTime3612(mainXarr)

# Transform with non linear expansions and techincal indicators
mainXarr = expandWithTransformations(mainXarr)
mainXarr = expandWithMAandMomentum(mainXarr, mainYarr, (nCols-1))

# Standardize data
standX = zScoreByColumn(mainXarr)
standY = mainDataArr[:, nCols:nCols]
allData = hcat(standX, standY)
bCols = size(standX)[2]
nRows = size(standX)[1]

# Set prediction parameters
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
# Initialise arrays
amountOfGammas = 5
realArr = zeros((nRows-trainingSize-predictions), 1)
ISRSquared = zeros((nRows-trainingSize-predictions), kmax*amountOfGammas)
bSolMatrix = zeros((nRows-trainingSize-predictions), kmax*amountOfGammas, bCols)

### STAGE 2 ###
println("STAGE 2 INITIATED")
#Initialise values for check later
bSolved = []
bestBeta = []

bigM = 10
#Spaced between 0 and half the SSTO since this would then get SSTO*absSumOfBeta which would force everything to 0
gammaArray = log10.(logspace(0.001, 5, amountOfGammas))

function solveAndLogForAllK(kmax, standXVali, standYVali, r, warmstart)
    # Initialise values
    best3Beta = zeros(3,bCols+3)
    gammaArray = log10.(logspace(5, 0.01, 3))
    println("Remember to check gammaArray line 121")
    bSolved = zeros(bCols,1)
    bestObjtives = zeros(kmax, length(gammaArray))
    println("Warmstart = ",warmstart)


    HC = cor(standX)
	for i in 1:kmax
        if warmstart == true
            bSolved, s = getBetaAndGamma(standX,standY,i)
        end
		for g in 1:length(gammaArray)
			gamma = gammaArray[g]

        	HCPairCounter = 0

            #println("Building model")
        	#Define parameters and model
        	stage2Model = JuMP.Model(solver = GurobiSolver(TimeLimit = 30, OutputFlag = 0, Threads = 2, PreMIQCPForm=0, MIPFocus=1, ImproveStartTime=40));

        	#Define variables
        	@variable(stage2Model, b[1:bCols], start = 0) #Beta values

        	#Define binary variable (5b)
        	@variable(stage2Model, 0 <= z[1:bCols] <= 1, Bin, start = 0)

        	@variable(stage2Model, v[1:bCols], start = 0) # auxiliary variables for abs

            # Set warmstart values
            if warmstart == true || g>1
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
            realArr[r,1] = standYVali[1]

		end
		#printNonZeroValues(bSolved)
	end
    println(warmstart)
    return bestObjtives
	#return #best3Beta, model
end

function solveAndLogForAllKHeuristics(kmax, standXVali, standYVali, r)
    # Initialise values
    best3Beta = zeros(3,bCols+3)
    gammaArray = log10.(logspace(5, 0.01, 3))
    bSolved = zeros(bCols,1)
    bestObjtives = zeros(kmax, length(gammaArray))
    #println("Warmstart heuristics")


    HC = cor(standX)

    #Define an array of variabels that can't be picked to gether
    HCArray = zeros(length(find(x -> x>=0.8, HC)),2)
    rho = 0.8
    itCounter = 1
    for k=1:size(standX)[2]
        for j=1:size(standX)[2]
            if k != j
                if HC[k,j] >= rho
                    HCArray[itCounter,1] = k
                    HCArray[itCounter,2] = j
                    itCounter += 1
                end
            end
        end
    end
    HCArray = HCArray[1:find(x -> x==0,HCArray[:,1])[1]-1,:]

	for i = 1:kmax
        bSolved = gradDecent(standX, zscore(standY), 100000, 1e-3, i, HC, 0, HCArray)
        tempX = find(bSolved)
        bSolved = zeros(bSolved)
        tempbSolved = inv(standX[:,tempX]'*standX[:,tempX])*standX[:,tempX]'*standY
        bSolved[tempX] = tempbSolved

        for g in 1:length(gammaArray)
			gamma = gammaArray[g]

            find(bSolved)

        	HCPairCounter = 0

            #println("Building model")
        	#Define parameters and model
        	stage2Model = JuMP.Model(solver = GurobiSolver(TimeLimit = 30, OutputFlag = 1, Threads = 2, PreMIQCPForm=0, MIPFocus=1, ImproveStartTime=40));

        	#Define variables
        	@variable(stage2Model, b[1:bCols], start = 0) #Beta values

        	#Define binary variable (5b)
        	@variable(stage2Model, 0 <= z[1:bCols] <= 1, Bin, start = 0)

        	@variable(stage2Model, v[1:bCols], start = 0) # auxiliary variables for abs

            # Set warmstart values
            for j in find(bSolved)
				setvalue(b[j], bSolved[j])
			    setvalue(v[j], abs(bSolved[j]))
				setvalue(z[j], 1)
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
            realArr[r,1] = standYVali[1]
		end
		#printNonZeroValues(bSolved)
	end
    println(warmstart)
    return bestObjtives
	#return #best3Beta, model
end

"""
Function that builds and solve the LASSO regression
"""
function solveLasso(Xtrain, Ytrain, gamma)
	bCols = size(Xtrain)[2]

    #Define model
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

    #Shrink values to 0
	for i in 1:length(bSolved)
		if bSolved[i] <= 1e-6
			if bSolved[i] >= -1e-6
				bSolved[i] = 0
			end
		end
	end

    #Count non zero prediction terms
	k = countnz(bSolved)
	return bSolved, k
end

"""
Function to search for lasso solution that uses a specific number of non zero variables k
"""
function getBetaAndGamma(standX,standY, kmax)
    # Initialise parameters
    bSolved=zeros(bCols)
	k=1000 #
	iter = 250 #start gamma index
    iterCounter = 1
	gammaArray = log10.(logspace(0, 10, 500)) #Searchs space
    distance = 125 # distance iter is moved in 1st iteration

    #Trying to find k=kmax within 25 iterations
	while (k!=kmax && iterCounter <= 25)
        if iter>500 || iter <1
            break
        end

        # solve lasse at current gamma and get k non zeros variables
		bSolved, k = solveLasso(standX,standY,gammaArray[iter])
        #println("k= $k \t distance = $distance, iter = $iter")
        if k > kmax
            iter += distance
        else
            iter -= distance
        end
        distance = Int64(ceil(distance/2)) #Half distance to narrow search space
        iterCounter += 1
		#println("Iteration: ",iter," gamma=",gammaArray[iter])
	end

    # If k is not found in the search, bruce force is used to find k
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
	return bSolved, k
end

"""
Build and solve function for stage 2 used in math heuristic
"""
function buildAndSolveProblem(kmax, standX, standY, r, gamma, leftBranchMatrix, rightBranchMatrix, searchSize, HC, rightBranchBool)
    #Define parameters and model
    stage2Model = JuMP.Model(solver = GurobiSolver(TimeLimit = 5, OutputFlag = 0, Threads = 2, PreMIQCPForm=0, MIPFocus=1))

    #Define variables
    @variable(stage2Model, b[1:bCols], start = 0) #beta variables
    @variable(stage2Model, 0 <= z[1:bCols] <= 1, Bin, start = 0) #binary variables
    @variable(stage2Model, v[1:bCols], start = 0) # auxiliary variables for abs(beta)
    @variable(stage2Model, T) #First objective term
    @variable(stage2Model, G) #Second objective term

    #Set warmstart solution
    for j in find(leftBranchMatrix)
        setvalue(b[j], leftBranchMatrix[j])
        setvalue(v[j], abs(leftBranchMatrix[j]))
        setvalue(z[j], 1)
    end

    #Define objective function (5a)
    @objective(stage2Model, Min, T+G)

    #Define SOC constraint
    @constraint(stage2Model, soc, norm( [1-T;2*(standX*b-standY)] ) <= 1+T)

    #Define constraints (5c)
    @constraint(stage2Model, conBigMN, -1*b .<= bigM*z) #from 0 to bCols -1
    @constraint(stage2Model, conBigMP,  1*b .<= bigM*z) #from bCols to 2bCols-1

    #Second objective term
    @constraint(stage2Model, 1*b .<= v) #from 2bCols to 3bCols-1
    @constraint(stage2Model, -1*b .<= v) #from 3bCols to 4bCols-1
    #gamma[g]*oneNorm <= G ---> -G <= -gamma[g]*oneNorm --> G >= gamma[g]*oneNorm
    @constraint(stage2Model, gammaConstr, gamma*ones(bCols)'*v <= G) #4bCols

    #Define kmax constraint (5d)
    @constraint(stage2Model, kMaxConstr, sum(z[j] for j=1:bCols) <= kmax) #4bCols+1

    #Constraint 5f - can only select one of a pair of highly correlated features
    rho = 0.8
    for k=1:bCols
        for j=1:bCols
            if k != j
                if HC[k,j] >= rho
                    @constraint(stage2Model,z[k]+z[j] <= 1) #from 4bCols+1 to (4bCols+1+HCPairCounter)
                end
            end
        end
    end

    #Constraint (5g) - only one transformation allowed (x, x^2, log(x) or sqrt(x))
    for j=1:(nCols-1)
        @constraint(stage2Model, z[j]+z[j+(nCols-1)]+z[j+2*(nCols-1)]+z[j+3*(nCols-1)] <= 1) #from (4bCols+1+HCPairCounter) to (4bCols+1+HCPairCounter+nCols)
    end

    #=
    Adding empty cuts to possibly contain cuts. Can generate a maximum of 6 cuts for
    each stage 3 and default value is 3 iterations between stage 2 and stage 3,
    so 18 empty constraints are added + 1 for fixing the solution
    =#
    nEmptyCuts = 19
    zOnes = ones(bCols)
    for c = 1:nEmptyCuts
        @constraint(stage2Model, sum(zOnes[j]*z[j] for j=1:bCols) <= bCols) #(4bCols+1+HCPairCounter+nCols) to (4bCols+1+HCPairCounter+nCols+18)
    end

    #Adding local branching
    #sum of binary values selected (1-z_j) + sum(z_j not selected) <= kmax
    # Loop over number of bsolved in branchingmatrix
    if leftBranchMatrix != 0
        @constraint(stage2Model, sum(1-z[j] for j=find(leftBranchMatrix))
            + sum(z[j] for j=find(x -> x == 0, leftBranchMatrix)) <= searchSize)
    end

    if rightBranchBool
        println("Added right branch")
        if leftBranchMatrix != rightBranchMatrix
            for row in 2:size(rightBranchMatrix)[2]
                @constraint(stage2Model, sum(1-z[j] for j=find(rightBranchMatrix[:,row]))
                    + sum(z[j] for j=find(x -> x == 0, rightBranchMatrix[:,row])) >= searchSize + 1)
            end
        end
    end

    #Solve Stage 2 model
    status = solve(stage2Model)

    if status != :InfeasibleOrUnbounded
        bSolved = getvalue(b)
        obj = Gurobi.getobjval(internalmodel(stage2Model))
        bound = Gurobi.getobjbound(internalmodel(stage2Model))
        gap = abs.(-1+bound/obj)
        #println("Objective is: ", obj, "\tGap is: ",gap)
        return bSolved, status, obj, gap
    else # if model is InfeasibleOrUnbounded all beta = 0 and obj and gap is set to a high value
        #println("\nWriting Problem!\n")
        #writeLP(stage2Model, "TestLP.LP"; genericnames=false)
        bSolved = zeros(bCols)
        obj = 9999
        gap = 9999
        return bSolved, status, obj, gap
    end

    return bSolved, status, obj, gap
end

function solveAndLogForAllKMath(kmax, standXVali, standYVali, r)
	status = :Optimal
    HC = cor(standX)
    gammaArray = log10.(logspace(5, 0.01, 3))
    bestObjtives = zeros(kmax, length(gammaArray))
	for i in 1:kmax
		lassoBsolved, k = getBetaAndGamma(standX,standY, i)   #Find warmstarted solution
		prediction = 0
        bSolved = zeros(bCols)
		for g in 1:length(gammaArray)
            bestbSolved = zeros(bCols)
            #println("\nNew Gamma \n")
            if g == 1
                leftBranchMatrix = lassoBsolved
                rightBranchMatrix = lassoBsolved
            else
                leftBranchMatrix = bSolved
                rightBranchMatrix = bSolved
            end
            #println("kMax = ",i," g = ", g)

            #Initialise parameters
            searchSize = 2
			oldObj = 1000.0
            obj = 1000.0
            bestObj = 9999
			gamma = gammaArray[g]
            iter = 1
            rightBranchBool = false
			while (true) #Continues the while loop is broken by a condition within the loop
                #println("Iteration $iter")
                oldObj = obj # Update old objective value

                bSolved, status, obj, gap = buildAndSolveProblem(i, standX, standY, r, gamma,
                    leftBranchMatrix, rightBranchMatrix, searchSize, HC, rightBranchBool) #Solve local problem
                #println("Kmax = $i and bsolved = ", countnz(bSolved))

                # InfeasibleOrUnbounded case
                if status == :InfeasibleOrUnbounded
                    #println("Went into ",status)
                    # End loop as last node can't be solved and the last optimal solve was with a worse objective
                    if (typeof(leftBranchMatrix) == Int64 || typeof(leftBranchMatrix) == Float64) && iter != 1
                        break
                    end

                    # Convert leftBranchMatrix into rightBranchMatrix with RHS >= searchSize
                    rightBranchMatrix = hcat(rightBranchMatrix, leftBranchMatrix)

                    # Delete leftBranchMatrix, bSolve = 0 if status = InfeasibleOrUnbounded
                    leftBranchMatrix = bSolved

                    # Weak diversification --> Increase searchSize + 1
                    if searchSize >= i+1 # Ensure that there is a k <= kmax constraint / limit
                        break
                    else
                        if i >= 8 && searchSize < i-2 # go quicker through small search areas when kMax is big.
                            searchSize += 2
                        else
                            searchSize += 1
                        end
                    end

                    iter += 1
                    # Rerun algorithm
                    continue
                end

                # Returns optimal with better solution
                if status == :Optimal && obj+1e-3 < bestObj
                    #println("Went into ",status, ", with better objective")
                    bestObj = obj
                    bestbSolved = bSolved

                    # Add constraint to rightBranchMatrix Delta(x,x_old) >= k + 1
                    # Diversify due to an optimale worse solution previously found or due to solving the last node to optimality
                    if typeof(leftBranchMatrix) == Int64 || typeof(leftBranchMatrix) == Float64
                        if searchSize < i+1
                            if i >= 8 && searchSize < i-2
                                searchSize += 2
                            else
                                searchSize += 1
                            end
                            iter += 1
                            continue
                        else
                            break
                        end
                    end

                    #println("leftBranchMatrix = ", leftBranchMatrix)
                    rightBranchBool = true
                    rightBranchMatrix = hcat(rightBranchMatrix, leftBranchMatrix)

                    # Add constraints to leftBranchMatrix Delta(x,x_new) <= k
                    leftBranchMatrix = bSolved

                    iter += 1
                    # Rerun algorithm
                    #println("Status = ", status,"\tObjective = ", obj, "\tgap = ", gap)
                    continue
                end

                # Return optimal with same or worse solution
                if status == :Optimal && obj+1e-3 >= bestObj
                    println("Went into ",status, ", with worse objective")

                    # Diversify due to an optimale worse solution previously found or due to solving the last node to optimality
                    if typeof(leftBranchMatrix) == Int64 || typeof(leftBranchMatrix) == Float64
                        if searchSize < i+1
                            if i >= 8 && searchSize < i-2
                                searchSize += 2
                            else
                                searchSize += 1
                            end
                        else
                            #Break loop as we can't diversify any more.
                            break
                        end
                    end

                    # Diversify due to an optimale worse solution previously found
                    if typeof(leftBranchMatrix) == Int64 || typeof(leftBranchMatrix) == Float64
                        leftBranchMatrix = 0
                        iter += 1
                        continue
                    else #First time we see optimal with worse solution -- Add leftbranch constraint to right branch and tries to solve rootnode
                        rightBranchBool = true
                        rightBranchMatrix = hcat(rightBranchMatrix, leftBranchMatrix)
                    end
                    leftBranchMatrix = 0 #Set leftBranchMatrix to an integer/float value

                    # Add constraint to rightBranchMatrix Delta(x,x_old) >= k + 1
                    # Solve to depth / timeLimit reached

                    iter += 1
                    continue
                end

                # Return TimeLimit / UserLimit with better solution
                if status == :UserLimit && obj+1e-3 < bestObj
                    #println("Went into ",status, ", with better objective")
                    bestObj = obj
                    bestbSolved = bSolved

                    # Update constraints to leftBranchMatrix Delta(x,x_new) <= k
                    leftBranchMatrix = bSolved

                    # Rerun algorith
                    iter += 1
                    continue
                end

                # Return TimeLimit / UserLimit with worse solution
                if status == :UserLimit && obj+1e-3 >= bestObj
                    #println("Went into ",status, ", with worse objective")
                    # End algorithm under the argument that the stanadard implementation would have solved it by now
                    iter += 1
                    break
                end
			end #While loops ends

            bestObjtives[Int64(i), Int64(g)] = bestObj

			#println("Number of non zero actually: ", countnz(bSolved))

			#println("Prediction =$prediction\t Real =$standYVali\t kMax =$i \t gamma =$g")
			#realArr[r,1] = standYVali[1]
            prediction = standXVali*bestbSolved

			println("Prediction =$prediction\t Real =$standYVali\t kMax =$i \t gamma =$g \t obj = $(bestObjtives[Int64(i),Int64(g)])")
		end

	end
    return bestObjtives
end

function AALR_Time_Run(standX, standY, standXVali, standYVali, trainingData, r)
    nRows = size(standX)[1]


    bSample = []
    allCuts = []
    signifBoolean = zeros(3)

    #mathheuristic
    warmstart = false
	bestObjtivesMath = @time(solveAndLogForAllKMath(kmax, standXVali, standYVali, r))
    println("This was time for mathheuristic")

    #Nonwarmstart
    warmstart = false
    bestObjtivesNormal = @time(solveAndLogForAllK(kmax, standXVali, standYVali, r, warmstart))
    print("This was time for non warmstart")
    println("")

    #lasso warmstart
    warmstart = true
    bestObjtivesWarmstart = @time(solveAndLogForAllK(kmax, standXVali, standYVali, r, warmstart))
    println("This was time for warmstart")

    #Heuristic warmstart
    println("Fucking start here!")
    bestObjtivesHeuristic = @time(solveAndLogForAllKHeuristics(kmax, standXVali, standYVali, r))
    println("This was time for heuristc")

    return bestObjtivesNormal, bestObjtivesWarmstart, bestObjtivesMath, bestObjtivesHeuristic
end

SSTO = sum((standY[i]-mean(standY[:]))^2 for i=1:length(standY))
#gammaArray = log10.(logspace(0.01, SSTO, 3))
gammaArray = log10.(logspace(0.01, 5, 3))

bestObjtivesNormal = 1
bestObjtivesWarmstart = 1
bestObjtivesMath = 1

#writedlm(path*"/bestNorm.csv",bestObjtivesNormal)
#writedlm(path*"/bestWarm.csv",bestObjtivesWarmstart)
#writedlm(path*"/bestMath.csv",bestObjtivesMath)

r=1
for r = 1:100:501#(nRows-trainingSize-predictions)
	standX = allData[r:(trainingSize+r), 1:bCols]
    nRows = size(Xtrain)[1]
    trainingData = Xtrain[:,1:nCols-1]
	standY = allData[r:(trainingSize+r), bCols+1]
	standXVali  = allData[(trainingSize+r+1):(trainingSize+r+predictions), 1:bCols]
	standYVali  = allData[(trainingSize+r+1):(trainingSize+r+predictions), bCols+1]
	bestObjtivesNormal, bestObjtivesWarmstart, bestObjtivesMath, bestObjtivesHeuristic  = AALR_Time_Run(standX, standY, standXVali, standYVali, trainingData, r)

    writedlm(path*string(r)*"bestNorm.csv",bestObjtivesNormal)
    writedlm(path*string(r)*"bestWarm.csv",bestObjtivesWarmstart)
    writedlm(path*string(r)*"bestMath.csv",bestObjtivesMath)
    writedlm(path*string(r)*"bestHeuristic.csv",bestObjtivesHeuristic)
end
