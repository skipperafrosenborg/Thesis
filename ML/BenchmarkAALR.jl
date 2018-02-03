#Pkg.add("CSV");
#Pkg.add("DataFrames");

using DataFrames;
using CSV;

println("Leeeeroooy Jenkins")

cd("$(homedir())/Documents/GitHub/Thesis/Data")
#All on monthly data
mainData = CSV.read("AmesHousingModClean.csv"; delim = ';')
#mainData = CSV.read("AmesHousingMod.csv")

dataSize = size(mainData)
nColsMain = dataSize[2]
nRowsMain = dataSize[1]
colNames = names(mainData)

nColsMain = size(mainData)[2]
#Converting it into a datamatrix instead of a dataframe
mainData2 = Array(mainData[2:nRowsMain, 1:nColsMain])
#stringMatrix = Array(mainData[2:nRowsMain,1:(nColsMain)])
#dataMatrixMain = parse.(Float64, stringMatrix)

using JuMP
using CPLEX

combinedData = copy(mainData2)
nCols = size(combinedData)[2]
nRows = size(combinedData)[1]

m = Model(solver = CplexSolver())
@variable(m, 0 <= z[1:nCols] <= 1, Bin )


HC = cor(combinedData)

@objective(m, Max, sum(z[i] for i=1:length(z)))

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
#@constraint(m, z[2] + z[3] <= 1)

print(m)

status = solve(m)

println("Objective value: ", getobjectivevalue(m))
kmax = getobjectivevalue(m)
zSolved = getvalue(z)
length(zSolved)
for i=1:length(zSolved)
	if zSolved[i]==0
		println("z[$i] = 0")
	end
end
println("STAGE 1 DONE")
println("STAGE 2 INITIATED")
using StatsBase

y = combinedData[1:250,nColsMain]
X = combinedData[1:250,1:nColsMain-1]
standY = zscore(y)
standX = zscore(X)

bCols = size(X)[2]

z = 0
b = 0
stage2Model = Model(solver = CplexSolver())
bigM = 10000
gamma = 10
@variable(stage2Model, 0 <= z[1:bCols] <= 1, Bin )
@variable(stage2Model, b[1:bCols])
@variable(stage2Model, T)
#@variable(stage2Model, O)


@objective(stage2Model, Min, T)
@constraint(stage2Model, norm(standY - standX*b) <= T)

for i=1:bCols
	@constraint(stage2Model, -bigM*z[i] <= b[i])
	@constraint(stage2Model, b[i] <= bigM*z[i])
end

@constraint(stage2Model, sum(z[i] for i=1:bCols) <= kmax)

#=
@variable(stage2Model, v[1:bCols])
for i=1:bCols
	@constraint(stage2Model, b[i]  <= v[i])
	@constraint(stage2Model, -b[i] <= v[i])
end

@constraint(stage2Model, sum(v[i] for i=1:bCols) <= O)
=#


#print(stage2Model)

status = solve(stage2Model)

println("Objective value: ", getobjectivevalue(stage2Model))
bSolved = getvalue(b)
for i=1:length(b)
	if b[i]!=0
		println("b[$i] = ", bSolved[i])
	end
end

println("STAGE 2 DONE")
SSTO = sum((standY[i]-mean(standY))^2 for i=1:length(standY))
Rsquared = 1-(getobjectivevalue(stage2Model))/SSTO
#Rsquared = (getobjectivevalue(stage2Model)-getvalue(O)*gamma)/SSTO
