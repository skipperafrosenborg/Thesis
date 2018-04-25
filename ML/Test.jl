println("Test")
#println(ARGS[1])
#println(ARGS[2])

#=
using JuMP
using Gurobi

m = JuMP.Model(solver = GurobiSolver())
@variable(m, 0 <= x <= 2 )
@variable(m, 0 <= y <= 30 )

@objective(m, Max, 5x + 3*y )
@constraint(m, 1x + 5y <= 3.0 )
@constraint(m, x == 3)

print(m)

status = solve(m)

obj = getobjectivevalue(m)
getobjectivebound(m)
println("Obj = ", obj)
println("ObjBound = ",getobjectivebound(m))
println("x = ", getvalue(x))
println("y = ", getvalue(y))
=#

using StatsBase
using DataFrames
using CSV


path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CPU/"
cd(path)

dataT = [Float64]
for j = 1:29
    dataT = vcat(dataT, Float64)
end

totalLog = zeros(30,30)

for i = 1:10
    println(i)
    mainData = Array(CSV.read("Data/"*string(i)*"Xtrain.csv", delim = ',', nullable=false))
    best3Beta = CSV.read(string(i)*"_CPU.csv", delim = ',', nullable=false, types = dataT)

    best3BetaArr = Array(best3Beta)

    best3Beta[1,1] = findMaxCor(mainData, best3BetaArr[1,7:end])
    best3Beta[2,1] = findMaxCor(mainData, best3BetaArr[2,7:end])
    best3Beta[3,1] = findMaxCor(mainData, best3BetaArr[3,7:end])

    CSV.write(string(i)*"_CPU.csv", best3Beta)

    totalLog[1+(i-1)*3:3+(i-1)*3,:]= Array(best3Beta)
end

writedlm("TotalLog.csv", totalLog, ",")

lassoBestK = zeros(24,30)
lassoSummary = zeros(24,30)
totalLog = zeros(30,29)

dataT = [Float64]
for j = 1:28
    dataT = vcat(dataT, Float64)
end

for i = 1:10
    println(i)
    mainData = Array(CSV.read("Data/"*string(i)*"Xtrain.csv", delim = ',', nullable=false))
    dataInput = CSV.read(string(i)*"_LassoCPU.csv", delim = ',', nullable=false, types = dataT)

    dataInputArr = Array{Float64}(dataInput)
    for j = 1:24
        #println("j = ",j)
        lassoBestK[j,1] = j
        lassoSummary[j,1] = j
        currentK = find(k -> (k==j), dataInputArr[:,4])
        if isempty(currentK)
            continue
        end
        val, index = findmax(dataInputArr[currentK,2])

        lassoSummary[j,2:end] += dataInputArr[currentK[index],:]

        lassoBestK[j,2:end] = dataInputArr[currentK[index],:]
        lassoBestK[j,2] = findMaxCor(mainData, lassoBestK[j,7:end])
    end

    for j = 1:3
        val, index = findmax(dataInputArr[:,3])
        totalLog[j+(i-1)*3:j+(i-1)*3,:]= Array(dataInputArr[index, :])
        dataInputArr[index, 3] = 0
    end

    writedlm(string(i)*"LassoSummary.csv", lassoBestK,",")
end

writedlm("TotalLasso.csv", totalLog, ",")

lassoSummary[:,2:end] = lassoSummary[:,2:end]/10
writedlm("TotalLassoSummary.csv", lassoSummary,",")
