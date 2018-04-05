println("Test")
#println(ARGS[1])
#println(ARGS[2])

using JuMP
using Gurobi

m = JuMP.Model(solver = GurobiSolver(TimeLimit=0))
@variable(m, 0 <= x <= 2 )
@variable(m, 0 <= y <= 30 )

@objective(m, Max, 5x + 3*y )
@constraint(m, 1x + 5y <= 3.0 )

print(m)

status = solve(m)

println("x = ", getvalue(x))
println("y = ", getvalue(y))
