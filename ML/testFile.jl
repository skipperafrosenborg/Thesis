using JuMP
using Gurobi
##### Simple LP
# Example taken directly from https://jump.readthedocs.org/en/latest/quickstart.html
type NodeData
    time::Float64  # in seconds since the epoch
    node::Int
    obj::Float64
    bestbound::Float64
end

bbdata = NodeData[]



function infocallback(cb)
    node      = cbgetexplorednodes(cb)
    #obj       = cbgetobj(cb)
    bestbound = cbgetbestbound(cb)
    println("HEJ")
    push!(bbdata, NodeData(time(),node,obj,bestbound))
end
println("***** Problem 1 *****")
m = Model(solver = GurobiSolver())
@variable(m, 0 <= x <= 2 )
@variable(m, 0 <= y <= 30 )

@objective(m, Max, 5x + 3*y )
@constraint(m, 1x + 5y <= 3.0 )

print(m)
addinfocallback(m, infocallback, when = :MIP)

status = solve(m)

# Save results to file for analysis later
open("bbtrack2.csv","w") do fp
    println(fp, "time,node,obj,bestbound")
    for bb in bbdata
        println(fp, bb.time, ",", bb.node, ",",bb.obj, ",", bb.bestbound)
    end
end

println("Objective value: ", getobjectivevalue(m))
println("x = ", getvalue(x))
println("y = ", getvalue(y))
