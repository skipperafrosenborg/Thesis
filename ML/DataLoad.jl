function loadHousingData(path)
    cd(path)
    mainData = CSV.read("AmesHousingModClean.csv", delim = ';', nullable=false)
    return mainData
end

function loadCPUData(path)
    cd(path)
    mainData = CSV.read("machine.data", header=["vendor name","Model name","MYCT",
    	"MMIN","MMAX","CACH","CHMIN","CHMAX","PRP","ERP"], datarow=1, nullable=false)
    mainData = copy(mainData[:,3:9])
    #delete!(mainData, :PRP)
    return mainData
end

function loadElevatorData(path)
    cd(path)
    mainData = CSV.read("Elevators/elevators.data", header=["climbRate", "Sgz", "p", "q", "curRoll", "absRoll", "diffClb",
    	"diffRollRate", "diffDiffClb", "SaTime1", "SaTime2", "SaTime3", "SaTime4", "diffSaTime1", "diffSaTime2",
    	"diffSaTime3", "diffSaTime4", "Sa", "Goal"], datarow=1, nullable=false)
    testData = CSV.read("Elevators/elevators.test", header=["climbRate", "Sgz", "p", "q", "curRoll", "absRoll", "diffClb",
    	"diffRollRate", "diffDiffClb", "SaTime1", "SaTime2", "SaTime3", "SaTime4", "diffSaTime1", "diffSaTime2",
    	"diffSaTime3", "diffSaTime4", "Sa", "Goal"], datarow=1, nullable=false)
    return mainData, testData
end
