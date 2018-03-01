function loadHousingData(path)
    cd(path)
    mainData = CSV.read("AmesHousingModClean.csv", delim = ';', nullable=false)
    return mainData
end

function loadConcrete(path)
    cd(path)
    mainData = CSV.read("Concrete_Data.csv", delim = ',', nullable=false)
    return mainData
end

<<<<<<< HEAD
function loadIndexData(path)
    cd(path)
    mainData = CSV.read("combinedIndexDataMod.csv", header=["NoDur","Durbl","Manuf","Enrgy","HiTec","Telcm","Shops",
    "Hlth","Utils","Other","Index","D12","E12","b.m","tbl","AAA","BAA","lty","ntis","Rfree","infl","ltr","corpr","svar","csp","CRSP_SPvw","CRSP_SPvwx"],
    delim = ',', nullable=false, types=Dict(25=>Float64))
    return mainData
end

=======
>>>>>>> master
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
