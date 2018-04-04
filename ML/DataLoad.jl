function loadHousingData(path)
    cd(path)
    mainData = CSV.read("AmesHousingModClean.csv", delim = ';', nullable=false)
    return mainData
end

function loadConcrete(path)
    cd(path)
    mainData = CSV.read("Concrete_Data.csv", delim = ',', nullable=false, types=Dict(3=>Float64))
    return mainData
end

function loadIndexDataNoDur(path)
    cd(path)
    mainData = CSV.read("monthlyNoDurReturn.csv", header=["NoDur","Durbl",
        "Manuf","Enrgy","HiTec","Telcm","Shops", "Hlth","Utils","Other","Index",
        "D12","E12","b.m","tbl","AAA","BAA","lty","ntis","Rfree","infl","ltr",
        "corpr","svar","csp","CRSP_SPvw","CRSP_SPvwx","Ycol"],
        delim = ',', nullable=false, types=Dict(25=>Float64))
    return mainData
end

function loadIndexDataNoDurLOGReturn(path)
    cd(path)
    mainData = CSV.read("monthlyNoDurLOGReturn.csv", header=["NoDur","Durbl",
        "Manuf","Enrgy","HiTec","Telcm","Shops", "Hlth","Utils","Other","Index",
        "D12","E12","b.m","tbl","AAA","BAA","lty","ntis","Rfree","infl","ltr",
        "corpr","svar","csp","CRSP_SPvw","CRSP_SPvwx","VIX","Ycol","Date","Resession"],
        delim = ',', nullable=false, types=Dict(25=>Float64, 28=>Float64))
    return mainData
end

function loadIndexDataNoDurVIX(path)
    cd(path)
    mainData = CSV.read("monthlyNoDurReturnVIX.csv", header=["NoDur","Durbl",
        "Manuf","Enrgy","HiTec","Telcm","Shops", "Hlth","Utils","Other","Index",
        "D12","E12","b.m","tbl","AAA","BAA","lty","ntis","Rfree","infl","ltr",
        "corpr","svar","csp","CRSP_SPvw","CRSP_SPvwx","VIX","Ycol"],
        delim = ',', nullable=false, types=Dict(25=>Float64))
    return mainData
end

function loadIndexDataOther(path)
    cd(path)
    mainData = CSV.read("monthlyOtherReturn.csv", header=["NoDur","Durbl",
        "Manuf","Enrgy","HiTec","Telcm","Shops", "Hlth","Utils","Other","Index",
        "D12","E12","b.m","tbl","AAA","BAA","lty","ntis","Rfree","infl","ltr",
        "corpr","svar","csp","CRSP_SPvw","CRSP_SPvwx","Ycol"],
        delim = ',', nullable=false, types=Dict(25=>Float64))
    return mainData
end

function loadIndexDataDailyNoDur(path)
    cd(path)
    mainData = CSV.read("dailyNoDurReturn.csv", header=["NoDur","Durbl",
        "Manuf","Enrgy","HiTec","Telcm","Shops", "Hlth","Utils","Other","Index"],
        delim = ',', nullable=false, types=Dict(11=>Float64))
    return mainData
end

function loadIndexDataDailyDurbl(path)
    cd(path)
    mainData = CSV.read("dailyDurblReturn.csv", header=["NoDur","Durbl",
        "Manuf","Enrgy","HiTec","Telcm","Shops", "Hlth","Utils","Other","Index"],
        delim = ',', nullable=false, types=Dict(11=>Float64))
    return mainData
end

function loadIndexDataDailyManuf(path)
    cd(path)
    mainData = CSV.read("dailyManufReturn.csv", header=["NoDur","Durbl",
        "Manuf","Enrgy","HiTec","Telcm","Shops", "Hlth","Utils","Other","Index"],
        delim = ',', nullable=false, types=Dict(11=>Float64))
    return mainData
end

function loadIndexDataDailyEnrgy(path)
    cd(path)
    mainData = CSV.read("dailyEnrgyReturn.csv", header=["NoDur","Durbl",
        "Manuf","Enrgy","HiTec","Telcm","Shops", "Hlth","Utils","Other","Index"],
        delim = ',', nullable=false, types=Dict(11=>Float64))
    return mainData
end

function loadIndexDataDailyHiTec(path)
    cd(path)
    mainData = CSV.read("dailyHiTecReturn.csv", header=["NoDur","Durbl",
        "Manuf","Enrgy","HiTec","Telcm","Shops", "Hlth","Utils","Other","Index"],
        delim = ',', nullable=false, types=Dict(11=>Float64))
    return mainData
end

function loadIndexDataDailyTelcm(path)
    cd(path)
    mainData = CSV.read("dailyTelcmReturn.csv", header=["NoDur","Durbl",
        "Manuf","Enrgy","HiTec","Telcm","Shops", "Hlth","Utils","Other","Index"],
        delim = ',', nullable=false, types=Dict(11=>Float64))
    return mainData
end

function loadIndexDataDailyShops(path)
    cd(path)
    mainData = CSV.read("dailyShopsReturn.csv", header=["NoDur","Durbl",
        "Manuf","Enrgy","HiTec","Telcm","Shops", "Hlth","Utils","Other","Index"],
        delim = ',', nullable=false, types=Dict(11=>Float64))
    return mainData
end

function loadIndexDataDailyHlth(path)
    cd(path)
    mainData = CSV.read("dailyHlthReturn.csv", header=["NoDur","Durbl",
        "Manuf","Enrgy","HiTec","Telcm","Shops", "Hlth","Utils","Other","Index"],
        delim = ',', nullable=false, types=Dict(11=>Float64))
    return mainData
end

function loadIndexDataDailyUtils(path)
    cd(path)
    mainData = CSV.read("dailyUtilsReturn.csv", header=["NoDur","Durbl",
        "Manuf","Enrgy","HiTec","Telcm","Shops", "Hlth","Utils","Other","Index"],
        delim = ',', nullable=false, types=Dict(11=>Float64))
    return mainData
end

function loadIndexDataDailyOther(path)
    cd(path)
    mainData = CSV.read("dailyOtherReturn.csv", header=["NoDur","Durbl",
        "Manuf","Enrgy","HiTec","Telcm","Shops", "Hlth","Utils","Other","Index"],
        delim = ',', nullable=false, types=Dict(11=>Float64))
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
