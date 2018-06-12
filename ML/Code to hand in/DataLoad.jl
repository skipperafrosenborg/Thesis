#Code written by Skipper af Rosenborg and Esben Bager
function loadDiabetes(path)
    cd(path)
    mainData = CSV.read("Diabetes/diabetes.csv", delim = ";", nullable=false)
    return mainData
end

function loadWineQuality(path)
    cd(path)
    mainData = CSV.read("WineQualityRed/winequality-red.csv", delim = ";", nullable=false, types=fill(Float64,12))
    return mainData
end

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


function loadCPUData(path)
    cd(path)
    mainData = CSV.read("machine.csv", header=["MYCT",
    	"MMIN","MMAX","CACH","CHMIN","CHMAX","PRP"], datarow=1, nullable=false)
    return mainData
end

function loadIndexDataLOGReturn(industry, path)
    cd(path)
    mainData = CSV.read("monthly"*industry*"LOGReturn2.csv", header=["NoDur","Durbl",
        "Manuf","Enrgy","HiTec","Telcm","Shops", "Hlth","Utils","Other","Index",
        "D12","E12","b.m","tbl","AAA","BAA","lty","ntis","Rfree","infl","ltr",
        "corpr","svar","csp","CRSP_SPvw","CRSP_SPvwx","VIX","Ycol","Date","Resession"],
        delim = ',', nullable=false, types=Dict(25=>Float64, 28=>Float64))
    return mainData
end

function loadRiskFreeRate(industry, path)
    cd(path)
    mainData = CSV.read("riskFreeRateLOG.csv",header =["RfreeRate"],delim = ',', nullable = false)
    return mainData
end
