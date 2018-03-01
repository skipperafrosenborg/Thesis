function stageThree(bestBeta1, bestK1, bestGamma1, bestBeta2, bestK2, bestGamma2, bestBeta3, bestK3, bestGamma3, X, Y, allCuts)
	#Condition Number
	#A high condition number indicates a multicollinearity problem. A condition
	# number greater than 15 is usually taken as evidence of
	# multicollinearity and a condition number greater than 30 is
	# usually an instance of severe multicollinearity
	summary = zeros(3)
	bCols = size(X)[2]
	nRows = size(X)[1]
	cuts = Matrix(0, bCols+1)
	rowsPerSample = nRows #All of rows in training data to generate beta estimates, but selected with replacement
	totalSamples = 25 #25 different times we will get a beta estimate
	nBoot = 10000
	for i=1:3
		xColumns = []
		bSample = Matrix(totalSamples, bCols)
		if i == 1
			bZeros = zeros(bCols)
			for j = 1:bCols
				if !isequal(bestBeta1[j],0)
					push!(xColumns, j)
				end
			end
			selectedX = X[:,xColumns]
			condNumber = cond(selectedX)
			if condNumber >= 15
				bZeros[xColumns] = 1
				subsetSize = size(xColumns)[1]
				newCut1 = [bZeros' subsetSize]
				cuts = [cuts; newCut1]
				println("A cut based on Condition number = $condNumber has been created from Beta$i")
			end

			bestZ1 = zeros(bestBeta1)
			for l=1:size(bestBeta1)[1]
				if bestBeta1[l] != 0
					bestZ1[l] = 1
				end
			end

			# test significance
			bZeros = zeros(bCols)
			createBetaDistribution(bSample, X, Y, bestK1, totalSamples, rowsPerSample,  bestGamma1, allCuts, bestZ1) #standX, standY, k, sampleSize, rowsPerSample

			confArray99 = createConfidenceIntervalArray(bSample, nBoot, 0.99)
			confArray95 = createConfidenceIntervalArray(bSample, nBoot, 0.95)
			confArray90 = createConfidenceIntervalArray(bSample, nBoot, 0.90)

			significanceResult = testSignificance(confArray99, confArray95, confArray90, bestBeta1)
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
				newCut1 = [bZeros' subsetSize]
				cuts = [cuts; newCut1]
				println("A cut based on parameters being non-significant in Beta$i has been created")
			end
			if isempty(significanceResultNONSI)
				summary[1]=1
			end


		elseif i == 2
			bZeros = zeros(bCols)
			for j = 1:bCols
				if !isequal(bestBeta1[j],0)
					push!(xColumns, j)
				end
			end
			selectedX = X[:,xColumns]
			condNumber = cond(selectedX)
			if condNumber >= 15
				bZeros[xColumns] = 1
				subsetSize = size(xColumns)[1]
				newCut2 = [bZeros' subsetSize]
				status = false
				if isequal(Array(newCut1), Array(newCut2))
					println("IDENTICAL CUTS #############################################")
				end

				if !isempty(cuts)
					cutRows = size(cuts)[1]
					for r = 1:cutRows
						if isequal(cuts[r,:], newCut)
							status = true
							println("Identical cut found")
						end
					end
				end
				if status == false
					cuts = [cuts; newCut]
				end
				println("A cut based on Condition number = $condNumber has been created from Beta$i")
			end


			bestZ2 = zeros(bestBeta2)
			for l=1:size(bestBeta2)[1]
				if bestBeta2[l] != 0
					bestZ2[l] = 1
				end
			end

			# test significance
			bZeros = zeros(bCols)
			createBetaDistribution(bSample, X, Y, bestK2, totalSamples, rowsPerSample,  bestGamma2, allCuts, bestZ2) #standX, standY, k, sampleSize, rowsPerSample
			confArray99 = createConfidenceIntervalArray(bSample, nBoot, 0.99)
			confArray95 = createConfidenceIntervalArray(bSample, nBoot, 0.95)
			confArray90 = createConfidenceIntervalArray(bSample, nBoot, 0.90)

			significanceResult = testSignificance(confArray99, confArray95, confArray90, bestBeta2)
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
				summary[2]=1
			end

		else
			bZeros = zeros(bCols)
			for j = 1:bCols
				if !isequal(bestBeta3[j],0)
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

			bestZ3 = zeros(bestBeta3)
			for l=1:size(bestBeta3)[1]
				if bestBeta3[l] != 0
					bestZ3[l] = 1
				end
			end
			# test significance
			bZeros = zeros(bCols)
			createBetaDistribution(bSample, X, Y, bestK3, totalSamples, rowsPerSample,  bestGamma3, allCuts, bestZ3) #standX, standY, k, sampleSize, rowsPerSample
			confArray99 = createConfidenceIntervalArray(bSample, nBoot, 0.99)
			confArray95 = createConfidenceIntervalArray(bSample, nBoot, 0.95)
			confArray90 = createConfidenceIntervalArray(bSample, nBoot, 0.90)

			significanceResult = testSignificance(confArray99, confArray95, confArray90, bestBeta3)
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
				summary[3]=1
			end

		end
	end
	if summary[1] == 1
		println("Beta1 is significant!")
	end
	if summary[2] == 1
		println("Beta2 is significant!")
	end
	if summary[3] == 1
		println("Beta3 is significant!")
	end
	return cuts
end
