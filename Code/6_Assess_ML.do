* =================================== *
*	Prepare
* =================================== *
use "Data/Modified/NO2", clear
append using "Data/Modified/PM2.5.dta"

*	Tidy up
replace dt = dofc(dt)
format dt %td
gsort idsensore dt
gen post = (dt>=mdy(2,22,2020))
lab var post "Lockdown"
gen diff = Observed - Predicted
lab var diff "$ \Delta_{Observed,Counterfactual} $"
bys idsensore: gegen x = mean(Predicted) if post==1
bys idsensore: gegen baseline = mean(x)
drop x

replace type = proper(type)
encode type, gen(tid)

*	Retrieve station metadata
merge m:1 idsensore using "Data/Modified/PollutionStations", keep(1 3) nogen keepus(comune nomestazione pollutantshort quota lat lng)
gen stationdetail = nomestazione + " - " + pollutantshort

egen x = tag(comune idsensore) if pollutantshort=="PM2.5"
bys comune: egen x2 = total(x)
gen poppm25 = pop/x2
drop x x2

egen x = tag(comune idsensore) if pollutantshort=="NO2"
bys comune: egen x2 = total(x)
gen popno2 = pop/x2
drop x x2

gsort idsensore dt
order idsensore dt Observed Predicted diff post, first
save "Data/Modified/TidyResults", replace

* =========================================================================== *
*	Performance table
* =========================================================================== *
use "Data/Modified/TidyResults", clear
keep idsensore pollutantshort test* train*
rename test_* *_test
rename train_* *_train
duplicates drop
reshape long mean_bias_ nmean_bias_ rmse_ crmse_ ncrmse_ pcc_ r2_, i(idsensore pollutantshort) j(set) string
gcollapse (mean) pcc_ mean_bias_ nmean_bias_ rmse_ crmse_ ncrmse_ , by(pollutantshort set)

foreach x of var pcc_ mean_bias_ nmean_bias_ rmse_ crmse_ ncrmse_ {
	replace `x' = round(`x', 0.001)
}
replace set = "Test" if set=="test"
replace set = "Train" if set=="train"
gsort pollutantshort -set

lab var pollutantshort "Pollutant"
lab var set "Dataset"
lab var mean_bias_ "MB"
lab var nmean_bias_ "nMB"
lab var rmse_ "RMSE"
lab var crmse_ "cRMSE"
lab var ncrmse_ "ncRMSE"
lab var pcc_ "Corr"
texsave using "Docs/tables/evaluation.tex", replace frag varlab nofix title("Accuracy of predictions, average values across monitors") align(llccccccc)  marker("table:evaluation") location(h) ///
	footnote("\textit{Notes: } \textit{Corr}: Pearson's correlation coefficient. \textit{MB}: Mean bias, where negative values indicate observed values below predicted values. \textit{nMB}: Normalized mean bias. \textit{RMSE}: Root mean squared error. \textit{nRMSE}: Normalized RMSE. \textit{cRMSE}: Centered RMSE. \textit{ncRMSE}: Normalized centered RMSE. Mean bias, RMSE and centered RMSE are expressed in $\mu g/m^3$. Mean bias, RMSE and centered RMSE are normalized dividing by mean observed concentrations. The centered RMSE is computed as $\big[ 1/N \sum (\widehat{y}_i  -\bar{\widehat{y}} - y_i + \bar{y})^2 \big]^{1/2}$.")

* =========================================================================== *
*	Summary of monitors
* =========================================================================== *
use "Data/Modified/TidyResults", clear
egen x = tag(pollutantshort type comune)
bys pollutantshort type: egen ncomuni = total(x)
drop x
egen x = tag(pollutantshort type idsensore)
bys pollutantshort type: egen nsensori = total(x)
drop x
gcollapse (first) ncomuni nsensori, by(pollutantshort type)
lab var pollutantshort "Pollutant"
lab var type "Type of monitor"
lab var ncomuni "Number of municipalities"
lab var nsensori "Number of monitors"
texsave using "Docs/tables/monitors.tex", replace frag varlab width(\hsize) nofix title("Pollution monitors by type.") marker("table:monitors") location(h) ///
	footnote("Background stations measure pollutions concentrations that are representative of the average exposure of the general population, or vegetation. Industrial stations are located in close proximity to an industrial area or an industrial source. Traffic stations are located in close proximity to a single major road.")

* =========================================================================== *
*	Analysis
* =========================================================================== *

* =================================== *
*	Parametric - OLS
* =================================== *

capture program drop getbaseline
program define getbaseline, sclass
	
	syntax [, PM NO Background Industrial Traffic]

	// Get pvalue fot the constant
	mat v = r(table)
	qui mat li v
	tempname rn cn conspvalue
	scalar `rn' = rownumb(v, "pvalue")
	scalar `cn' = colnumb(v, "_cons")
	scalar `conspvalue' = v[`rn', `cn']

	// Get population weights
	if "`pm'"!="" & "`no'"=="" {
		if "`background'"!="" {
			loc weight poppm25b
		}
		else if "`industrial'"!="" {
			loc weight poppm25i
		}
		else if "`traffic'"!="" {
			loc weight poppm25t
		}
		else {
			loc weight poppm25
		}
	}
	else if "`pm'"=="" & "`no'"!="" {
		if "`background'"!="" {
			loc weight popno2b
		}
		else if "`industrial'"!="" {
			loc weight popno2i
		}
		else if "`traffic'"!="" {
			loc weight popno2t
		}
		else {
			loc weight popno2
		}
	}
	else {	// No weights
		loc weight = 1
	}

	// Average counterfactual during lockdown
	di "Weight is `weight'"
	qui sum Predicted [w=`weight'] if e(sample) & post==1
	// Baseline
	if `conspvalue' > 0.1 {
		sreturn local baseline = r(mean)
	}
	else {
		sreturn local baseline = r(mean) + _b[_cons]
	}
end

capture program drop wgt_pollutant_type
program define wgt_pollutant_type

	egen x = tag(comune idsensore) if pollutantshort=="`1'" & type=="`2'"
	bys comune: egen x2 = total(x)
	gen `3' = pop/x2
	drop x x2

end

* --------------- *
*	Weighted - Baseline
* --------------- *
use "Data/Modified/TidyResults", clear
eststo clear
eststo olspm25: reg diff post if pollutantshort=="PM2.5" [w=poppm25], vce(clus idsensore)
getbaseline, pm
estadd scalar Baseline = `s(baseline)'
eststo olsno2: reg diff post if pollutantshort=="NO2" [w=popno2], vce(clus idsensore)
getbaseline, no
estadd scalar Baseline = `s(baseline)'
esttab ols* using "Docs/tables/ols.tex", replace label b(2) se(2) stats(Baseline N, fmt(a3 0) lab("Average baseline concentration" "Observations")) star(* 0.1 ** 0.05 *** 0.01) nogap compress varwidth(30) nodep mtit("PM 2.5" "NO2") sub(\_ _) mgroups("$ \Delta_{Observed,Counterfactual} $", pattern(1 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) booktabs nonote ///
	postfoot("\bottomrule\end{tabular}" "\begin{tablenotes}[flushleft]\footnotesize" ///
		"\item \textit{Notes:} Regression weighted by population within 20 kilometers of a monitoring station. Territory within less than 20 kilometers from two or more monitors is assigned to the closest one." ///
		"The dependent variable is the difference between the observed values and the counterfactual." /// 
		"\textit{Lockdown} is a dummy variable equal to 0 from January 1, 2020 to February 22, and equal to 1 after February 22, 2020." ///
		"\textit{Average baseline concentration} is the population-weighted average of counterfactual values during the lockdown, less the constant in case the latter is statistically significant at 10\%." ///
		"Standard errors, in brackets, are clustered by monitor. * p$<$0.1, ** p$<$0.05, *** p$<$0.01." ///
		"\end{tablenotes}" "}")

* --------------- *
*	Weighted - By station type
* --------------- *
use "Data/Modified/TidyResults", clear
// Create weights
wgt_pollutant_type "PM2.5" "Background" poppm25b
wgt_pollutant_type "PM2.5" "Industrial" poppm25i
wgt_pollutant_type "PM2.5" "Traffic" poppm25t
wgt_pollutant_type "NO2" "Background" popno2b
wgt_pollutant_type "NO2" "Industrial" popno2i
wgt_pollutant_type "NO2" "Traffic" popno2t

eststo clear
eststo typepm25b: reg diff post if pollutantshort=="PM2.5" & type=="Background" [w=poppm25b], vce( robust)
global didPM25 =_b[post]
getbaseline, pm b
estadd scalar Baseline = `s(baseline)'
distinct idsensore if e(sample)
estadd scalar NStations r(ndistinct)
eststo typepm25i: reg diff post if pollutantshort=="PM2.5" & type=="Industrial" [w=poppm25i], vce( robust)
getbaseline, pm i
estadd scalar Baseline = `s(baseline)'
distinct idsensore if e(sample)
estadd scalar NStations r(ndistinct)
eststo typepm25t: reg diff post if pollutantshort=="PM2.5" & type=="Traffic" [w=poppm25t], vce( robust)
getbaseline, pm t
estadd scalar Baseline = `s(baseline)'
distinct idsensore if e(sample)
estadd scalar NStations r(ndistinct)
eststo typeno2b: reg diff post if pollutantshort=="NO2" & type=="Background" [w=popno2b], vce( robust)
global didNO2 = _b[post]
getbaseline, no b
estadd scalar Baseline = `s(baseline)'
distinct idsensore if e(sample)
estadd scalar NStations r(ndistinct)
eststo typeno2i: reg diff post if pollutantshort=="NO2" & type=="Industrial" [w=popno2i], vce( robust)
getbaseline, no i
estadd scalar Baseline = `s(baseline)'
distinct idsensore if e(sample)
estadd scalar NStations r(ndistinct)
eststo typeno2t: reg diff post if pollutantshort=="NO2" & type=="Traffic" [w=popno2t], vce( robust)
getbaseline, no t
estadd scalar Baseline = `s(baseline)'
distinct idsensore if e(sample)
estadd scalar NStations r(ndistinct)
loc frag "&\multicolumn{3}{c}{PM 2.5}  &\multicolumn{3}{c}{NO2} \\\cmidrule(lr){2-4} \cmidrule(lr){5-7}"
esttab using "Docs/tables/ols_type.tex", replace label nogap compress b(2) se(2) varwidth(15) nodep stats(Baseline NStations N, fmt(a3 0) lab("Average baseline concentration" "Number of monitors" "Observations"))  ///
	mtit("Background" "Industrial" "Traffic" "Background" "Industrial" "Traffic") sub(\_ _) nonum  booktabs nonote ///
	mgroups("$ \Delta_{Observed,Counterfactual} $", pattern(1 0 0) prefix( \multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}`frag'))  ///
	postfoot("\bottomrule\end{tabular}" "\begin{tablenotes}[flushleft]\footnotesize" ///
		"\item \textit{Notes: } Regression weighted by population within 20 kilometers of a monitoring station. Territory within less than 20 kilometers from two or more monitors is assigned to the closest one." ///
		"The dependent variable is the difference between the observed values and the counterfactual." /// 
		"\textit{Lockdown} is a dummy variable equal to 0 from January 1, 2020 to February 22, and equal to 1 after February 22, 2020." ///
		"\textit{Average baseline concentration} is the population-weighted average of counterfactual values during the lockdown, less the constant in case the latter is statistically significant at 10\%." ///
		"Robust standard errors are in brackets. * p$<$0.1, ** p$<$0.05, *** p$<$0.01." ///
		"\end{tablenotes}" "}")

* --------------- *
*	Unweighted - Baseline
* --------------- *
use "Data/Modified/TidyResults", clear
eststo clear
eststo olspm25: reg diff post if pollutantshort=="PM2.5", vce(clus idsensore)
getbaseline, pm
estadd scalar Baseline = `s(baseline)'
eststo olsno2: reg diff post if pollutantshort=="NO2", vce(clus idsensore)
getbaseline, no
estadd scalar Baseline = `s(baseline)'
esttab ols* using "Docs/tables/ols_unweighted.tex", replace label b(2) se(2) stats(Baseline N, fmt(a3 0) lab("Average baseline concentration" "Observations")) star(* 0.1 ** 0.05 *** 0.01) nogap compress varwidth(30) nodep mtit("PM 2.5" "NO2") sub(\_ _) mgroups("$ \Delta_{Observed,Counterfactual} $", pattern(1 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) booktabs nonote ///
	postfoot("\bottomrule\end{tabular}" "\begin{tablenotes}[flushleft]\footnotesize" ///
		"\item \textit{Notes: } Unweighted regression." ///
		"The dependent variable is the difference between the observed values and the counterfactual." /// 
		"\textit{Lockdown} is a dummy variable equal to 0 from January 1, 2020 to February 22, and equal to 1 after February 22, 2020." ///
		"\textit{Average baseline concentration} is the average of counterfactual values during the lockdown, less the constant in case the latter is statistically significant at 10\%." ///
		"Standard errors, in brackets, are clustered by monitor. * p$<$0.1, ** p$<$0.05, *** p$<$0.01." ///
		"\end{tablenotes}" "}")

* --------------- *
*	Weighted - By station type
* --------------- *
use "Data/Modified/TidyResults", clear
// Create weights
wgt_pollutant_type "PM2.5" "Background" poppm25b
wgt_pollutant_type "PM2.5" "Industrial" poppm25i
wgt_pollutant_type "PM2.5" "Traffic" poppm25t
wgt_pollutant_type "NO2" "Background" popno2b
wgt_pollutant_type "NO2" "Industrial" popno2i
wgt_pollutant_type "NO2" "Traffic" popno2t

eststo clear
eststo typepm25b: reg diff post if pollutantshort=="PM2.5" & type=="Background", vce(rob)
getbaseline, pm b
estadd scalar Baseline = `s(baseline)'
distinct idsensore if e(sample)
estadd scalar NStations r(ndistinct)
eststo typepm25i: reg diff post if pollutantshort=="PM2.5" & type=="Industrial", vce(rob)
getbaseline, pm i
estadd scalar Baseline = `s(baseline)'
distinct idsensore if e(sample)
estadd scalar NStations r(ndistinct)
eststo typepm25t: reg diff post if pollutantshort=="PM2.5" & type=="Traffic", vce(rob)
getbaseline, pm t
estadd scalar Baseline = `s(baseline)'
distinct idsensore if e(sample)
estadd scalar NStations r(ndistinct)
eststo typeno2b: reg diff post if pollutantshort=="NO2" & type=="Background", vce(rob)
getbaseline, no b
estadd scalar Baseline = `s(baseline)'
distinct idsensore if e(sample)
estadd scalar NStations r(ndistinct)
eststo typeno2i: reg diff post if pollutantshort=="NO2" & type=="Industrial", vce(rob)
getbaseline, no i
estadd scalar Baseline = `s(baseline)'
distinct idsensore if e(sample)
estadd scalar NStations r(ndistinct)
eststo typeno2t: reg diff post if pollutantshort=="NO2" & type=="Traffic", vce(rob)
getbaseline, no t
estadd scalar Baseline = `s(baseline)'
distinct idsensore if e(sample)
estadd scalar NStations r(ndistinct)
loc frag "&\multicolumn{3}{c}{PM 2.5}  &\multicolumn{3}{c}{NO2} \\\cmidrule(lr){2-4} \cmidrule(lr){5-7}"
esttab using "Docs/tables/ols_type_unweighted.tex", replace label nogap compress b(2) se(2) varwidth(15) nodep stats(Baseline NStations N, fmt(a3 0) lab("Average baseline concentration" "Number of monitors" "Observations"))  ///
	mtit("Background" "Industrial" "Traffic" "Background" "Industrial" "Traffic") sub(\_ _) nonum  booktabs nonote ///
	mgroups("$ \Delta_{Observed,Counterfactual} $", pattern(1 0 0) prefix( \multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span}`frag'))  ///
	postfoot("\bottomrule\end{tabular}" "\begin{tablenotes}[flushleft]\footnotesize" ///
		"\item \textit{Notes: } Unweighted regression." ///
		"The dependent variable is the difference between the observed values and the counterfactual." /// 
		"\textit{Lockdown} is a dummy variable equal to 0 from January 1, 2020 to February 22, and equal to 1 after February 22, 2020." ///
		"\textit{Average baseline concentration} is the average of counterfactual values during the lockdown, less the constant in case the latter is statistically significant at 10\%." ///
		"Robust standard errors are in brackets. * p$<$0.1, ** p$<$0.05, *** p$<$0.01." ///
		"\end{tablenotes}" "}")
