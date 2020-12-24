import delimited "Data/Original/Population/Mortality_table.csv", clear
keep if territory=="Lombardia"
rename tipo_dato15 statname
keep if statname=="PROBDEATH" | statname=="LIFEXP"
drop if gender=="total"
keep territory statname gender eta1 value

gsort territory eta1 gender statname

*	Age
gen age = ""
replace age = regexs(1) if regexm(eta1, "Y([0-9]+)-")
replace age = "0" if eta1=="Y_UN4"
destring age, replace
drop eta1

*	Prob of 1 in 1000 dying to P(death)
replace value = value/1000 if statname=="PROBDEATH"

*	Reshape
replace statname = "_p" if statname=="PROBDEATH"
replace statname = "_le" if statname=="LIFEXP"
// drop if statname=="n" // I use pop data at municipality level. Pop Lombardia not needed
greshape wide value*, i(territory age) j(statname gender) string

*	From statistics at 5-yr level to 10-yr level
gsort age
gen rest = (age/10 - floor(age/10))*2
gen age10 = floor(age/10)*10

*	From prob of dying in 5 years to prob of dying in 10 years
*	p10 = p5_1 
gen x1 = value_p_males if rest==0
bys territory age10 (age): egen alpha = mean(x1)
gen x2 = value_p_males if rest==1
bys territory age10 (age): egen beta = mean(x2)
gen x3 = alpha+(1-alpha)*beta
gen p_males = x3
drop x1 x2 x3 alpha beta

gen x1 = value_p_females if rest==0
bys territory age10 (age): egen alpha = mean(x1)
gen x2 = value_p_females if rest==1
bys territory age10 (age): egen beta = mean(x2)
gen x3 = alpha+(1-alpha)*beta
gen p_females = x3
drop x1 x2 x3 alpha beta

replace age = age10
drop age10

gcollapse (mean) value_le* p*, by(territory age)

*	Then to probability of dying within 1 year
replace p_females = 1-(1-p_females)^(1/10)
replace p_males = 1-(1-p_males)^(1/10)

save "temp", replace


* =========================================================================== *
*	YLS for the entire Lombardia
* =========================================================================== *
/* 
Life-years saved in 2 months_gender,age = 
	
	Coeff * DID/10 * BaselineRisk_gender,age * 1/6 * LifeExp_gender,age * N_gender,age

	(Coeff * DID/10) * BaselineRisk_gender,age

*/

use "Data/Modified/Controls.dta", clear
keep name male* female*
reshape long male female, i(name) j(age)
rename name comune
gcollapse (sum) male female, by(age)
tempfile poplomb
save `poplomb'


use "temp", clear
drop if age<30
rename value_* *

merge 1:1 age using `poplomb', keep(3) nogen

capture program drop mortality_yls
program define mortality_yls

	loc baseline_risk_var  `1'
	loc life_exp_var  `2'
	loc N_var  `3'
	loc delta_scalar  `4'
	loc RR_scalar  `5'
	loc mortality_newvarname `6'
	loc yls_newvarname `7'

	loc beta = log(`RR_scalar')/10
	
	gen `mortality_newvarname' = `N_var'*`baseline_risk_var' * ( 1 - 1/( exp(`beta'*`delta_scalar') )) / 6
	gen `yls_newvarname' = `mortality_newvarname'*`life_exp_var'

end

capture program drop linear_mortality_yls
program define linear_mortality_yls

	/* Compute mortality and YLS the easy way */

	loc baseline_risk_var  `1'
replace baseline_risk_var  `NO2	loc life_exp_var  `2'
	loc N_var  `3'
	loc delta_scalar  `4'
	loc RR_scalar  `5'
	loc mortality_newvarname `6'
	loc yls_newvarname `7'

	loc beta = log(`RR_scalar')/10
	
	gen `mortality_newvarname' = `N_var'*`baseline_risk_var' * (`RR_scalar'-1) * `delta_scalar'/10 / 6
	gen `yls_newvarname' = `mortality_newvarname'*`life_exp_var'

end

scalar didPM25 = $didPM25
scalar didNO2 = $didNO2
scalar RREEAPM25 = 1.062
scalar RRLepeulePM25 = 1.14
scalar RRKrewskiPM25 = 1.056
scalar RREEANO2 = 1.055

*	PM 2.5
mortality_yls p_females le_females female didPM25 RREEAPM25 mortEEAPM25_female lysEEAPM25_female
mortality_yls p_females le_females female didPM25 RRLepeulePM25 mortLepeulePM25_female lysLepeulePM25_female
mortality_yls p_females le_females female didPM25 RRKrewskiPM25 mortKrewskiPM25_female lysKrewskiPM25_female
mortality_yls p_males le_males male didPM25 RREEAPM25 mortEEAPM25_male lysEEAPM25_male
mortality_yls p_males le_males male didPM25 RRLepeulePM25 mortLepeulePM25_male lysLepeulePM25_male
mortality_yls p_males le_males male didPM25 RRKrewskiPM25 mortKrewskiPM25_male lysKrewskiPM25_male

*	NO2
mortality_yls p_females le_females female didNO2 RREEANO2 mortEEANO2_female lysEEANO2_female
mortality_yls p_males le_males male didNO2 RREEANO2 mortEEANO2_male lysEEANO2_male

gcollapse (sum) lys* mort*

loc stubs lysEEAPM25 lysLepeulePM25 lysKrewskiPM25 lysEEANO2 mortEEAPM25 mortLepeulePM25 mortKrewskiPM25 mortEEANO2 
foreach x of loc stubs {
	gen `x' = `x'_female +`x'_male
	drop `x'_female  `x'_male
}

rename * x_*
gen i = 1
reshape long x_, i(i) j(x) string
drop i

preserve
	use "Data/Modified/Controls.dta", clear
	replace POP_2011 =  "1242123" if POP_2011=="1.242.123"
	gen x = real(POP)
	destring POP_2011, replace dpcomma ignore(".")
	gcollapse (sum) POP_2011
	sum POP_2011
	scalar pop = r(mean)
restore

replace x_ = round(x_/pop*100000, 0.1)

gen desc = ""
replace desc = "Years of life saved" if regexm(x, "lys")
replace desc = "Avoided deaths" if regexm(x, "mort")
gen pollutant = "PM 2.5" if regexm(x, "PM25")
replace pollutant = "NO2" if regexm(x, "NO2")
gen rr_source =  "EEA/WHO" if regexm(x, "EEA")
replace rr_source = "Lepeule et al. (2012)" if regexm(x, "Lepeule")
replace rr_source = "Krewski et al. (2009)" if regexm(x, "Krewski")
gen rr = 1.056 if regexm(x, "Krewski") & pollutant=="PM 2.5"
replace rr = 1.14 if regexm(x, "Lepeule") & pollutant=="PM 2.5"
replace rr = 1.062 if regexm(x, "EEA") & pollutant=="PM 2.5"
replace rr = 1.055 if regexm(x, "EEA") & pollutant=="NO2"
bys desc: gen n = _n
sort desc pollutant rr_source
replace desc = "" if n!=4
drop n

gen estimate = x_
drop x_ x
replace estimate = -estimate

lab var desc ""
lab var estimate "Avoided deaths"
lab var pollutant "Pollutant"
lab var rr_source "Source of HR"
lab var rr "Hazard ratio"
texsave using "Docs/tables/lifeyears_lomb.tex", replace frag varlab width(\hsize)  nofix align(llcccccc) ///
	title("Avoided premature deaths and years of life saved per 100,000 in Lombardy due to improved air quality during lockdown.") marker("table:lifeyears_lomb") ///
	footnote("In Lombardy, from February 22 to May 3 2020, every 100,000 people 155 died after testing positive for COVID-19 and 1891 years of life have been directly lost to the virus. The  hazard ratio is the ratio of two concentration-response functions, or hazard rates, between a high and a low concentration differing by 10 $\mu g/m^3$. Avoided premature deaths are calculated using the population-weighted change in concentrations at background stations.")

tempfile LYS
save `LYS' 


* =========================================================================== *
*	Compare with COVID-19 YLL
* =========================================================================== *

* --------------- *
*	Life expectancy
* --------------- *
import delimited "Data/Original/Population/Mortality_table.csv", clear
keep if territory=="Lombardia"
rename tipo_dato15 statname
keep if statname=="LIFEXP"
drop if gender=="total"
keep territory statname gender eta1 value

gsort territory eta1 gender statname

*	Age
gen age = ""
replace age = regexs(1) if regexm(eta1, "Y([0-9]+)-")
replace age = "0" if eta1=="Y_UN4"
destring age, replace
lab var age "5-year age group starting at -age-"
drop eta1

*	Gender
gen female = (gender=="females")

keep age female value
tempfile lifeexp
save `lifeexp'

* --------------- *
*	Covid mortality
* --------------- *
use "Data/Modified/Mortality", clear

keep if datadecesso < mdy(5,4,2020)
fre datadecesso
fre deceased
keep if deceased==1
keep age female

preserve
	gcollapse (count) count = age
	sum count
	scalar avoided = r(mean)
restore

*	Round to 5-year groups
replace age = floor(age/5)*5

*	Count 
gen n = 1
gcollapse (count) n, by(age female)
*	Merge
merge 1:1 age female using `lifeexp', keep(1 3) nogen

* --------------- *
*	Compute years of life saved
* --------------- *
gen le = n*value
collapse (sum) le n
sum le
scalar covid = r(mean)

preserve
	use "Data/Modified/Controls.dta", clear
	replace POP_2011 =  "1242123" if POP_2011=="1.242.123"
	gen x = real(POP)
	destring POP_2011, replace dpcomma ignore(".")
	gcollapse (sum) POP_2011
	sum POP_2011
	scalar pop = r(mean)
restore

replace le = le/pop*100000
replace n = n/pop*100000
replace le = round(le)
replace n = round(n)

rename * x_*
gen i = 1
reshape long x_, i(i) j(desc) string
drop i
rename x_ estimate
replace desc = "Deaths" if desc=="n"
replace desc = "Years of life lost" if desc=="le"
gsort -desc

lab var desc ""
lab var estimate ""

texsave using "Docs/tables/covid.tex", replace varlab frag  width(0.5\hsize)  nofix marker("table:covid") ///
	title("COVID-19 deaths and years of life lost per 100,000 in Lombardy from February to May 2020.") ///
	footnote("Years of life lost are computed using gender- and age-specific life expectancy.")
