const jstat = require('jstat');
var { jStat } = require('jstat');

function round(float, pos = 4)
{
	return Math.round(Math.pow(10, pos) * float)/ Math.pow(10, pos);
}

const COMPARISON = {
	SMALLER:-2,
	SMALLER_EQUAL:-1,
	EQUAL:0,
	GREATER_EQUAL:1,
	GREATER:2,
}

function print(...args)
{
	console.log(...args);
}

function stringToArr(...strs) {
	// sample input: "5.9 5.3 1.6 7.4 8.6 3.2 2.1", "4.0 7.3 8.4 5.9 6.7 4.5 6.3"
	let ret = [];
	for (let str of strs) {
		ret.push(...str.split(" ").map(f => parseFloat(f)))
	}
	return ret;
}

function erf(x){
    // erf(x) = 2/sqrt(pi) * integrate(from=0, to=x, e^-(t^2) ) dt
    // with using Taylor expansion, 
    //        = 2/sqrt(pi) * sigma(n=0 to +inf, ((-1)^n * x^(2n+1))/(n! * (2n+1)))
    // calculationg n=0 to 50 bellow (note that inside sigma equals x when n = 0, and 50 may be enough)
    var m = 1.00;
    var s = 1.00;
    var sum = x * 1.0;
    for(var i = 1; i < 50; i++){
        m *= i;
        s *= -1;
        sum += (s * Math.pow(x, 2.0 * i + 1.0)) / (m * (2.0 * i + 1.0));
    }  
    return 2 * sum / Math.sqrt(3.14159265358979);
}

function quick_Sort(origArray) {
	if (origArray.length <= 1) { 
		return origArray;
	} else {

		var left = [];
		var right = [];
		var newArray = [];
		var pivot = origArray.pop();
		var length = origArray.length;

		for (var i = 0; i < length; i++) {
			if (origArray[i] <= pivot) {
				left.push(origArray[i]);
			} else {
				right.push(origArray[i]);
			}
		}

		return newArray.concat(quick_Sort(left), pivot, quick_Sort(right));
	}
}

function __quartile(arr, num)
{
	return (arr[Math.floor((arr.length-1) * .25 * num)] + arr[Math.ceil((arr.length-1) * .25 * num)]) * .5;
}

/**
 * param: 
 * k: P(μ-kσ,μ+kσ)
 */
function chebyshev(k) {
	return {
		"Min. % within k standard deviations of mean": (1. - (1./(k*k)))*100.+"%",
		"Max. % beyond k standard deviations of mean":(1./(k*k))*100.+"%"
	};
}

function sum_arr(arr, operation_function = (x) => {return x})
{
	var r = 0;
	for (let i = 0; i < arr.length; i++)
		r += operation_function(arr[i]);
	return r;
}

function factorial(n)
{
	let r = 1;
	for (;n>1;n--)
		r *= n;
	return r;
}

function duplicateItems(arr = new Array)
{
	var len = arr.length;
	var items = new Array;
	var contains;
	for (let i = 0; i < arr.length; i++) {
		contains = false;
		for (let j = 0; j < items.length; j++) {
			if (items[j][0] == arr[i])
			{
				items[j][1] ++;
				contains = true;
				break;
			}
		}
		if (!contains)
		{
			items.push([arr[i], 1, 0.0]);
		}
	}
	for (let i = 0; i < items.length; i++) {
		items[i][2] = items[i][1]/len;
	}
	return items;
}

function mean_uni(arr)
{
	let r = 0;
	for (let i = 0; i < arr.length;i++)
		r += arr[i];
	return r/arr.length;
}

function mean(x_arr, px_arr)
{
	let r = 0;
	for (let i = 0; i < x_arr.length;i++)
		r += x_arr[i] * px_arr[i];
	return r;
}

function median(arr)
{
	return __quartile(arr, 2);
}

function range(arr)
{
	let max =arr[0], min = arr[0];
	for (let i = 1; i < arr.length; i++)
	{
		if (arr[i] > max)
			max = arr[i];
		else if (arr[i] < min)
			min = arr[i];
	}
	return max - min;
}

function variation(arr)
{
	let sqx = 0;
	let x = 0;
	for (let i = 0;i<arr.length;i++)
	{
		sqx += arr[i] * arr[i];
		x += arr[i];
	}
	return sqx-((x*x)/arr.length);
}

function variance_uni_samp(arr)
{
	return variation(arr)/(arr.length-1);
}

function variance_uni_pop(arr)
{
	return variation(arr)/(arr.length);
}

function variance(x_arr, px_arr) {
	let sqx = 0;
	let x = 0;
	for (let i = 0;i<x_arr.length;i++)
	{
		sqx += x_arr[i] * x_arr[i] * px_arr[i];
		x += x_arr[i] * px_arr[i];
	}
	return sqx-x*x;
}

function standardDev_uni_samp(arr)
{
	return Math.sqrt(variance_uni_samp(arr));
}

function standardDev_uni_pop(arr)
{
	return Math.sqrt(variance_uni_pop(arr));
}

function standardDev(x_arr, px_arr) {
	return Math.sqrt(variance(x_arr, px_arr));
}

function sampleMean_fromMean(mu) {
	return mu;
}

function mean_fromSampleMean(muxbar) {
	return muxbar;
}

function sampleVariance_fromVariance(sigma2, sizeOfSample)
{
	return sigma2 * sizeOfSample;
}

function Variance_fromSampleVariance(sigma2xbar, sizeOfSample)
{
	return sigma2xbar / sizeOfSample;
}

// AKA sample standard dev == standard error
function standardDev_fromSampleStandardDev(sigmabar, sizeOfSample) {
	return sigmabar * Math.sqrt(sizeOfSample);
}

// AKA sample standard dev == standard error
// s_{\bar{x}} = \frac{\sigma}{\sqrt{n}}
function sampleStandardDev_fromStandardDev(sigma, sizeOfSample) {
	return sigma / Math.sqrt(sizeOfSample);
}

function standardDev_fromSampleProportion(p, sizeOfSample)
{
	return Math.sqrt((p * (1-p))/sizeOfSample);
}

function gcd(a, b){
	if (!b) {
		return a;
	}
  
	return gcd(b, a % b);
}

/**
 * formula is
 * (n r) =
 * n!/(r!(n-r)!)
 */
function combination(from, pick)
{
	let deno = 1;
	let nume = 1;
	for (let i = 0; i < pick; i++)
	{
		nume *= from - i;
		deno *= i + 1;
		if (i > 10 || !(i % 5))
		{
			let cd = gcd(nume, deno);
			deno /= cd;
			nume /= cd;
		}
	}
	return nume / deno;
}

function permutation(from, pick)
{
	let r = 1;
	for (let i = 0; i < pick; i++)
		r *= from - i;
	return r;
}

function lowerQuartile(arr)
{
	return __quartile(arr, 1);
}

function upperQuartile(arr)
{
	return __quartile(arr, 3);
}

function interQuartileRange(arr)
{
	return upperQuartile(arr) - lowerQuartile(arr);
}

function IQRfromQLandQU(QL, QU) {
	return QU-QL;
}

function getBoxplotParams(arr)
{
	let ql = lowerQuartile(arr);
	let m = median(arr);
	let qu = upperQuartile(arr);
	let lqr = qu-ql;

	let low_out = [];
	let upp_out = [];

	arr.forEach(ele => {
		if (ele < ql-3*lqr)
			low_out.push(ele);
		else if (ele > qu+3.*lqr)
			upp_out.push(ele)
	})

	return {
		"lower outliers": low_out,
		"lower outer fence":ql-3*lqr,
		"lower inner fence":ql-1.5*lqr,
		"lower quartile":ql,
		"median":m,
		"upper quartile":qu,
		"upper inner fence":qu+1.5*lqr,
		"upper outer fence":qu+3.*lqr,
		"upper outliers": upp_out,
		"interquartile range (IQR)": lqr
	};
}

function getArrayInfo(arr)
{
	let ordered = quick_Sort(arr);
	let v = variance(ordered);
	return {
		"variance": v,
		"mean":mean(ordered),
		"standard deviation": Math.sqrt(v),
		"box plot params": getBoxplotParams(ordered)
	};
}

function zScore(value, arr)
{
	return (value - mean(arr))/standardDev(arr);
}

function zScore(value, mean, standardDev) {
	return (value - mean)/standardDev;
}

function getUniformDitributionInfo(arr_of_values)
{
	let mu = 0;
	let sqExp = 0;
	arr_of_values.forEach(ele =>{
		mu += ele;
		sqExp += ele * ele;
	});
	mu /= arr_of_values.length;
	sqExp /= arr_of_values.length;
	let sigma = sqExp - mu*mu;
	return {
		"mean(μ)":mu,
		"variance(σ^2)":sigma,
		"standard deviation(σ)":Math.sqrt(sigma)
	}
}

class Distribution
{
	mean()
	{
		return 0;
	}

	variance()
	{
		return 0;
	}

	info()
	{
		let mu = this.mean();
		let sd = Math.sqrt(this.variance());
		return {
			"mean(μ)":mu,
			"variance(σ^2)":this.variance(),
			"standard deviation(σ)":sd,
			"[μ-σ,μ+σ]":[mu-sd,mu+sd],
			"[μ-2σ,μ+2σ]":[mu-2*sd,mu+2*sd],
		};
	}
}

class BernoulliDistribution extends Distribution
{
	constructor(possibility)
	{
		super();
		this.possibility = possibility;
	}

	probability(x)
	{
		return this.possibility;
	}

	mean()
	{
		return this.possibility;
	}

	variance()
	{
		return this.possibility * (1.-this.possibility);
	}
}

class BinomialDistribution extends Distribution
{
	constructor(totalNr, possibility)
	{
		super();
		this.totalNr = totalNr;
		this.possibility = possibility;
	}

	probability(x)
	{
		return combination(this.totalNr, x)*Math.pow(this.possibility, x)*Math.pow(1.-this.possibility, this.totalNr-x);
	}

	mean()
	{
		return this.totalNr * this.possibility;
	}

	variance()
	{
		return this.totalNr * this.possibility * (1.-this.possibility);
	}
}

class PoissonDistribution extends Distribution
{
	/*
	lambda: Mean number of events during a given unit of time, area, volume, etc.
	*/
	constructor(lambda_AKA_mean)
	{
		super();
		this.lambda = lambda_AKA_mean;
	}

	probability(x)
	{
		return (Math.pow(this.lambda, x)*Math.pow(Math.E, -this.lambda))/factorial(x);
	}

	mean()
	{
		return this.lambda;
	}

	variance()
	{
		return this.lambda;
	}
}

class HypergeometricDistribution extends Distribution
{
	/** 
	 * Drawing n elements WITHOUT replacement
	** nrTotalElements: N
	** nrSuccessInElements: r
	** nrElementsDrawn: n
	*/
	constructor(nrTotalElements, nrSuccessInElements, nrElementsDrawn)
	{
		super();
		this.N = nrTotalElements;
		this.r = nrSuccessInElements;
		this.n = nrElementsDrawn;
	}

	probability(nrSuccessDrawn)
	{
		return (combination(this.r, nrSuccessDrawn)*combination(this.N-this.r,this.n-nrSuccessDrawn))/combination(this.N, this.n);
	}

	mean()
	{
		return (this.n*this.r)/(this.N);
	}

	/**
	** variance = [r(N-r)n(N-n)]/[N*N*(N-1)]
	*/
	variance()
	{
		return (this.r*(this.N-this.r)*this.n*(this.N-this.n))/(this.N*this.N*(this.N-1));
	}

	
	availableRange()
	{
		return [Math.max(0, this.n-(this.N-this.r)), Math.min(this.r, this.n)];
	}
}

class UniformDistribution extends Distribution
{
	constructor(from, to)
	{
		super();
		this.c = from<to?from:to;
		this.d = from<to?to:from;
	}

	mean()
	{
		return (this.c+this.d)/2;
	}

	variance()
	{
		return ((this.d-this.c)*(this.d-this.c))/12;
	}

	probability(from, to)
	{
		return (to-from)/(this.d-this.c);
	}

	probabilityFrom(from)
	{
		return this.probability(from, this.d);
	}

	probabilityTo(to)
	{
		return this.probability(this.c, to);
	}
}

function confidenceLevel_fromAlpha(alpha)
{
	return 1-alpha;
}

class NormalDistribution extends Distribution
{
	constructor(mean = 0., standardDeviation = 1.)
	{
		super();
		this.mu = mean;
		this.sigma = standardDeviation;
	}

	getStandardZ(z)
	{
		return (z-this.mu)/this.sigma;
	}

	getZFromStandardZ(prob)
	{
		return prob * this.sigma + this.mu;
	}

	static CDF(z)
	{
		var r = (erf(z/Math.sqrt(2))+1)/2;
		return Math.round(r*10000) * .0001;
	}

	probabilityTo(z)
	{
		z = this.getStandardZ(z);
		return NormalDistribution.CDF(z);
	}

	probabilityFrom(z)
	{
		return 1. - this.probabilityTo(z);
	}

	probability(from, to)
	{
		return this.probabilityTo(to) - this.probabilityTo(from);
	}

	static inverseCDF(p)
	{
		var a1 = -39.6968302866538, a2 = 220.946098424521, a3 = -275.928510446969;
		var a4 = 138.357751867269, a5 = -30.6647980661472, a6 = 2.50662827745924;
		var b1 = -54.4760987982241, b2 = 161.585836858041, b3 = -155.698979859887;
		var b4 = 66.8013118877197, b5 = -13.2806815528857, c1 = -7.78489400243029E-03;
		var c2 = -0.322396458041136, c3 = -2.40075827716184, c4 = -2.54973253934373;
		var c5 = 4.37466414146497, c6 = 2.93816398269878, d1 = 7.78469570904146E-03;
		var d2 = 0.32246712907004, d3 = 2.445134137143, d4 = 3.75440866190742;
		var p_low = 0.02425, p_high = 1 - p_low;
		var q, r;
		var retVal;
	
		if ((p < 0) || (p > 1))
		{
			alert("NormSInv: Argument out of range.");
			retVal = 0;
		}
		else if (p < p_low)
		{
			q = Math.sqrt(-2 * Math.log(p));
			retVal = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
		}
		else if (p <= p_high)
		{
			q = p - 0.5;
			r = q * q;
			retVal = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
		}
		else
		{
			q = Math.sqrt(-2 * Math.log(1 - p));
			retVal = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
		}
	
		return retVal;
	}

	static isSuitableForNormalApproximation(n, p)
	{
		return n > 9 * Math.max(p/(1-p), (1-p)/p);
	}

	static approximateBinomialDist(n,p)
	{
		if (!NormalDistribution.isSuitableForNormalApproximation(n, p))
			console.error("%c[ERROR]\nValue of n and p not suitable for normal approximation\nn <= max( p /(1 - p ), (1 - p )/ p )\n"+ n+ "<= max("+p+"/(1 -"+p+"), (1 -"+p+")/"+p+")", "color: red");
		else
			return new NormalDistribution(n*p, Math.sqrt(n*p*(1-p)));
	}

	static binomialApprox(k, comparison = COMPARISON.EQUAL)
	{
		// half-unit correction for continuity
		switch (comparison) {
			case COMPARISON.EQUAL:
				return probability(k-.5, k+.5);
			case COMPARISON.GREATER:
				return probabilityFrom(k+.5);
			case COMPARISON.SMALLER:
				return probabilityTo(k-.5);
			case COMPARISON.GREATER_EQUAL:
				return probabilityFrom(k-.5);
			case COMPARISON.SMALLER_EQUAL:
				return probabilityTo(k+.5);
			default:
				throw ("[ERROR_INVALID_COMPARISON]: ", __LINE__);
		}
	}

	static ConfidenceLevel_fromZ_alphaOver2(Z_alphaOver2)
	{
		var r = 1. - 2. * (1. - NormalDistribution.CDF(Z_alphaOver2));
		return Math.round(r*1000)/1000;
	}

	static Z_alphaOver2_fromConfidenceLevel(confidenceLevel)
	{
		var r = NormalDistribution.inverseCDF(1. - (1. - confidenceLevel) * .5);
		return Math.round(r*1000)/1000;
	}
}

class TDistribution extends Distribution
{
	constructor(degreeOfFreedom)
	{
		super();
		this.df = degreeOfFreedom;
	}

	static PDF(x, df)
	{
		return jStat.studentt.pdf(x, df);
	}

	static PDF_oneTailed(x, df)
	{
		return TDistribution.PDF(x, df) * .5;
	}

	static CDF(x, df)
	{
		return jStat.studentt.cdf(x, df);
	}

	static inverseCDF(p, df)
	{
		var r = -jStat.studentt.inv(p, df);
		return Math.abs(r) > 100? Math.round(r*100)/100:r>10?Math.round(r*1000)/1000:Math.round(r*1000)/1000;
	}

	static t_alphaOver2_fromConfidenceLevel_nrOfSample(confidenceLevel, nrOfSample)
	{
		console.log("t_alphaOver2 = t_" + (1. - confidenceLevel)*.5 + "\t dof = " + (nrOfSample - 1));
		return TDistribution.inverseCDF((1. - confidenceLevel)*.5, nrOfSample-1);
	}
}

class ChiSquareDistribution extends Distribution
{
	constructor(degreeOfFreedom)
	{
		super();
		this.df = degreeOfFreedom;
	}

	static inverseCDF(p, df)
	{
		var r = jStat.chisquare.inv(1. - p, df);
		return r > 100? Math.round(r * 100)/100: r>10?Math.round(r * 10000)/10000 : Math.round(r * 100000)/100000;
	}
}

class ExponentialDistribution extends Distribution
{
	constructor(theta)
	{
		super();
		this.theta = theta;
	}

	mean()
	{
		return this.theta;
	}

	variance()
	{
		return this.theta * this.theta;
	}

	standardDeviation()
	{
		return this.theta;
	}

	probabilityFrom(to)
	{
		return Math.pow(Math.E, -to/this.theta);
	}

	probabilityTo(from)
	{
		return 1. - this.probabilityFrom(from);
	}

	probability(from, to)
	{
		return this.probabilityTo(to) - this.probabilityTo(from);
	}
}

function recursiveCounter(x_arr, px_arr, epoch)
{
	if (epoch <= 1) {
		let list_arr = [];
		for (let i = 0; i < x_arr.length; i++)
			list_arr.push([x_arr[i]]);

		return [[...x_arr], [...px_arr], [...list_arr]];
	}

	var prev_arrs = recursiveCounter(x_arr, px_arr, epoch-1);
	var prev_x_arr = prev_arrs[0];
	var prev_px_arr = prev_arrs[1];
	var prev_list_arr = prev_arrs[2];

	var new_x_arr = new Array;
	var new_px_arr = new Array;
	var new_list_arr = new Array;
	
	for (let i = 0; i < x_arr.length; i ++)
	{
		for (let j = 0; j < prev_x_arr.length; j++) 
		{
			new_x_arr.push(x_arr[i] + prev_x_arr[j]);
			new_px_arr.push(px_arr[i] * prev_px_arr[j]);

			var new_list = [...prev_list_arr[j]];
			new_list.push(x_arr[i]);
			new_list.sort();
			new_list_arr.push(new_list);
		}
	}
	return [[...new_x_arr], [...new_px_arr], [...new_list_arr]];
}

function getSamplingDistributionOfSampleMeanFromProbabilityDistribution(x_arr, px_arr, sizeOfSample)
{
	// x_arr : (values of x)               [ 1,  2,  3,  4,  5]
	// px_arr: (values of probability of x)[.2, .3, .2, .2, .1]
	// sizeOfSample: 2
	var values = recursiveCounter(x_arr, px_arr, sizeOfSample);
	var xs = values[0];
	var pxs = values[1];
	var lists = values[2];

	var final_x = new Array;
	var final_px = new Array;
	var counter = new Array;
	var medians = new Array;
	var p_medians = new Array;

	for (let i = 0; i < xs.length; i++)
	{
		var cur_median = median(lists[i]);
		if (medians.includes(cur_median))
		{
			var index = medians.indexOf(cur_median);
			p_medians[index] += pxs[i];
		}
		else
		{
			medians.push(cur_median);
			p_medians.push(pxs[i]);
		}

		if (final_x.includes(xs[i]))
		{
			var index = final_x.indexOf(xs[i]);
			final_px[index] += pxs[i];
			counter[index] ++;
		}
		else
		{
			final_x.push(xs[i]);
			final_px.push(pxs[i]);
			counter.push(1);
		}
	}

	for (let i = 0; i < final_x.length; i++) {
		final_x[i] /= sizeOfSample;
	}

	return [final_x, final_px, counter, medians, p_medians];
}

function getSamplingError(standardDev, sampleSize, confidenceLevel) {
	var σ_sqrtN = standardDev / Math.sqrt(sampleSize);
	var Z_alphaOver2 = NormalDistribution.Z_alphaOver2_fromConfidenceLevel(confidenceLevel);
	console.log("SE = Z_alphaOver2 * σ_sqrtN = ");
	return Z_alphaOver2 * σ_sqrtN;
}

function getSampleSize(standardDev, samplingError, confidenceLevel)
{
	var Z_alphaOver2 = NormalDistribution.Z_alphaOver2_fromConfidenceLevel(confidenceLevel);
	console.log("n = (Z_alphaOver2 * Z_alphaOver2) * (standardDev * standardDev) / (samplingError * samplingError) = ");
	console.log(`n = (${Z_alphaOver2} * ${Z_alphaOver2}) * (${standardDev} * ${standardDev}) / (${samplingError} * ${samplingError}) = `);
	return Z_alphaOver2 * Z_alphaOver2 * standardDev * standardDev / (samplingError * samplingError);
}

function isSampleSizeLargeEnough(sampleSize, p, is_pHat)
{
	if (is_pHat)
		return sampleSize > 15 * Math.max(1/p, 1/(1-p));
	else
		return sampleSize > Math.max(p/(1-p),(1-p)/p);
}

function standardDevFromRangeOfObservation(range)
{
	return range * .25;
}

function getCI(xBar, standardDev, sampleSize, confidenceLevel)
{
	if (sampleSize >= 30)
	{
		var σ_sqrtN = standardDev / Math.sqrt(sampleSize);
		var Z_alphaOver2 = NormalDistribution.Z_alphaOver2_fromConfidenceLevel(confidenceLevel);
		return {
			"σ_sqrtN": σ_sqrtN,
			"Z_alphaOver2":Z_alphaOver2,
			"xBar±Z_alphaOver2*σ_sqrtN": xBar + " ± " + Z_alphaOver2 + " * " + σ_sqrtN,
			"result":xBar + " ± " + (Z_alphaOver2 * σ_sqrtN),
			"interval": [xBar-Z_alphaOver2*σ_sqrtN,xBar+Z_alphaOver2*σ_sqrtN]
		};
	}
	else
	{
		var σ_sqrtN = standardDev / Math.sqrt(sampleSize);
		var t_alphaOver2 = TDistribution.t_alphaOver2_fromConfidenceLevel_nrOfSample(confidenceLevel, sampleSize);
		return {
			"σ_sqrtN": σ_sqrtN,
			"t_alphaOver2":t_alphaOver2,
			"xBar±t_alphaOver2*σ_sqrtN": xBar + " ± " + t_alphaOver2 + " * " + σ_sqrtN + " = " + xBar + " ± " + (t_alphaOver2 * σ_sqrtN),
			"interval": [xBar-t_alphaOver2*σ_sqrtN,xBar+t_alphaOver2*σ_sqrtN]
		};
	}

}

function getSamplingError4PP(pHat, sampleSize, confidenceLevel) {
	let qHat = 1. - pHat;
	var sqrtpqOverN = Math.sqrt((pHat * qHat) / sampleSize);
	var Z_alphaOver2 = NormalDistribution.Z_alphaOver2_fromConfidenceLevel(confidenceLevel);
	console.log("SE = Z_alphaOver2 * sqrtpqOverN = ");
	return Z_alphaOver2 * sqrtpqOverN;
}

function getSampleSize4PP(pHat, samplingError, confidenceLevel)
{
	let qHat = 1. - pHat;
	var Z_alphaOver2 = NormalDistribution.Z_alphaOver2_fromConfidenceLevel(confidenceLevel);
	console.log("n = (Z_alphaOver2 * Z_alphaOver2) * (pHat * qHat) / (samplingError * samplingError) = ");
	console.log(`n = (${Z_alphaOver2} * ${Z_alphaOver2}) * (${pHat} * ${qHat}) / (${samplingError} * ${samplingError}) = `);
	return Z_alphaOver2 * Z_alphaOver2 * pHat * qHat / (samplingError * samplingError);
}

function getSampleSize4PP_withoutP(samplingError, confidenceLevel)
{
	return getSampleSize4PP(.5, samplingError, confidenceLevel);
}

function getAdjustedP(nrOfSuccess, sampleSize)
{
	return (nrOfSuccess + 2)/(sampleSize + 4);
}

function getMean4PP(pHat)
{
	return pHat;
}

function getSamplingMean(pHat)
{
	return pHat;
}

function getSamplingVariance(pHat, sampleSize) {
	return (pHat * (1-pHat)) / sampleSize;
}

function getSamplingStandardDev(pHat, sampleSize)
{
	return Math.sqrt(getSamplingVariance(pHat, sampleSize));
}

function getCI4PP(pHat, sampleSize, confidenceLevel)
{
	let qHat = 1. - pHat;
	if (pHat * sampleSize >= 15 && qHat * sampleSize >= 15)
	{
		var sqrtpqOverN = Math.sqrt((pHat * qHat) / sampleSize);
		var sqrtpqOvernplus4 = Math.sqrt((pHat * qHat) / (sampleSize + 4));
		var Z_alphaOver2 = NormalDistribution.Z_alphaOver2_fromConfidenceLevel(confidenceLevel);

		return {
			"sqrtpqOverN": sqrtpqOverN,
			"Z_alphaOver2":Z_alphaOver2,
			"pHat±Z_alphaOver2*sqrtpqOverN": pHat + " ± " + Z_alphaOver2 + " * " + sqrtpqOverN + " = " + pHat + " ± " + (Z_alphaOver2 * sqrtpqOverN),
			"interval": [pHat-Z_alphaOver2*sqrtpqOverN,pHat+Z_alphaOver2*sqrtpqOverN],
			"p~±Z_alphaOver2*sqrtpqOvernplus4": pHat + " ± " + Z_alphaOver2 + " * " + sqrtpqOvernplus4 + " = " + pHat + " ± " + (Z_alphaOver2 * sqrtpqOvernplus4),
			"adjusted interval": [pHat-Z_alphaOver2*sqrtpqOvernplus4,pHat+Z_alphaOver2*sqrtpqOvernplus4]
		};
	}
}

function variance_chiSquare(sampleSize, sampleVariance, confidenceLevel)
{
	var alphaOver2 = (1. - confidenceLevel) * .5;
	var x_alphaOver2 = ChiSquareDistribution.inverseCDF(alphaOver2, sampleSize-1);
	var x_1minus_alphaOver2 = ChiSquareDistribution.inverseCDF(1. - alphaOver2, sampleSize-1);
	var lowerBound = ((sampleSize-1) * sampleVariance)/x_alphaOver2;
	var upperBound = ((sampleSize-1) * sampleVariance)/x_1minus_alphaOver2;

	return {
		"alphaOver2": alphaOver2,
		"x_alphaOver2":x_alphaOver2,
		"x_1minus_alphaOver2": x_1minus_alphaOver2,
		"((n-1)s^2)/x^2_{a/2} <= sigma^2 <= ((n-1)s^2)/x^2_{1-a/2}": `((${alphaOver2}-1)${sampleVariance})/${x_alphaOver2} <= sigma^2 <= ((${alphaOver2}-1)${sampleVariance})/${x_1minus_alphaOver2}`,
		"CI": [lowerBound, upperBound],
		"standard dev CI": [Math.sqrt(lowerBound), Math.sqrt(upperBound)]
	}
}

function confidenceLevel2alpha(CL) {
	var r = .5 * (1 - CL);
	console.log("alpha: " + r);
	return .5 * (1 - CL);
}

function get_mean_from_sampleProportion(pHat) {
	return pHat;
}

function get_variance_from_sampleProportion(p, sampleSize) {
	return (p * (1-p))/sampleSize;
}

function get_size_threshold_p_unknown(pHat) {
	return 15 * Math.max(1/pHat, 1/(1-pHat));
}

function get_size_threshold_sampleProportion(p)
{
	return 9*Math.max(p/(1-p),(1-p)/p);
}

function get_z_fromAlpha(alpha)
{
	return NormalDistribution.CDF(alpha) - .5;
}

function get_z_fromConfidenceLevel(CL) {
	return NormalDistribution.CDF(confidenceLevel2alpha(CL));
}

function get_degreeOfFreedom_fromSampleSize(sampleSize) {
	return sampleSize - 1;
}

function get_t_fromAlpha(alpha, degreeOfFreedom) {
	return TDistribution.inverseCDF(alpha, degreeOfFreedom);
}

function get_t_fromConfidenceLevel(CL, degreeOfFreedom) {
	return TDistribution.inverseCDF(confidenceLevel2alpha(CL), degreeOfFreedom)
}

function get_chi_square_fromAlpha(alpha, degreeOfFreedom) {
	return ChiSquareDistribution.inverseCDF(alpha, degreeOfFreedom);
}

function get_chi_square_fromConfidenceLevel(CL, degreeOfFreedom) {
	var alpha = confidenceLevel2alpha(CL);
	return {
		"alpha1": alpha,
		"x1": ChiSquareDistribution.inverseCDF(alpha, degreeOfFreedom),
		"alpha2": 1-alpha,
		"x2": ChiSquareDistribution.inverseCDF(1-alpha, degreeOfFreedom),
	};
}

function get_probabilityDist_of_sample(popMean, popStandardDev, sampleSize)
{
	return new NormalDistribution(popMean, popStandardDev / Math.sqrt(sampleSize));
}

function covariation(arr1, arr2)
{
	if (arr1.length != arr2.length)
	{
		console.log(`COVARIATION: length of ${arr1} and ${arr2} are difference`);
		return;
	}
	var xy = 0, x = 0, y = 0;
	for (let i = 0; i < arr1.length; i++) {
		xy += arr1[i]*arr2[i];
		x += arr1[i];
		y += arr2[i];
	}
	return xy - (x*y)/arr1.length;
}

function __covaiance(arr1, arr2, n)
{
	if (typeof(arr1) == typeof([]) && typeof(arr2) == typeof([]))
	{
		return covariation(arr1, arr2)/n;
	}
	else
	{
		console.log(`COVARIANCE: ${arr1} and/or ${arr2} are/is not array`);
		return;
	}
}

function covaiance_pop(arr1, arr2)
{
	return __covaiance(arr1, arr2, arr1.length);
}

function covaiance_samp(arr1, arr2)
{
	return __covaiance(arr1, arr2, arr1.length-1);
}

function pearson_r(arr1, arr2)
{
	var cv = covariation(arr1, arr2);
	var xv = variation(arr1);
	var yv = variation(arr2);
	console.log({
		"covariation": cv,
		"variation_x": xv,
		"variation_y": yv
	});
	var r = covariation(arr1, arr2)/ Math.sqrt(variation(arr1)*variation(arr2));
	console.log(`[${arr1}]\nand\n[${arr2}]\nhas ${r>0?"direct": r<0?"inverse":"no"} relationship`);
	return r;
}

class LinearRegression
{
	sum_x()
	{
		var sum = 0;
		for (let i = 0; i < this.x_arr.length; i++) {
			sum += this.x_arr[i];
		}
		return sum;
	}

	sum_y()
	{
		var sum = 0;
		for (let i = 0; i < this.y_arr.length; i++) {
			sum += this.y_arr[i];
		}
		return sum;
	}

	estimate(x)
	{
		return this.intercept + this.slope * x;
	}

	total_predicted()
	{
		var ar = new Array(this.x_arr.length);
		var sum = 0;
		for (let i = 0; i < this.x_arr.length; i++) {
			ar[i] = round(this.estimate(this.x_arr[i]), 3);
			sum += ar[i];
		}
		return {
			"total predicted":ar,
			"sum":round(sum, 3)
		};
	}

	constructor(x_arr, y_arr)
	{
		if (x_arr.length != y_arr.length)
			throw("LINEAR REGRESSION: element array with different length!");
		this.x_arr = x_arr;
		this.y_arr = y_arr;
		this.slope = covariation(x_arr, y_arr)/variation(x_arr);
		this.intercept = mean_uni(y_arr) - this.slope * mean_uni(x_arr);
		this.standardErrorOfEstimate = 0;
		var r = 0;
		var temp;
		for (let i = 0; i < y_arr.length; i++) {
			temp = y_arr[i] - this.estimate(x_arr[i]);
			r += temp * temp;
		}
		this.standardErrorOfEstimate = round(Math.sqrt(r/(y_arr.length - 2)), 3);

		console.log({
			"slope = covariation_xy/variation_x = ": round(this.slope, 3),
			"intercept = mean_y - slope * mean_x = " : round(this.intercept, 3),
			"standard error of estimate (s_e) = sqrt(sum(y_i - y_est)^2/(n-2)) = ": this.standardErrorOfEstimate,
			"trend line" : `y = ${this.intercept} + ${this.slope}x`
		});
	}

	Pearson_r()
	{
		return pearson_r(this.x_arr, this.y_arr);
	}

	residual(ary_index)
	{
		var r = this.y_arr[ary_index] - this.estimate(this.x_arr[ary_index]);
		console.log(`y_i - y_est = ${r}`);
		console.log(`this is a ${r > 0? "over-estimate": r < 0? "under-estimate": "exact estimation"}`);
		return r;
	}

	total_residual()
	{
		var ar = new Array(this.x_arr.length);
		var sum = 0;
		for (let i = 0; i < this.x_arr.length; i++) {
			ar[i] = this.y_arr - this.estimate(this.x_arr[i]);
			sum += ar[i];
		}
		return {
			"total residual":ar,
			"sum":round(sum, 3)
		};
	}

	unexplained(ary_index)
	{
		var r = this.residual(ary_index);
		r = r*r;
		console.log(`(y_i - y_est)^2 = ${r}`);
		return r;
	}

	total_unexplained()
	{
		var ar = new Array(this.x_arr.length);
		var sum = 0;
		var temp;
		for (let i = 0; i < this.x_arr.length; i++) {
			temp = this.y_arr[i] - this.estimate(this.x_arr[i]);
			ar[i] = round(temp * temp, 4);
			sum += ar[i];
		}
		return {
			"total unexplained":ar,
			"sum":round(sum, 3)
		};
	}

	standardized_residual(ary_index)
	{
		var r = this.residual(ary_index)/this.standardErrorOfEstimate;
		console.log(`standardized residual: (y_i - y_est)/S_e = ${r}`);
		return r;
	}

	static getExplained(total, unexplained)
	{
		var explained = total - unexplained;
		return {
			"Total": total,
			"Unexplained": unexplained,
			"Explained = Total - Unexplained": explained,
			"percentage of variation explained": round(explained/total*100, 4) + " %"
		};
	}

	getExplained(total)
	{
		return total - this.unexplained();
	}
}

function separateVarianceEstimate(standardDev1, sampleSize1, standardDev2, sampleSize2)
{
	var sigma_xBar1_minus_xBar2 = Math.sqrt(standardDev1*standardDev1/sampleSize1+standardDev2*standardDev2/sampleSize2);
	console.log(
		{
			"standard error: sigma_{x1 mean - x2 mean} = s_p*sqrt(1/n1+1/n2)":round(sigma_xBar1_minus_xBar2),
			"degree of freedom: min(n1-1, n2-1)":Math.min(sampleSize1, sampleSize2)-1
		}
	);
	return {
		"standardError":sigma_xBar1_minus_xBar2, "dof":Math.min(sampleSize1, sampleSize2)-1
	}
}

function pooledVarianceEstimate(standardDev1, sampleSize1, standardDev2, sampleSize2)
{
	var s_p = Math.sqrt((standardDev1 * standardDev1 * (sampleSize1 -1) + standardDev2 * standardDev2 * (sampleSize2 -1))/(sampleSize1 + sampleSize2 - 2));
	var sigma_xBar1_minus_xBar2 = s_p * Math.sqrt(1/sampleSize1+1/sampleSize2);
	console.log(
		{
			"pooled standard deviation: s_p = sqrt of (s1^2(n1-1)+s2^2(n2-1))/(n1+n2-2)": round(s_p),
			"standard error: sigma_{x1 mean - x2 mean} = s_p*sqrt(1/n1+1/n2)":round(sigma_xBar1_minus_xBar2),
			"degree of freedom: n1+n2-2":sampleSize1+sampleSize2-2
		}
	);
	return {
		"s_p":s_p, "standardError":sigma_xBar1_minus_xBar2, "dof":sampleSize1+sampleSize2-2
	};
}

function SVE_twoSampleStatistics_mean(sampleSize1, mean1, standardDev1, sampleSize2, mean2, standardDev2, hypothesizedDiff_between2PopMeans=0, levelOfConfidence=-1, alpha=-1)
{
	var res = separateVarianceEstimate(standardDev1, sampleSize1, standardDev2, sampleSize2);
	var standardError = res.standardError;
	var dof = res.dof;

	var t_star = ((mean1 - mean2) - hypothesizedDiff_between2PopMeans)/standardError;
	var p_t = TDistribution.PDF(t_star, dof);

	var alpha = alpha>=0?alpha:levelOfConfidence>=0?1-levelOfConfidence:-1;
	console.log(
		{
			"Test Statistic: t* = ((x1 mean - x2 mean) - (mu1 - mu2))/(sigma_{x1 mean - x2 mean})": round(t_star),
			"two-tailed p value":round(p_t, 4),
			"one-tailed p value":round(p_t * .5, 4),
			"confidence level": alpha>=0?1-alpha:"NAN",
			"alpha": alpha>=0?alpha:"NAN",
			"if (2_tailed)p-value>alpha, accept H0": alpha<0? "NAN":p_t>alpha?"accepted":"rejected",
			"if (1_tailed)p-value>alpha, accept H0": alpha<0? "NAN":p_t*.5>alpha?"accepted":"rejected",
		}

	);
	return t_star;
}

function PVE_twoSampleStatistics_mean(sampleSize1, mean1, standardDev1, sampleSize2, mean2, standardDev2, hypothesizedDiff_between2PopMeans=0, alpha=-1, levelOfConfidence=-1)
{
	var res = pooledVarianceEstimate(standardDev1, sampleSize1, standardDev2, sampleSize2);
	var standardError = res.standardError;
	var dof = res.dof;

	var t_star = ((mean1 - mean2) - hypothesizedDiff_between2PopMeans)/standardError;
	var p_t = 1-TDistribution.CDF(t_star, dof);

	var alpha = alpha>=0?alpha:levelOfConfidence>=0?1-levelOfConfidence:-1;
	console.log(
		{
			"Test Statistic: t* = ((x1 mean - x2 mean) - (mu1 - mu2))/(sigma_{x1 mean - x2 mean})": round(t_star),
			"two-tailed p value":round(2*p_t, 4),
			"one-tailed p value":round(p_t, 4),
			"confidence level": alpha>=0?1-alpha:"NAN",
			"alpha": alpha>=0?alpha:"NAN",
			"if (2_tailed)p-value>alpha, accept H0": alpha<0? "NAN":2*p_t>alpha?"accepted":"rejected",
			"if (1_tailed)p-value>alpha, accept H0": alpha<0? "NAN":p_t>alpha?"accepted":"rejected",
		}

	);
	return t_star;
}

function variance_from_sums(sum_x, sum_x_sqr, sampleSize) {
	return (sum_x_sqr - sum_x * sum_x/sampleSize)/(sampleSize-1);
}

function pairsSampleData(x_arr, y_arr, rounded = false)
{
	if (x_arr.length != y_arr.length)
		return "pairsSampleData::not correct array length";

	var sum_x = 0, sum_x_sqr = 0, sum_y = 0, sum_y_sqr = 0;
	var sum_d = 0, sum_d_sqr = 0, temp;
	for (let i = 0; i < x_arr.length; i++) {
		sum_x += x_arr[i];
		sum_x_sqr += x_arr[i] * x_arr[i];
		sum_y += y_arr[i];
		sum_y_sqr += y_arr[i] * y_arr[i];

		temp = y_arr[i] - x_arr[i];
		sum_d += temp;
		sum_d_sqr += temp * temp;
	}
	var variance_x = variance_from_sums(sum_x, sum_x_sqr, x_arr.length);
	var variance_y = variance_from_sums(sum_y, sum_y_sqr, y_arr.length);
	var data =
	{
		"sampleSize": x_arr.length,
		"x": {
			"sum": sum_x,
			"squaredSum": sum_x_sqr,
			"mean": rounded? round(sum_x/x_arr.length):sum_x/x_arr.length,
			"variance": rounded? round(variance_x):variance_x,
			"standardDev": rounded? round(Math.sqrt(variance_x)):Math.sqrt(variance_x)
		},
		"y": {
			"sum": sum_y,
			"squaredSum": sum_y_sqr,
			"mean": rounded? round(sum_y/y_arr.length):sum_y/y_arr.length,
			"variance": rounded? round(variance_y):variance_y,
			"standardDev": rounded? round(Math.sqrt(variance_y)):Math.sqrt(variance_y)
		},
		"diff": {
			"sum":sum_d,
			"squaredSum": sum_d_sqr
		}
	};
	console.log(data);
	return data;

}

function pairsSampleStatistics_mean(sum_diff_x, sum_diff_xsquared, sampleSize, hypothesizedDiff_between2PopMeans=0, alpha=-1, levelOfConfidence=-1)
{
	var s_d = Math.sqrt((sum_diff_xsquared - (sum_diff_x*sum_diff_x)/sampleSize)/(sampleSize-1));
	var sigma_d_mean = s_d/Math.sqrt(sampleSize);
	var d_bar = sum_diff_x / sampleSize;
	var t_star = (d_bar-hypothesizedDiff_between2PopMeans)/sigma_d_mean;
	var dof = sampleSize - 1;
	var p_t = 1-TDistribution.CDF(t_star, dof);
	alpha = alpha>=0?alpha:levelOfConfidence>=0?1-levelOfConfidence:-1;
	console.log(
		{
			"s_d: sqrt((sum(d^2) - sum(d)/n)/(n-1))":round(s_d),
			"sigma_d_mean = s_d/sqrt(n)":round(sigma_d_mean),
			"mean difference: d_mean = sum(d)/n":round(d_bar),
			"t* = (d_mean - d hypothesis)/sigma_d_mean": round(t_star),
			"dof": dof,
			"one-tailed p value: TDist.PDF(p_t)":p_t,
			"two-tailed p value: 2TDist.PDF(p_t)":p_t * 2,
			"confidence level": alpha>=0?1-alpha:"NAN",
			"alpha": alpha>=0?alpha:"NAN",
			"if (2_tailed)p-value>alpha, accept H0": alpha<0? "NAN":p_t>alpha?"accepted":"rejected",
			"if (1_tailed)p-value>alpha, accept H0": alpha<0? "NAN":p_t * 2>alpha?"accepted":"rejected",
		}
	)
}

function mannWhitneyU(a_arr, b_arr, alpha = -1, levelOfConfidence=-1) {
	var ordered = new Array(a_arr.length + b_arr.length);
	var idx = new Array(a_arr.length + b_arr.length);
	var group = new Array(a_arr.length + b_arr.length);
	for (let i = 0; i < a_arr.length; i++) {
		ordered[i] = a_arr[i];
		idx[i] = i + 1;
		group[i] = 0;
	}
	for (let i = 0; i < b_arr.length; i++) {
		ordered[a_arr.length + i] = b_arr[i];
		idx[a_arr.length + i] = a_arr.length + i + 1;
		group[a_arr.length + i] = 1;
	}


	var temp_num, temp_grp;
	for (let i = 0; i < ordered.length; i++)
	{
		for (let j = 0; j < ordered.length-i-1; j++)
		{
			if (ordered[j] > ordered[j+1])
			{
				temp_num = ordered[j];
				ordered[j] = ordered[j+1];
				ordered[j+1] = temp_num;
	
				temp_grp = group[j];
				group[j] = group[j+1];
				group[j+1] = temp_grp;
			}
		}
	}

	var last = "NAN";
	var duplicated = 0;
	var temp_idx;
	
	for (let i = 0; i < ordered.length; i++)
	{
		if (last == ordered[i])
			duplicated++;
		else if (duplicated != 0)
		{
			temp_idx = 0;
			for (let j = 0; j <= duplicated; j++)
				temp_idx += idx[i-j-1];
			temp_idx /= (duplicated+1);
			for (let j = 0; j <= duplicated; j++)
				idx[i-j-1] = temp_idx;
			duplicated = 0;
		}
		last = ordered[i];
	}

	var R1 = 0;
	var R2 = 0;

	for (let i = 0; i < idx.length; i++)
	{
		if (group[i] == 0)
			R1 += idx[i];
		else
			R2 += idx[i];
	}

	var U1 = a_arr.length * b_arr.length + (a_arr.length * (a_arr.length + 1))*.5 - R1;
	var U2 = a_arr.length * b_arr.length + (b_arr.length * (b_arr.length + 1))*.5 - R2;
	var U = Math.min(U1, U2);

	var mu_u = a_arr.length * b_arr.length * .5;
	var sigma_u = Math.sqrt(a_arr.length * b_arr.length * (a_arr.length + b_arr.length + 1) / 12);
	var z_score = (U - mu_u)/sigma_u;
	var p_t = NormalDistribution.CDF(z_score);
	alpha = alpha>=0?alpha:levelOfConfidence>=0?1-levelOfConfidence:-1;
	console.log(
		{
			"R1":R1,
			"R2":R2,
			"U1 = n1n2 + (n1(n1+1))/2 - R1": U1,
			"U2 = n1n2 + (n2(n2+1))/2 - R2": U2,
			"Mann-Whitney U":U,
			"mu_u = n1n2/2":round(mu_u,4),
			"sigma_u = sqrt(n1n2(n1+n2+1)/12)":round(sigma_u,4),
			"z score = (U - mu_u)/sigma_u":round(z_score, 4),
			"one-tailed p value":p_t,
			"two-tailed p value":2*p_t,
			"confidence level": alpha>=0?1-alpha:"NAN",
			"alpha": alpha>=0?alpha:"NAN",
			"if (2_tailed)p-value>alpha, accept H0": alpha<0? "NAN":2*p_t>alpha?"accepted":"rejected",
			"if (1_tailed)p-value>alpha, accept H0": alpha<0? "NAN":p_t>alpha?"accepted":"rejected",
		}
	)
}

function proportionStatistics(sampleSize1, p1, sampleSize2, p2, hypothesizedDiff_between2PopMeans=0, alpha = -1, levelOfConfidence=-1)
{
	var x1 = Math.round(sampleSize1 * p1);
	var x2 = Math.round(sampleSize2 * p2);
	var p_p = (x1+x2)/(sampleSize1+sampleSize2); 
	var q_p = 1 - p_p;
	var sigma_p1Minusp2 = Math.sqrt(p_p * q_p * (sampleSize1 + sampleSize2) / (sampleSize1*sampleSize2));
	var z_star = (p1 - p2 - hypothesizedDiff_between2PopMeans)/sigma_p1Minusp2;
	var p_t = 1-NormalDistribution.CDF(z_star);
	alpha = alpha>=0?alpha:levelOfConfidence>=0?1-levelOfConfidence:-1;
	console.log(
		{
			"x1":x1,
			"x2":x2,
			"p_p = (x1+x2)/(n1+n2)": round(p_p, 4),
			"q_p = 1-p_p": round(q_p, 4),
			"sigma_p1Minusp2 = sqrt(p_p*q_p(n1+n2)/n1n2)": round(sigma_p1Minusp2, 4),
			"z * = ((p1-p2)- p hypothesis)/sigma_p1Minusp2": round(z_star),
			"one-tailed p value":p_t,
			"two-tailed p value":2*p_t,
			"confidence level": alpha>=0?1-alpha:"NAN",
			"alpha": alpha>=0?alpha:"NAN",
			"if (2_tailed)p-value>alpha, accept H0": alpha<0? "NAN":2*p_t>alpha?"accepted":"rejected",
			"if (1_tailed)p-value>alpha, accept H0": alpha<0? "NAN":p_t>alpha?"accepted":"rejected",
		}
	)
}

function group_statistics(value_arr, group_arr, rounded = false)
{
	var has_col = false;
	var cols = new Array(); // element [col_name, element_nr, col_sum, col_sumSquared]
	for (let i = 0; i < group_arr.length; i++) {
		has_col = false;
		for (let j = 0; j < cols.length; j++) {
			if (cols[j][0] == group_arr[i])
			{
				cols[j][1] ++;
				cols[j][2] += value_arr[i];
				cols[j][3] += (value_arr[i] * value_arr[i]);
				has_col = true;
				break;
			}
		}
		if (!has_col)
		{
			cols.push([group_arr[i], 1, value_arr[i], value_arr[i]*value_arr[i]]);
		}
	}
	var sampleSize, sum, squaredSum, variance, data;
	var r = new Array;
	for (let i = 0; i < cols.length; i++) {
		sampleSize = cols[i][1];
		sum = cols[i][2];
		squaredSum = cols[i][3];
		variance = (squaredSum-sum*sum/sampleSize)/(sampleSize-1);
		data = 
		{
			"groupID": cols[i][0],
			"sampleSize": sampleSize,
			"sum": sum,
			"squaredSum": squaredSum,
			"mean": rounded? round(sum/sampleSize):sum/sampleSize,
			"variance": rounded? round(variance):variance,
			"standardDev": rounded? round(Math.sqrt(variance)):Math.sqrt(variance)
		};
		console.log(data);
		r.push(data);
	}
	
	return r;
}

function F_test(value_arr, group_arr, alpha = -1, levelOfConfidence=-1)
{
	var sum_x = sum_arr(value_arr);
	var sum_xsquared = sum_arr(value_arr, (x)=>{return x*x;});
	var sum_xx_over_n = sum_x*sum_x/value_arr.length;
	var ss_total = sum_xsquared - sum_xx_over_n;

	var has_col = false;
	var cols = new Array(); // element [col_name, element_nr, col_sum]
	for (let i = 0; i < group_arr.length; i++) {
		has_col = false;
		for (let j = 0; j < cols.length; j++) {
			if (cols[j][0] == group_arr[i])
			{
				cols[j][1] ++;
				cols[j][2] += value_arr[i];
				has_col = true;
				break;
			}
		}
		if (!has_col)
		{
			cols.push([group_arr[i], 1, value_arr[i]]);
		}
	}
	for (let i = 0; i < cols.length; i++) {
		console.log(`group_${cols[i][0]}:\telement_nr: ${cols[i][1]}\tcol_sum: ${cols[i][2]}`);
	}

	var col_sqr_sum = sum_arr(cols, (col)=>{ return col[2]*col[2]/col[1];});
	var ss_btn = col_sqr_sum - sum_xx_over_n;
	var ss_wtn = sum_xsquared - col_sqr_sum;

	var df_btn = cols.length - 1;
	var ms_btn = ss_btn/df_btn;

	var df_wtn = group_arr.length - cols.length;
	var ms_wtn = ss_wtn/df_wtn;

	var df_total = df_btn + df_wtn;
	var f_star = ms_btn/ms_wtn;

	var p_t = 1-jStat.centralF.cdf( f_star, df_btn, df_wtn );
	alpha = alpha>=0?alpha:levelOfConfidence>=0?1-levelOfConfidence:-1;

	console.log(
		{
			"sum x": sum_x,
			"sum(x)^2/n":sum_xx_over_n,
			"sum(x^2)":sum_xsquared,
			"sum(sum(Col_i)^2/n)":col_sqr_sum,
			"SS_total": ss_total,
			"SS_between": ss_btn,
			"SS_within": ss_wtn,
			"df_between": df_btn,
			"MS_between":ms_btn,
			"df_within": df_wtn,
			"MS_within":ms_wtn,
			"df_total": df_total,
			"F* = MS_between/MS_within":f_star,
			"p value":p_t,
			"confidence level": alpha>=0?1-alpha:"NAN",
			"alpha": alpha>=0?alpha:"NAN",
			"if p-value>alpha, sample variances are equal": alpha<0? "NAN":p_t>alpha?"variances are equal":"variances are not equal",
		}
	)

}

function contingencyTable_statistics(nrCol, values) {
	var nrRow = values.length / nrCol;
	if (nrRow - round(nrRow) != 0)
	{
		console.log("contingencyTable_statistics::number of rows is not integer!");
		return;
	}

	var out = "   \t";
	for (let i = 0; i < nrCol; i++) {
		out += `\tcol ${i+1}`;
	}
	out += `\trow total\n`;
	var row_margin = new Array(nrRow+1).fill(0);
	var col_margin = new Array(nrCol+1).fill(0);
	for (let i = 0; i < nrRow; i++) {
		out += `row ${i+1}\t`;
		for (let j = 0; j < nrCol; j++) {
			out += `\t${values[nrCol*i+j]}`;
			row_margin[i] += values[nrCol*i+j];
			row_margin[nrRow] += values[nrCol*i+j];
			col_margin[j] += values[nrCol*i+j];
			col_margin[nrCol] += values[nrCol*i+j];
		}
		out += `\t${row_margin[i]}\n`;
	}
	out += `col total`;
	for (let i = 0; i <= nrCol; i++) {
		out += `\t${col_margin[i]}`;
	}

	out += "\n\ncell\tobserved\texpected\tO-E\t\t(O-E)^2/E\n";
	var expected, OMinusE, OMinusEsquaredByE;
	var s_OMinusEsquaredByE=0;
	for (let i = 0; i < nrRow; i++) {
		for (let j = 0; j < nrCol; j++) {
			expected = row_margin[i]*col_margin[j]/col_margin[nrCol];
			OMinusE = values[nrCol*i+j] - expected;
			OMinusEsquaredByE = OMinusE*OMinusE/expected;
			s_OMinusEsquaredByE += OMinusEsquaredByE;
			out += `R${i+1}-C${j+1}\t${values[nrCol*i+j]}\t\t${round(expected)}\t\t${round(OMinusE, 3)}\t\t${round(OMinusEsquaredByE)}\n`;
		}
	}
	out += `sum\t${col_margin[nrCol]}\t\t${col_margin[nrCol]}\t\t0\t\t${round(s_OMinusEsquaredByE)}`;

	out += `\n\ntable info:\n`;
	var dof = (nrRow-1)*(nrCol-1);
	out += `χ square: ${round(s_OMinusEsquaredByE)}\n`
	out += `dof (rows-1)*(cols-1): ${dof}\n`;
	out += `critical value(α=0.05): ${round(jStat.chisquare.inv(0.95, dof))}\n`;
	var p_t = 1-jStat.chisquare.cdf(s_OMinusEsquaredByE, dof);
	out += `p-value: ${round(p_t)}\n`;
	out += `if p-value>α(α=0.05), accept H0: ${p_t>0.05?"accepted":"rejected"}`;
	console.log(out);
}

var v_arr = [38.00,42.00,50.00,57.00,80.00,70.00,32.00,20.00,48.00,58.00,66.00,80.00,62.00,73.00,39.00,73.00,58.00,64.00,80.00,70.00,60.00,55.00,72.00,73.00,81.00,50.00,68.00];
var g_arr = [1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,];

var tb =
[
106	,104, 48,
43	,65	, 30,
26	,59	, 35,
];
// var tb =
// [
// 10, 12, 20,
// 15, 10, 8,
// 8, 5, 12,
// ];

contingencyTable_statistics(3, tb);
// var res = group_statistics(v_arr, g_arr, true);
// F_test(v_arr, g_arr, 0.05);
