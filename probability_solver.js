var { jStat } = require('jstat');

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

function variance_uni(arr)
{
	let sqx = 0;
	let x = 0;
	for (let i = 0;i<arr.length;i++)
	{
		sqx += arr[i] * arr[i];
		x += arr[i];
	}
	return (sqx-(x*x/arr.length))/(arr.length-1);
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

function standardDev_uni(arr)
{
	return Math.sqrt(variance_uni(arr));
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

function standardDev_fromSampleStandardDev(sigma, sizeOfSample) {
	return sigma * Math.sqrt(sizeOfSample);
}

function sampleStandardDev_fromStandardDev(sigmaxbar, sizeOfSample) {
	return sigmaxbar / Math.sqrt(sizeOfSample);
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
	constructor(lambda)
	{
		super();
		this.lambda = lambda;
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
		return (this.d-this.c)/sqrt(12);
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
		return (erf(z/Math.sqrt(2))+1)/2;
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
		return n > Math.max(p/(1-p), (1-p)/p);
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
			"xBar±Z_alphaOver2*σ_sqrtN": xBar + " ± " + Z_alphaOver2 + " * " + σ_sqrtN + " = " + xBar + " ± " + (Z_alphaOver2 * σ_sqrtN),
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



// var terms = require("./probability_terms.json");

// console.log(mean([0,1,2], [1/3,1/3,1/3]));
// console.log(variance([0,1,2], [1/3,1/3,1/3]));

// var value = getSamplingDistributionOfSampleMeanFromProbabilityDistribution([0,1,2], [1/3,1/3,1/3],10);
// var xs = value[0];
// var pxs = value[1];

// console.log(xs);
// console.log(mean(xs, pxs));
// console.log(variance(xs, pxs));

// console.log(d.probabilityFrom(3.67));

// var d = new NormalDistribution();

// console.log(d.probability(0, .25));

// console.log(NormalDistribution.ConfidenceLevel_fromZ_alphaOver2(1.645));
// console.log(getCI(25.9, 2.7, 90, .99));

// console.log(count);
// console.log(mean(xs, pxs));
// console.log(variance(xs, pxs));

// console.log(mean([ 1,  2,  3,  4,  5], [.2, .3, .2, .2, .1]));

// console.log(TDistribution.inverseCDF(.005, 8));
// console.log(TDistribution.inverseCDF(.95, 16));
// console.log(TDistribution.inverseCDF(.95, 16));
// console.log(TDistribution.inverseCDF(.95, 16));

// console.log(getCI4PP(.76, 144, .9));
// console.log(getSampleSize4PP(.2, .06, .95));
// console.log(getSampleSize4PP(.5, .02, .95));
// console.log(variance_chiSquare(144,141787,.95));
// console.log(NormalDistribution.Z_alphaOver2_fromConfidenceLevel(.95));

console.log(variance_chiSquare(50,2.5*2.5,.9));
console.log(variance_chiSquare(15,.02*.02,.9));
console.log(variance_chiSquare(22,31.6*31.6,.9));
console.log(variance_chiSquare(5,1.5*1.5,.9));