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

function mean(arr)
{
	let r = 0;
	for (let i = 0; i < arr.length;i++)
		r += arr[i];
	return r/arr.length;
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

function variance(arr)
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

function standardDev(arr)
{
	return Math.sqrt(variance(arr));
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

	probabilityTo(z)
	{
		z = this.getStandardZ(z);
		return (erf(z/Math.sqrt(2))+1)/2;
	}

	probabilityFrom(z)
	{
		return 1. - this.probabilityTo(z);
	}

	probability(from, to)
	{
		return this.probabilityTo(to) - this.probabilityTo(from);
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

// var terms = require("./probability_terms.json");