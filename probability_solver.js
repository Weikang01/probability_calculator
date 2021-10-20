function print(...args)
{
	console.log(...args);
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
	return (arr[Math.floor((a.length-1) * .25 * num)] + arr[Math.ceil((a.length-1) * .25 * num)]) * .5;
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

class DiscreteDistribution
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

class BernoulliDistribution extends DiscreteDistribution
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

class BinomialDistribution extends DiscreteDistribution
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

class PoissonDistribution extends DiscreteDistribution
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

class HypergeometricDistribution extends DiscreteDistribution
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

var terms = require("./probability_terms.json");