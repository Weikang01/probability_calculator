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
	let ordered = quick_Sort(arr);
	let len = arr.length;
	if (len % 2 == 0)
		return (ordered[(len - 1) / 2] + ordered[(len + 1) / 2]) * .5;
	else
		return ordered[len / 2];
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

function standarDev(arr)
{
	return Math.sqrt(variance(arr));
}

function gcd(a, b){
	if (!b) {
		return a;
	}
  
	return gcd(b, a % b);
}

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
