#!/usr/bin/env wolframscript
(* fit_4clique_parallel_binding.wls
   Parallel binding (iid binding) + ER (uniform p) + uniform g
   Fit g so that E[#K4] = Q
   Run with: wolframscript -script fit_4clique_parallel_binding.wls --n 2077 --m 63023 --Q 26302487 --R 100000
   Or:       wolframscript -script fit_4clique_parallel_binding.wls --n 2077 --p 0.0292 --Q 26302487 --R 100000
*)

(* -------------------------
   CLI parsing helpers
   ------------------------- *)
ClearAll[getArg, hasArg, usageAndExit];

args = Rest[$ScriptCommandLine]; (* drop script name *)

hasArg[key_] := MemberQ[args, key];

getArg[key_, default_: Missing["NoDefault"]] := Module[{pos},
  pos = FirstPosition[args, key, Missing["NotFound"]];
  If[pos === Missing["NotFound"],
    If[default === Missing["NoDefault"], Missing["NotFound"], default],
    If[pos[[1]] == Length[args], Missing["MissingValue"], args[[pos[[1]] + 1]]]
  ]
];

usageAndExit[msg_] := (
  Print[msg];
  Print["Usage:"];
  Print["  wolframscript -script fit_4clique_parallel_binding.wls --n <int> (--m <int> | --p <real>) --Q <real> --R <int> [--wp <int>] [--g0 <real>] [--eps <int>]"];
  Print["Examples:"];
  Print["  wolframscript -script fit_4clique_parallel_binding.wls --n 2077 --m 63023 --Q 26302487 --R 100000"];
  Print["  wolframscript -script fit_4clique_parallel_binding.wls --n 2077 --p 0.0292 --Q 26302487 --R 100000 --wp 120"];
  Exit[1];
);

(* -------------------------
   Read required args
   ------------------------- *)
nStr = getArg["--n"];
QStr = getArg["--Q"];
RStr = getArg["--R"];

If[nStr === Missing["NotFound"] || QStr === Missing["NotFound"] || RStr === Missing["NotFound"],
  usageAndExit["Error: missing required argument(s). Required: --n, --Q, --R."]
];

n = ToExpression[nStr];
QIn = ToExpression[QStr];
R = ToExpression[RStr];

If[!IntegerQ[n] || n <= 0, usageAndExit["Error: --n must be a positive integer."]];
If[!IntegerQ[R] || R <= 0, usageAndExit["Error: --R must be a positive integer."]];

(* p is either provided directly or derived from m *)
pStr = getArg["--p", Missing["NotFound"]];
mStr = getArg["--m", Missing["NotFound"]];

If[pStr === Missing["NotFound"] && mStr === Missing["NotFound"],
  usageAndExit["Error: you must provide either --p or --m (to derive p)."]
];

If[pStr =!= Missing["NotFound"] && mStr =!= Missing["NotFound"],
  Print["Warning: both --p and --m were provided; using --p and ignoring --m."]
];

If[pStr =!= Missing["NotFound"],
  pIn = ToExpression[pStr],
  m = ToExpression[mStr];
  If[!IntegerQ[m] || m < 0, usageAndExit["Error: --m must be a nonnegative integer."]];
  pIn = N[m/(n (n - 1)/2)];
];

(* Optional args *)
wp = ToExpression[getArg["--wp", "80"]];
g0 = ToExpression[getArg["--g0", "0.3"]];
epsExp = ToExpression[getArg["--eps", "30"]];

If[!IntegerQ[wp] || wp < 20, usageAndExit["Error: --wp must be an integer >= 20."]];
If[!NumericQ[g0] || g0 <= 0 || g0 >= 1, usageAndExit["Error: --g0 must be in (0,1)."]];
If[!IntegerQ[epsExp] || epsExp < 6, usageAndExit["Error: --eps must be an integer >= 6 (meaning epsilon = 10^-eps)."]];

(* -------------------------
   Precision controls
   ------------------------- *)
eps = SetPrecision[10^-epsExp, wp];
chopTol = SetPrecision[10^(-wp/2), wp];

p = SetPrecision[pIn, wp];
Q = SetPrecision[QIn, wp];
one = SetPrecision[1, wp];
zero = SetPrecision[0, wp];

(* -------------------------
   Algorithm 3 scalars under uniform p,g
   ------------------------- *)
r[g_?NumericQ] := Module[{gg = SetPrecision[g, wp]},
  Min[(one - (one - p)^(one/R))/gg^2, one]
];

prem[g_?NumericQ] := Module[{gg = SetPrecision[g, wp]},
  Max[one - (one - p)/(one - gg^2)^R, zero]
];

(* -------------------------
   4 nodes, 6 edges, 4 triangles
   ------------------------- *)
V4 = Range[4];
allEdges = Subsets[V4, {2}];

triangleEdgeSets =
  Sort /@ (Subsets[#, {2}] & /@ Subsets[V4, {3}]);

isTriangle[Et_] := (Length[Et] == 3 && MemberQ[triangleEdgeSets, Sort[Et]]);
isK4[Et_] := (Length[Et] == 6 && Sort[Et] === Sort[allEdges]);

allEdgeSubsets = Subsets[allEdges];

(* -------------------------
   Single-round probabilities p_round(Et)
   ------------------------- *)
pRound[g_?NumericQ][Et_] := Module[{rg = r[g], gg = SetPrecision[g, wp]},
  Which[
    Et === {},
      (one - gg)^4 + 4 gg (one - gg)^3 +
      Binomial[4, 2] gg^2 (one - gg)^2 (one - rg) +
      Binomial[4, 3] gg^3 (one - gg) (one - rg) +
      gg^4 (one - rg),

    Length[Et] == 1 && MemberQ[allEdges, First@Et],
      rg * gg^2 * (one - gg)^2,

    isTriangle[Et],
      rg * gg^3 * (one - gg),

    isK4[Et],
      rg * gg^4,

    True,
      zero
  ]
];

(* -------------------------
   \tilde{A}(F) and Möbius inversion for Pr(final = K4)
   ------------------------- *)
tildeA[g_?NumericQ][F_] := Module[{sum, premg = prem[g]},
  sum = Total[pRound[g] /@ Subsets[F]];
  (sum^R) * (one - premg)^(Length[allEdges] - Length[F])
];

pK4[g_?NumericQ] := Chop[
  Total@Table[
    (-one)^(Length[allEdges] - Length[F]) * tildeA[g][F],
    {F, allEdgeSubsets}
  ],
  chopTol
];

expectedK4[g_?NumericQ] := SetPrecision[Binomial[n, 4], wp] * pK4[g];

(* -------------------------
   Solve for g
   ------------------------- *)
gHatFR = Quiet @ Check[
  g /. FindRoot[
    expectedK4[g] - Q,
    {g, SetPrecision[g0, wp], eps, one - eps},
    WorkingPrecision -> wp,
    AccuracyGoal -> Floor[wp/2],
    PrecisionGoal -> Floor[wp/2],
    MaxIterations -> 300
  ],
  $Failed
];

gHat = If[gHatFR === $Failed,
  g /. Last @ NMinimize[
    {(expectedK4[g] - Q)^2, eps < g < one - eps},
    g,
    WorkingPrecision -> wp,
    AccuracyGoal -> Floor[wp/2],
    PrecisionGoal -> Floor[wp/2],
    MaxIterations -> 800
  ],
  gHatFR
];

(* -------------------------
   Report
   ------------------------- *)
rHat    = N[r[gHat], wp];
premHat = N[prem[gHat], wp];
EK4     = N[expectedK4[gHat], wp];

Print["Inputs:"];
Print["  n = ", n];
If[pStr === Missing["NotFound"],
  Print["  m = ", m, "  (p derived as m / Binomial[n,2])"],
  Print["  p = ", NumberForm[p, {Infinity, 30}]]
];
Print["  R = ", R];
Print["  Q = ", NumberForm[Q, {Infinity, 30}]];
Print["  wp = ", wp, ", g0 = ", g0, ", eps = 10^-", epsExp];

Print["Results:"];
Print["  gHat = ", NumberForm[gHat, {Infinity, 60}]];
Print["  r(gHat) = ", NumberForm[rHat, {Infinity, 60}]];
Print["  p_rem(gHat) = ", NumberForm[premHat, {Infinity, 60}]];
Print["  expectedK4(gHat) = ", NumberForm[EK4, {Infinity, 30}]];
