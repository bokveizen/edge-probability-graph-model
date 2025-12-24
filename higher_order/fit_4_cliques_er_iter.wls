(* ::Package:: *)

(* Local binding (Algorithm 2)+ER+uniform g Fit g so that E[#K4]=Q *)

(*-------------------------Inputs-------------------------*)
n = 1692;
m = 47309;
QIn = 9608852;
pIn = m/(n (n - 1)/2);
R = 100000;

(*-------------------------Precision controls-------------------------*)
wp = 80;
eps = SetPrecision[10^-30, wp];
chopTol = SetPrecision[10^(-wp/2), wp];

p = SetPrecision[pIn, wp];
Q = SetPrecision[QIn, wp];
one = SetPrecision[1, wp];
zero = SetPrecision[0, wp];

(*-------------------------4 nodes,6 edges,4 triangles-------------------------*)
V4 = Range[4];
allEdges = Subsets[V4, {2}]; (* 6 unordered pairs *)
triangleEdgeSets =
  Sort /@ (Subsets[#, {2}] & /@ Subsets[V4, {3}]); (* 4 triangles *)

(*-------------------------Encode "covered edges so far" as a 6-bit mask-------------------------*)
edgeIndex = AssociationThread[allEdges -> Range[0, Length[allEdges] - 1]];

edgeMaskOfNodes[S_] :=
  Module[{pairs},
    If[Length[S] < 2,
      0,
      pairs = Subsets[S, {2}];
      Total[BitShiftLeft[1, edgeIndex /@ pairs]]
    ]
  ];

nodeSubsets = Subsets[V4]; (* 16 node subsets *)
nodeSubsetSizes = Length /@ nodeSubsets;
edgeMaskBySubset = edgeMaskOfNodes /@ nodeSubsets;

(*-------------------------Base vector at "no rounds left"
Remaining uncovered edges become singleton groups:
need p^(#uncovered)=p^(6-BitCount[maskCovered])-------------------------*)

bitCount[x_Integer] := DigitCount[x, 2, 1];

F0[] := Table[p^(Length[allEdges] - bitCount[mask]), {mask, 0, 63}];

(*-------------------------Weighted transition matrix W(g):
For each current covered mask i and sampled node subset S:
em=edges(S) newMask=em\ covered next=covered ∪ em
weight=1 if newMask empty else p
Then W[i,next]+=Pr(S sampled)*weight-------------------------*)

WMat[g_?NumericQ] :=
  Module[{gg = SetPrecision[g, wp], probs, W, i, sIdx, em, next, newMask, weight, pr},
    probs = (gg^#*(one - gg)^(4 - #)) & /@ nodeSubsetSizes;
    W = ConstantArray[zero, {64, 64}];
    For[i = 0, i <= 63, i++,
      For[sIdx = 1, sIdx <= Length[nodeSubsets], sIdx++,
        em = edgeMaskBySubset[[sIdx]];
        pr = probs[[sIdx]];
        newMask = BitAnd[em, BitXor[i, 63]]; (* em & ~i within 6 bits *)
        next = BitOr[i, em];
        weight = If[newMask == 0, one, p];
        W[[i + 1, next + 1]] += pr*weight;
      ];
    ];
    W
  ];

(*-------------------------Clique probability on a fixed 4-node set:
probK4(g)=(W(g)^R*F0)[mask=0]-------------------------*)

probK4[g_?NumericQ] :=
  Module[{W = WMat[g], v},
    v = MatrixPower[W, R].F0[];
    Chop[v[[1]], chopTol]
  ];

expectedK4[g_?NumericQ] := SetPrecision[Binomial[n, 4], wp]*probK4[g];

(*-------------------------Solve for g in (0,1)-------------------------*)
g0 = SetPrecision[0.3, wp];

gHatFR =
  Quiet @ Check[
    g /. FindRoot[
      expectedK4[g] - Q,
      {g, g0, eps, one - eps},
      WorkingPrecision -> wp,
      AccuracyGoal -> Floor[wp/2],
      PrecisionGoal -> Floor[wp/2],
      MaxIterations -> 200
    ],
    $Failed
  ];

gHat =
  If[gHatFR === $Failed,
    g /. Last @ NMinimize[
      {(expectedK4[g] - Q)^2, eps < g < one - eps},
      g,
      WorkingPrecision -> wp,
      AccuracyGoal -> Floor[wp/2],
      PrecisionGoal -> Floor[wp/2],
      MaxIterations -> 400
    ],
    gHatFR
  ];

(*-------------------------Report-------------------------*)
pK4Hat = N[probK4[gHat], wp];
EK4Hat = N[expectedK4[gHat], wp];

Print["Local binding fit results:"];
Print["  n = ", n, ", p = ", NumberForm[p, {Infinity, 20}], ", R = ", R];
Print["  Q = ", QIn];
Print["  gHat = ", NumberForm[gHat, {Infinity, 50}]];
Print["  Pr(K4 on 4 nodes) = ", NumberForm[pK4Hat, {Infinity, 40}]];
Print["  expectedK4(gHat) = ", NumberForm[EK4Hat, {Infinity, 30}]];
