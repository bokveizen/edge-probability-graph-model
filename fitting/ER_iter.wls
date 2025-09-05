#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(**)


(* ::Input:: *)
(*nDict=<|"facebook"->4039,"hamsterster"->2000,"polblogs"->1222,"web-spam"->4767,"bio-CE-PG"->1692,"bio-SC-HT"->2077|>;*)
(*mDict=<|"facebook"->88234,"hamsterster"->16097,"polblogs"->16717,"web-spam"->37375,"bio-CE-PG"->47309,"bio-SC-HT"->63023|>;*)
(*tDict=<|"facebook"->4478819,"hamsterster"->530606,"polblogs"->1038613,"web-spam"->2278717,"bio-CE-PG"->4982692,"bio-SC-HT"->6940974|>;*)
(*TDict=<|"facebook"->4836030,"hamsterster"->157953,"polblogs"->303129,"web-spam"->387051,"bio-CE-PG"->2353812,"bio-SC-HT"->4192980|>;*)
(*dataset="facebook";*)
(*n0=nDict[dataset];*)
(*m0=mDict[dataset];*)
(*t0 =tDict[dataset];*)
(*T0=TDict[dataset]/3;*)
(*ratio0=T0/t0;*)
(*R0=100000;*)


(* ::Input:: *)
(*$MinPrecision=50;*)
(*$MaxPrecision=50;*)
(*Clear[n,m,a,R];*)
(*p[n_,m_]:=m/(n*(n-1)/2);*)
(*pg3[a_]:=a^3;*)
(*pg2[a_]:=(1-a)*a*a;*)
(*pglt2[a_]:=1-pg3[a]-3*pg2[a];*)
(*pgge2[a_]:=pg3[a]+3*pg2[a];*)
(*pgge2ex1[a_]:=pg3[a]+2*pg2[a];*)
(*pp3[a_,R_]:=pg3[a]*(1-pglt2[a]^(R-1)) /pgge2[a];*)
(*pp1all[a_,R_]:=(pg3[a]*(1-pglt2[a])^(R-1)/pgge2ex1[a])-pp3[a,R];*)
(*ppindep[a_,R_]:=1-pp3[a,R]-3*pp1all[a,R];*)
(*P3[n_,m_,a_,R_]:=pp3[a,R]*p[n,m]+3*pp1all[a,R]*p[n,m]*p[n,m]+ppindep[a,R]*p[n,m]*p[n,m]*p[n,m];*)
(*P2[n_,m_,a_,R_]:=pp1all[a,R]*(1-p[n,m])*p[n,m]+ppindep[a,R]*(1-p[n,m])*p[n,m]*p[n,m];*)
(*P1[n_,m_,a_,R_]:=pp1all[a,R]*(1-p[n,m])*p[n,m]+ppindep[a,R]*(1-p[n,m])*(1-p[n,m])*p[n,m];*)
(*E3[n_,m_,a_,R_]:=Binomial[n,3]*P3[n,m,a,R];*)
(*E2[n_,m_,a_,R_]:=Binomial[n,3]*P2[n,m,a,R]*3;*)
(*E32ratio[n_,m_,a_,R_]:=E3[n,m,a,R]/E2[n,m,a,R];*)
(*p0=p[n0,m0];*)


(* ::Input:: *)
(*(* Find the alpha to fit the expected number of triangles *)*)
(*axT=a/.FindRoot[E3[n0,m0,a,R0]==T0,{a,0.001},WorkingPrecision->50];*)
(*(* Find the alpha to fit the closed-triangle ratio *)*)
(*axR = a /. FindRoot[E32ratio[n0,m0,a,R0]==ratio0,{a,0.001},WorkingPrecision->50];*)


(* ::Input:: *)
(*Print[n0, " ", NumberForm[N[p0,50]]," ", NumberForm[N[axT,50]], " ", R0, "\n\n", n0, " ", NumberForm[N[p0,50]]," ", NumberForm[N[axR,50]], " ", R0];*)
