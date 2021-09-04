"""
Module for Sector Based Linear Regression
(See in: DOI: 10.14232/actacyb.23.4.2018.3)

Author: Nagy Gábor
"""

import numpy


def regnd(n):
    "The coordinates of the nodes of a regular n-dimensional hypertetrahedron, which centre is in the origin, and the distantces of the nodes from the origin are 1"
    if n==0:
        return [[]]
    else:
        return [ [ y*(1-(1/n)**2)**0.5 for y in x]+[-1/n] for x in regnd(n-1)]+[[0.0]*(n-1)+[1.0]]
        

def scalendl(lst, lm):
    "Scale a list of coordinates by lm"
    return [ [ y*lm for y in x] for x in lst]


def lintr(lst, la, lb):
    "Calculate the linear transformation of the coordinates of the lst by x'=ax+b with the parameteres of la and lb."
    return [ [ la[i]*li[i]+lb[i] for i in range(len(lst[0]))] for li in lst]

    
def calctr(ptlst):
    "Calculate the transfomation parameters for linear transform to (-1,...,-1)-(+1,...,+1) n-dimension cube every point of the lst"
    nr=range(len(ptlst[0]))
    if len(ptlst)<2:
        return ([1 for i in nr], [-ptlst[0][i] for i in nr])
    mincoord=list(map(min, numpy.transpose(ptlst)))
    maxcoord=list(map(max, numpy.transpose(ptlst)))
    apar=[ 2/(maxcoord[i]-mincoord[i]) for i in nr]
    return (apar, [-1-mincoord[i]*apar[i] for i in nr])
    

def minidx(lst):
    "Return the index of the minimal value of the list."
    wlst=list(lst)
    ret=0
    for i in range(1,len(wlst)):
        if wlst[i]<wlst[ret]:
            ret=i
    return ret


def linpar2cth(linpars):
    "Calculate centre heights from the linear parameteres."
    dim=len(linpars)-1
    ctcoords=[x+[1] for x in scalendl(regnd(dim), dim/(dim+1))]
    return [sum([x[i]*linpars[i] for i in range(dim+1)]) for x in ctcoords]


def cth2linpar(cths, chidx=-1, chcoord=[]):
    "Calculate linear parameters from the centre height."
    dim=len(cths)-1
    ctcoords=[x+[1] for x in scalendl(regnd(dim), dim/(dim+1))]
    wcths=list(cths)
    if chidx>-1:
        ctcoords[chidx]=chcoord[:-1]+[1]
        wcths[chidx]=chcoord[-1]
    return list(numpy.linalg.solve(ctcoords, wcths))


def sblr(ptlst, q=0.5):
    "Calculate linear equation to the pointlist by SBLR method"
    dim=len(ptlst[0])-1
    ctps=scalendl(regnd(dim),dim/(dim+1))
    ctcoords=[x+[1] for x in ctps]
    ptsep=[[] for x in range(dim+1)]
    pa,pb=calctr(ptlst)
    pa[-1]=1
    pb[-1]=0
    for pti in lintr(ptlst, pa, pb):
        ptsep[minidx([ sum([(pti[j]-ctps[i][j])**2 for j in range(dim)]) for i in range(dim+1)])].append(pti)
    if min(map(len, ptsep))<1:
        raise ValueError('Wrong distribution, empty sector!')
    cths=[numpy.percentile([x[-1] for x in si], q*100) for si in ptsep]
    fullstep=0
    noch=0
    secid=0
    while noch<dim+1 and fullstep<100*dim**2:
        oldcth=cths[secid]
        cths[secid]=numpy.percentile([linpar2cth(cth2linpar(cths, secid, x))[secid] for x in ptsep[secid]], q*100)
        if abs(oldcth-cths[secid])>1E-5:
            noch=0
        else:
            noch+=1
        secid=(secid+1)%(dim+1)
        fullstep+=1
    retpar=cth2linpar(cths)
    return [retpar[i]*pa[i] for i in range(dim)]+[sum([retpar[i]*pb[i] for i in range(dim)])+retpar[-1]]
        
    
