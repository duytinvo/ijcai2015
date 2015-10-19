# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:28:45 2015

@author: duytinvo
"""

import subprocess
from argparse import ArgumentParser

def scaling():
    cmd1=["../libSVM/svm-scale -l -1 -u 1 -s ../liblinear/range ../data/output/training > ../liblinear/train.scale"]
    subprocess.call(cmd1)
    cmd2=["../libSVM/svm-scale -r ../liblinear/range ../data/output/testing > ../liblinear/test.scale"]
    subprocess.call(cmd2)



def predict(ci,trfile,tfile,pfile):
    traincmd=["../liblinear/train", "-c", "0.001", trfile]
    traincmd[2]=ci
    subprocess.call(traincmd)
    model=trfile.split('/')[-1]+'.model'
    predcmd=["../liblinear/predict", tfile, model, pfile]
    p = subprocess.Popen(predcmd, stdout=subprocess.PIPE)
    output, err = p.communicate()
    preddev=float(output.split()[2].strip('%'))
    print "Predict: Learning liblinear with c=%s: %f"%(ci,preddev)
    return output
    
def CV(ci,trfile):
    traincmd=["../liblinear/train", "-c", "0.001", "-v", "5", trfile]
    traincmd[2]=ci
    p = subprocess.Popen(traincmd, stdout=subprocess.PIPE)
    output, err = p.communicate()
    preddev=float(output.split()[-1].strip('%'))
    print "CV: Learning liblinear with c=%s: %f"%(ci,preddev)
    return preddev

def DEV(ci,trfile,devfile,pfile='../liblinear/pred.tune.conll'):
    traincmd=["../liblinear/train", "-c", "0.001", trfile]
    traincmd[2]=ci
    subprocess.call(traincmd)
    model=trfile.split('/')[-1]+'.model'
    predcmd=["../liblinear/predict", devfile, model, pfile]
    p = subprocess.Popen(predcmd, stdout=subprocess.PIPE)
    output, err = p.communicate()
    preddev=float(output.split()[2].strip('%'))
    print "DEV: Learning liblinear with c=%s: %f"%(ci,preddev)
    return preddev    
    
def frange(start, stop, step):
    r = start
    while r <= stop:
        yield r
        r *= step
def tuneC(trfile):
    c=[]
    crange=frange(0.00001,1,10)
    c.extend([i for i in crange])
    crange=frange(0.00003,3,10)
    c.extend([i for i in crange])
    crange=frange(0.00005,5,10)
    c.extend([i for i in crange])
    crange=frange(0.00007,7,10)
    c.extend([i for i in crange])
    crange=frange(0.00009,10,10)
    c.extend([i for i in crange])
    c.sort()
    tunec=[]
    for ci in c:
        tunec.append([ci,CV(str(ci),trfile)])
    tunec=sorted(tunec,key=lambda x: x[1],reverse=True)
    return tunec
    
def main(trfile,tfile,pfile):
    tunec=tuneC(trfile)
    test=predict(str(tunec[0][0]),trfile,tfile,pfile)
    bestc=tunec[0][0]
    bestCV=tunec[0][-1]
    bestAcc=test
    print '\n'    
    print 80*'*'
    print "Tuning: Five-fold CV on %s, the best accuracy is %f at c=%f"%(trfile,bestCV,bestc)
    #print "Tuning: Testing on %s, the best accuracy is %f at c=%f"%(devfile,bestCV,bestc)
    print "Testing: Testing on %s, %s"%(tfile,bestAcc)
    print 80*'*'
#main("train.conll.basic.scale","test.conll.basic.scale","pred.conll.basic.scale")
#main("train.conll.basic.tune.scale","test.conll.basic.tune.scale","pred.conll.basic.tune.scale")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--trainfile", dest="trainfile", help="Training file", default='../liblinear/train.scale')
    #parser.add_argument("--devfile", dest="devfile", help="Dev file", default='./liblinear/devconll')
    parser.add_argument("--testfile", dest="testfile", help="Testing file", default='../liblinear/test.scale')
    parser.add_argument("--predfile", dest="predfile", help="Predicted file", default='../liblinear/predresults')
    args = parser.parse_args()
    main(args.trainfile,args.testfile,args.predfile)