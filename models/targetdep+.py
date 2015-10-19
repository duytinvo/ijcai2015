# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import gensim
from gensim import utils
from twtokenize import tokenize
from collections import defaultdict
import argparse

class twprocess(object):
    def __init__(self,tw):
        self.tweet=self.wdprocess(tw)
    def lemma(self,wrd):
        tokens=['ing','s','es','ies','ed','er','or','ful','ic','ly','cally','ically']
        for token in tokens:
            if len(wrd)>=4 and wrd.endswith(token):
                wrd=wrd.rstrip(token)
                break
        return wrd
    def wrdprocess(self,wrd):
        '''
                process the lenthen word
        '''
#        wrd=self.lemma(wrd)
        x=defaultdict(int)
        for i in xrange(len(wrd)-1):
            if wrd.count(wrd[i])>3 and wrd[i]==wrd[i+1]:    
                x[wrd[i]]+=1
        for w,num in x.iteritems():
            wrd=wrd.replace(w*num,w*2)
        return wrd
    def wdprocess(self,tw):
        for i in xrange(len(tw)):    
            if tw[i].startswith('@'):
                tw[i]=tw[i].replace(tw[i],'<username>')
            if tw[i].startswith('http'):
                tw[i]=tw[i].replace(tw[i],'<url>')
            if tw[i].isdigit():
                tw[i]='<digit>'
#            tw[i]=self.wrdprocess(tw[i])
        return tw
#----------------------------------------------------------------------------
class streamtw(object):
    """Iterate over sentences from the Brown corpus (part of NLTK data)."""
    def __init__(self, dirname):
        self.i=0
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):

            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for line in utils.smart_open(fname):
                self.i+=1
                if self.i==1:
                    tw=line.lower().strip()
                if self.i==2:
                    target=line.lower().strip()
                if self.i==3:
                    senti=int(line.strip())+1
                    tw=tw.replace(target,' '+target+' ')
                    tw=tw.replace(''.join(target.split()),' '+'_'.join(target.split())+' ')
                    tw=tw.replace(target,' '+'_'.join(target.split())+' ')
                    tweet=tokenize(tw)
#                    tweetpro=twprocess(tweet).tweet                    
                    yield (tweet,'_'.join(target.split()),senti)
                    self.i=0
#------------------------------------------------------------------------------
class streampos(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for line in utils.smart_open(fname):
                tw,pos,score,tokens=line.strip().split('\t')
                yield pos
#------------------------------------------------------------------------------

def sentiscore(fname):
    x=defaultdict(float)
    with open(fname,'r') as f:
        for line in f:
            word,score=line.strip().split('\t')
            x[word]=float(score)
    #        data.append([word,score,npos,nneg])
    return x
score=sentiscore('../sources/lexicons/Maxdiff-Twitter-Lexicon.txt')
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class lexicon(object):
    def __init__(self,Wfile='../sources/lexicons/wilson/subjclueslen1-HLTEMNLP05.tff',
                 Bfiles=['../sources/lexicons/binhliu/negative-words.txt','../sources/lexicons/binhliu/positive-words.txt'],
                 Mfile='../sources/lexicons/mohammad/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt',size=100):
        self.pos,self.neg,self.neu=self.Wilson(Wfile)
#        self.pos,self.neg=self.Mohammad(Mfile)
        self.updateBinhliu(Bfiles)
#        self.updateMohammad(Mfile)
        self.intersect()
#        self.reneg,self.reppos=self.sentirepresentation(size)
    def extract(self,word):
        return word.split('=')[-1]
    def Wilson(self,filename):
        neg=[]
        pos=[]
        neu=[]
        with open(filename,'r') as f:
            for line in f:
                wtype,wlen,word1,pos1,stemmed1,priorpolarity=line.strip().split(' ')
                if self.extract(priorpolarity)=='negative':
    #                t=[extract(word1),extract(wtype)+'_'+extract(priorpolarity),extract(pos1),extract(stemmed1)]
                    neg.append(self.extract(word1))
                elif self.extract(priorpolarity)=='positive':
    #                t=[extract(word1),extract(wtype)+'_'+extract(priorpolarity),extract(pos1),extract(stemmed1)]
                    pos.append(self.extract(word1))
                else:
    #                t=[extract(word1),extract(wtype)+'_'+extract(priorpolarity),extract(pos1),extract(stemmed1)]
                    neu.append(self.extract(word1))
        pos=set(pos)
        neg=set(neg)
        neu=set(neu)
        return pos,neg,neu
    def Mohammad(self,filename):
        neg=[]
        pos=[]
        with open(filename,'r') as f:
            for line in f:
                wd,senti,flag=line.strip().split('\t')
                if senti=='negative' and flag=='1':
                    neg.append(wd)
                elif senti=='positive' and flag=='1':
                    pos.append(wd)
                else:
                    pass
        pos=set(pos)
        neg=set(neg)
        return pos,neg
    def Binhliu(self,filename):    
        neg=np.loadtxt(filename[0],skiprows=1,dtype='string',comments=';')
        pos=np.loadtxt(filename[1],skiprows=1,dtype='string',comments=';')
        pos=set(pos)
        neg=set(neg)
        return pos,neg
    def updateBinhliu(self,Bfiles):
        lexBinh=self.Binhliu(Bfiles)
        self.pos.update(lexBinh[0])
        self.neg.update(lexBinh[1])
    def updateMohammad(self,Mfile):
        lexMoh=self.Mohammad(Mfile)
        self.pos.update(lexMoh[0])
        self.neg.update(lexMoh[1])
    def intersect(self):
        temp1=self.pos.intersection(self.neg)
        temp2=self.pos.intersection(self.neu)
        temp3=self.neg.intersection(self.neu)
        self.pos.difference_update(temp1)
        self.pos.difference_update(temp2)
        self.neg.difference_update(temp1)
        self.neg.difference_update(temp3)
        self.neu.difference_update(temp2)
        self.neu.difference_update(temp3)
    def intersect2(self):
        temp1=self.pos.intersection(self.neg)
        self.pos.difference_update(temp1)
        self.neg.difference_update(temp1)
#    def sentirepresentation(self,size):
#        neg=np.array([]) 
#        pos=np.array([])
#        for i in self.neg:
#            try:
#                neg=np.concatenate([neg,embedmodel[i]])
#            except:
#                pass 
#        for j in self.pos:
#            try:
#                pos=np.concatenate([pos,embedmodel[j]])
#            except:
#                pass 
#        neg=neg.reshape(len(neg)/size,size)
#        repneg=neg.mean(axis=0)
#        pos=pos.reshape(len(pos)/size,size)
#        reppos=pos.mean(axis=0)
#        return repneg,reppos

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class streamdata(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.rstrip().split('\t')
def readTang(fname='../sources/wordemb/sswe'):
    embs=streamdata(fname)
    embedmodel={}
    for tw2vec in embs:
        wd=tw2vec[0]
        value=[float(i) for i in tw2vec[1:]]
        embedmodel[wd]=np.array(value)
    return embedmodel
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class targettw(object):
    def __init__(self,w2vf='../sources/wordemb/w2v/c10_w3_s100',sswef='../sources/wordemb/sswe'):
        self.w2v=gensim.models.Word2Vec.load(w2vf) 
        self.sswe=readTang(sswef) 
        self.lexicons=lexicon()
    def emdsswe(self,i,loc,uni,target):
        f=np.array([])    
        l=np.array([])
        r=np.array([])
        t=np.array([])
        ls=np.array([])
        rs=np.array([])
        #embbeddings of fulltw features
        f=self.sswe.get(uni,self.sswe['<unk>'])
        #embbeddings of  left context features
        if i<loc:
            l=self.sswe.get(uni,self.sswe['<unk>'])
            if self.lexicons.pos.issuperset(set([uni])):
                try:
                    ls=self.sswe[uni]
                except:
                    pass
            if self.lexicons.neg.issuperset(set([uni])):
                try:
                    ls=self.sswe[uni]
                except:
                    pass
        #embbeddings of  target features  
        elif(i==loc):
            t=self.sswe.get(target.replace('_',''),self.sswe['<unk>'])               
            target2=target.split('_')
            for wd in target2:
                t=self.sswe.get(wd,self.sswe['<unk>'])
            
        #embbeddings of  right context features
        else:
            r=self.sswe.get(uni,self.sswe['<unk>'])
            if self.lexicons.pos.issuperset(set([uni])):
                try:
                    rs=self.sswe[uni]
                except:
                    pass
            if self.lexicons.neg.issuperset(set([uni])):
                try:
                    rs=self.sswe[uni]
                except:
                    pass 
        return [f,l,t,r,ls,rs]
    def emdw2v(self,i,loc,uni,target):
        f=np.array([])    
        l=np.array([])
        r=np.array([])
        t=np.array([])
        ls=np.array([])
        rs=np.array([])
        try:
            f=self.w2v[uni]
        except:
            pass 
        #embbeddings of  left context features
        if i<loc:
            try:
                l=self.w2v[uni]
            except:
                pass
            try:
                if self.lexicons.pos.issuperset(set([uni])):
                    ls=self.w2v[uni]
            except:
                pass
            try:
                if self.lexicons.neg.issuperset(set([uni])):
                    ls=self.w2v[uni]
            except:
                pass
        #embbeddings of  target feature  
        elif(i==loc):
            try:
                t=self.w2v[target.replace('_','')]
            except:
                pass                
            target2=target.split('_')
            for wd in target2:
                try:
                    t=self.w2v[wd]
                except:
                    pass    
        #embbeddings of  right context features
        else:
            try:
                r=self.w2v[uni]
            except:
                pass
            try:
                if self.lexicons.pos.issuperset(set([uni])):
                    rs=self.w2v[uni]
            except:
                pass
            try:
                if self.lexicons.neg.issuperset(set([uni])):
                    rs=self.w2v[uni]
            except:
                pass   
        return [f,l,t,r,ls,rs]

    def concattw(self,feature,size,tw,etype,loc,target):
        fulltw=np.array([])    
        left=np.array([])
        right=np.array([])
        tar=np.array([])
        leftsenti=np.array([])
        rightsenti=np.array([])
        for i,uni in enumerate(tw):
            if etype=='w2v':
                f,l,t,r,ls,rs=self.emdw2v(i,loc,uni,target)
            if etype=='sswe':
                f,l,t,r,ls,rs=self.emdsswe(i,loc,uni,target)
            fulltw=np.concatenate([fulltw,f])
            left=np.concatenate([left,l])
            tar=np.concatenate([tar,t])
            right=np.concatenate([right,r])
            leftsenti=np.concatenate([leftsenti,ls])
            rightsenti=np.concatenate([rightsenti,rs])
        #padding       
        if list(left)==[]:
            left=np.zeros((2*size,))
        if list(right)==[]:
            right=np.zeros((2*size,))
        if list(fulltw)==[]:
            fulltw=np.zeros((2*size,)) 
        if list(tar)==[]:
            tar=np.zeros((2*size,))
#
        if len(left)<=size:
            left=np.concatenate([left,np.zeros((size,))])
        if len(right)<=size:
            right=np.concatenate([right,np.zeros((size,))])
        if len(fulltw)<=size:
            fulltw=np.concatenate([fulltw,np.zeros((size,))])
        if len(tar)<=size:
            tar=np.concatenate([tar,np.zeros((size,))])   

        if list(leftsenti)==[]:
            leftsenti=np.zeros((size,))
        if list(rightsenti)==[]:
            rightsenti=np.zeros((size,))       

        fulltw=fulltw.reshape(len(fulltw)/size,size)
        left=left.reshape(len(left)/size,size)
        right=right.reshape(len(right)/size,size)        
        tar=tar.reshape(len(tar)/size,size)
        leftsenti=leftsenti.reshape(len(leftsenti)/size,size)   
        rightsenti=rightsenti.reshape(len(rightsenti)/size,size)  
        #target-ind features
        feature=np.concatenate([feature,fulltw.max(axis=0)])
        feature=np.concatenate([feature,fulltw.min(axis=0)])
        feature=np.concatenate([feature,fulltw.mean(axis=0)]) 
        feature=np.concatenate([feature,fulltw.std(axis=0)])
        feature=np.concatenate([feature,fulltw.prod(axis=0)])
        #target-dep features
        feature=np.concatenate([feature,left.max(axis=0)])
        feature=np.concatenate([feature,tar.max(axis=0)])
        feature=np.concatenate([feature,right.max(axis=0)]) 
              
        feature=np.concatenate([feature,left.min(axis=0)])  
        feature=np.concatenate([feature,tar.min(axis=0)])
        feature=np.concatenate([feature,right.min(axis=0)])  
        
        feature=np.concatenate([feature,left.mean(axis=0)])      
        feature=np.concatenate([feature,tar.mean(axis=0)])       
        feature=np.concatenate([feature,right.mean(axis=0)]) 
        
        feature=np.concatenate([feature,left.std(axis=0)])
        feature=np.concatenate([feature,tar.std(axis=0)])
        feature=np.concatenate([feature,right.std(axis=0)])  
        
        feature=np.concatenate([feature,left.prod(axis=0)])
        feature=np.concatenate([feature,tar.prod(axis=0)])
        feature=np.concatenate([feature,right.prod(axis=0)]) 
        #target-dep senti features
        feature=np.concatenate([feature,leftsenti.sum(axis=0)])
        feature=np.concatenate([feature,rightsenti.sum(axis=0)])
        feature=np.concatenate([feature,leftsenti.max(axis=0)])
        feature=np.concatenate([feature,rightsenti.max(axis=0)])
        return feature
    def allfeat(self,dataf):
        size1=len(self.w2v['the'])
        size2=len(self.sswe['the'])
        twtriple=streamtw(dataf)
        y=[]
        x=np.array([])
        for triple in twtriple:
            feature=np.array([])
            tw=triple[0]
            target=triple[1]
            y.append(triple[2])
            loc=tw.index(target) 
            feature=self.concattw(feature,size1,tw,'w2v',loc,target) 
            feature=self.concattw(feature,size2,tw,'sswe',loc,target)
            x=np.concatenate([x,feature])
        x=x.reshape((len(y),len(x)/len(y)))
        return(x,y)

def writevec(filename,x,y):
    f=open(filename,'wb')
    for i in xrange(len(y)):
        f.write(str(y[i])+'\t')
        feature=x[i]
        for (j,k) in enumerate(feature):
            f.write(str(j+1)+':'+str(k)+' ')
        f.write('\n')
    f.close() 
    
if __name__ == "__main__":
    features=targettw()
    print "extracting features for training"
    x_train,y_train=features.allfeat('../data/training/')
    writevec('../data/output/training',x_train,y_train)
    print "extracting features for testing"
    x_test,y_test=features.allfeat('../data/testing/')
    writevec('../data/output/testing',x_test,y_test)
    



