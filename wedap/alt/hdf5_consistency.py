
def consistent(h5f):
    result=True 
    violations=0
    for i,v in enumerate(h5f['iterations'].values()): 
        if i==0: 
            prevset=set([x[-1][0] for x in v['pcoord'][:]]) 
            print("iteration",i+1,prevset) 
        else: 
            newset=set([x[0][0] for x in v['pcoord'][:]]) 
            if not newset.issubset(prevset): 
                print("consistency violation iter",i,prevset,newset) 
                result=False 
                violations+=1 
            prevset=set([x[-1][0] for x in v['pcoord'][:]]) 
            print("iteration",i+1,prevset) 
    if violations>0: 
        print("N_violations:",violations) 
    return result