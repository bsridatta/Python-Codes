def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def prune(tree, validation):
    resultTree = tree
    resultAcc = dtree.check(resultTree, validation)
    alternatives = dtree.allPruned(resultTree)
    #print("mashook pak eko")
    for alternative in alternatives:
        tempTree = alternative
        tempAcc = dtree.check(tempTree, validation)
        if(tempAcc > resultAcc):
            resultTree = tempTree
            resultAcc = tempAcc
            
    if(resultTree == tree):
        return resultTree

    return prune(resultTree, validation)

def assignment7(dataset, fractions):
    errorInitial = []
    errorPruned = []
    meanErrorInitial = []
    meanErrorPruned = []
    varianceErrorInitial = []
    varianceErrorPruned = []

    for fraction in fractions:
        for i in range(0,250):
            monktrain, monkval = partition(monk1, fraction)
            initialTree = dtree.buildTree(monktrain, m.attributes)
            prunedTree = prune(initialTree, monkval)
            errorInitial.append(1-dtree.check(initialTree, m.monk1test))
            errorPruned.append(1-dtree.check(prunedTree, m.monk1test))

        meanErrorInitial.append(np.mean(errorInitial))
        meanErrorPruned.append(np.mean(errorPruned))
        varianceErrorInitial.append(np.var(errorInitial))
        varianceErrorPruned.append(np.var(errorPruned))
    
    plot_result(meanErrorInitial, meanErrorPruned, varianceErrorInitial, varianceErrorPruned)

def plot_result(meanErrorInitial, meanErrorPruned, varianceErrorInitial, varianceErrorPruned):
    fig, ax =plt.subplots(1,2, figsize=(20,10))
    fig.subplots_adjust(wspace=.25)
    ax[0].set_title('Mean Error')
    ax[1].set_title('Variance')

    df = pd.DataFrame(np.c_[meanErrorInitial,meanErrorPruned], index=fractions).reset_index()
    df.columns = ['Fraction','Initial', 'Pruned']
    df = pd.melt(df, id_vars="Fraction", var_name="Tree", value_name="Mean Error")
    seaborn.factorplot(x='Fraction', y='Mean Error', hue='Tree', data=df, kind='bar', ax=ax[0])

    df = pd.DataFrame(np.c_[varianceErrorInitial,varianceErrorPruned], index=fractions).reset_index()
    df.columns = ['Fraction','Initial', 'Pruned']
    df = pd.melt(df, id_vars="Fraction", var_name="Tree", value_name="Variance Error")
    seaborn.factorplot(x='Fraction', y='Variance Error', hue='Tree', data=df, kind='bar', ax=ax[1])

## Monk 1
fractions = [.3, .4, .5, .6, .7, .8]
assignment7(m.monk1, fractions)
## Monk 3
fractions = [.3, .4, .5, .6, .7, .8]
assignment7(m.monk3, fractions)