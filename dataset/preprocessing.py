import dataset
from random import shuffle

# Filter the img sizes and classes
def respectTheRule(i):
    respectDermoscopic = True
    respectSize = True
    respectBenignMalignant = True
    respectDiagnosisConfirm = True

    # if i['size_x'] == 3024 and i['size_y'] == 2016:
    if i['size_x'] != 600 and i['size_y'] != 450:
        respectSize = False

    if i['type'] != 'dermoscopic':
        respectDermoscopic = False

    if i['diagnosis_confirm_type'] == None:
        respectDiagnosisConfirm = False

    return respectDermoscopic and respectSize and respectBenignMalignant and respectDiagnosisConfirm

def filterIndex(index=dataset.loadIndex()):
    filteredKeys = list(filter(lambda k: respectTheRule(index[k]), index.keys()))
    filteredImgs = {}
    for k in filteredKeys:
        filteredImgs[k] = index[k]
    return filteredImgs

def transformIndexToInOutArrays(index):
    _in = []
    _out = []
    labels = []
    for k,v in index.items():
        imgClass = '%s_%s' % (v['diagnosis'], v['benign_malignant'])

        if imgClass not in labels:
            labels.append(imgClass)

        _in.append(k) # The input is the id os img
        _out.append(labels.index(imgClass)) # The output is the index of the img class

    return _in, _out, labels

# Example:
#trainIns, trainOuts, testIns, testOuts, labels = preprocessing.getTrainingSets()
def getTrainingSets(testProportion=.2):
    index = filterIndex()
    _in,_out,labels = transformIndexToInOutArrays(index)

    byClassIns = {}
    trainById = {}
    testById = {}

    for i in range(len(labels)):
        byClassIns[i] = []

    for i in range(len(_in)):
        byClassIns[_out[i]].append(_in[i])

    # Shuffle the imgs
    for i in range(len(labels)):
        shuffle(byClassIns[i])

    for i in range(len(labels)):
        howManyToTest = int(len(byClassIns[i])*testProportion)
        for id in byClassIns[i][howManyToTest:]:
            trainById[id] = i
        for id in byClassIns[i][:howManyToTest]:
            testById[id] = i

    # At this point trainById and testById are like: {'<img_id>':output}
    trainIns = []
    trainOuts = []
    testIns = []
    testOuts = []
    for i,o in trainById.items():
        trainIns.append(i)
        trainOuts.append(o)

    for i,o in testById.items():
        testIns.append(i)
        testOuts.append(o)

    # Mix the classes in training set
    tempArray = []
    for i in range(len(trainIns)):
        tempArray.append({trainIns[i]:trainOuts[i]})
    trainIns = []
    trainOuts = []
    shuffle(tempArray)
    for inOut in tempArray:
        for k,v in inOut.items():
            trainIns.append(k)
            trainOuts.append(v)

    # Change the imgId by the imgName
    for i in range(len(trainIns)):
        trainIns[i] = index[trainIns[i]]['name']
    for i in range(len(testIns)):
        testIns[i] = index[testIns[i]]['name']

    return trainIns, trainOuts, testIns, testOuts, labels
