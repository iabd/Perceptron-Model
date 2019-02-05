#                                                                                                                                      
#  Created by Abhinav Dwivedi on 25/01/2019.                                                                                                                             
#  Copyright © 2019 Abhinav Dwivedi. All rights reserved.                                                                                              
# 
import numpy as np
import random
import operator

def loadMonk(file_, filetype, encodeLabel=False):
    filename="data/monk/monks-{}.{}".format(file_, filetype)
    def encode(vector, label=False):
        if label:
            twoFeatures = {'0': [1, 0], '1': [0, 1]}
            return twoFeatures[str(vector)]

        else:
            retVector=[]
            twoFeatures={'1':[1,0], '2':[0,1]}
            threeFeatures={'1':[1,0,0],'2':[0,1,0],'3':[0,0,1]}
            fourFeatures={'1':[1,0,0,0],'2':[0,1,0,0],'3':[0,0,1,0],'4':[0,0,0,1]}
            encodingDict={
                '0':threeFeatures,
                '1':threeFeatures,
                '2':twoFeatures,
                '3':threeFeatures,
                '4':fourFeatures,
                '5':twoFeatures
            }
            for idx, val in enumerate(vector):
                retVector.extend(encodingDict[str(idx)][str(val)])
            return retVector

    with open(filename) as f:
        data_=[]
        labels=[]
        for line in f.readlines():
            rows=[x for x in line.split(' ')][2:-1]
            temp=encode(rows)
            assert len(temp)==17
            data_.append(encode(rows))
            label=line[1]
            if encodeLabel:
                label=encode(label, label=True)
            else:
                label=[label]
            labels.append(label)
            # breakpoint()
        data_=np.array(data_, dtype='float16')
        labels=np.array(labels, dtype='float16')

    return data_, labels
