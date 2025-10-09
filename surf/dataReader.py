import numpy as np
import struct



if __name__=='__main__':
    dataPath = 'surf/trainTerm1.dat'
    data = []
    maxFeatId = np.array([0],dtype=np.uint32)
    with open(dataPath, 'rb') as file:
        dataType = struct.unpack('i', file.read(4))[0]
        itemCnt = struct.unpack('i', file.read(4))[0]
        featLevelCnt = struct.unpack('i', file.read(4))[0]
        potentialGridCnt = struct.unpack('i', file.read(4))[0]
        cnt = featLevelCnt*potentialGridCnt
        sphereHarmonicSize = struct.unpack('i', file.read(4))[0]
        rgbData = np.zeros([itemCnt, 3])
        dirData = np.zeros([itemCnt, sphereHarmonicSize])
        featsId = np.zeros(
            [itemCnt*potentialGridCnt, featLevelCnt], dtype=np.uint32)
        if dataType==1:
            print('dataType == 1')
            for i in range(itemCnt):
                rgbData[i,:] = np.array(struct.unpack('3f', file.read(12)))
                dirData[i, :] = np.array(struct.unpack('16f', file.read(4*16)))
                featsId[potentialGridCnt*i:potentialGridCnt*i+potentialGridCnt, :] = np.array(struct.unpack(
                    str(cnt)+'I', file.read(4*cnt)),dtype=np.uint32).reshape(potentialGridCnt, featLevelCnt)
            maxFeatId = np.max(featsId, axis=0)
        pass

