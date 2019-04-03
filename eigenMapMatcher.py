import cv2
import argparse
import numpy as np
from scipy.sparse import csgraph
from numpy import linalg as LA
from scipy.spatial.distance import cdist, squareform
from scipy import interpolate
import tkinter
import matplotlib.pyplot as plt

def imagePreProcess(image):#to read to gray, scale
    #Resizing image to 48000 pixels.
    im = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    print(im.shape)
    h,w = im.shape #height and width
    tp = 48000 #total pixels
    scale = np.sqrt(tp/(h*w))
    print('scale factor',scale)
    res=cv2.resize(im, None, fx=scale, fy=scale, interpolation =
                   cv2.INTER_CUBIC)
    print('resized image',res.shape)
    return res,im

class DenseDetector(): 
    def __init__(self, step_size=20, feature_scale=20, img_bound=20): 
        # Create a dense feature detector 
        self.initXyStep = step_size
        self.initFeatureScale = feature_scale
        self.initImgBound = int(img_bound)
 
    def detect(self, img):
        keypoints = []
        cols, rows = img.shape[:2]
        for i, x in enumerate(range(self.initImgBound, rows, self.initXyStep)):
            for j, y in enumerate(range(self.initImgBound, cols, self.initXyStep)):
                keypoints.append(cv2.KeyPoint(float(x), float(y),
                                              self.initFeatureScale))
        print("arun printing each iteration i,j",i,j)
        return keypoints

class mserRelated():
    def preprocessImages(self, image, h, w):
        #crop the image
        width = image.shape[1] #width
        image = image[:,int(width/3):int(2*width/3),:]
        #resize the image
        finalImage = cv2.resize(image, (w,h), interpolation =
                                cv2.INTER_CUBIC)
        return finalImage

    def detectKP(self, im1, im2, origIm1, origIm2, imageName):
        mserDet = cv2.MSER_create()
        kp2 = mserDet.detect(im2)
        kp1 = mserDet.detect(im1)

        imKP1 = cv2.drawKeypoints(im1, kp1, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        imKP2 = cv2.drawKeypoints(im2, kp2, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        sift1=cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift1.compute(im1, kp1)
        sift2=cv2.xfeatures2d.SIFT_create()
        kp2, des2 = sift2.compute(im2, kp2)


        #bf = cv2.BFMatcher_create(cv2.NORM_HAMMING2, True)#cv2.BFMatcher()
        bf = cv2.BFMatcher()
        #matches = bf.match(des1,des2)#bf.knnMatch(des1,des2, k=2)
        matches = bf.knnMatch(des1,des2, k=2)

        good = []
        for m, n in matches:
                if m.distance < 0.99 * n.distance:
                            good.append(m)

        good = sorted(good, key=lambda x: x.distance)
        print(len(good))

        matching_result = cv2.drawMatches(im1, kp1, im2, kp2, good[-4:], None,
                                          flags=2)

        matching_result2 = cv2.drawMatches(origIm1, kp1, origIm2, kp2,
                                           good[-4:], None,
                                          flags=2)

        resultOutputPath="./matchesOutput/"
        cv2.imwrite(resultOutputPath+imageName, matching_result)

        cv2.imwrite(resultOutputPath+"orig_"+imageName, matching_result2)


        return kp1, kp2, imKP1, imKP2


def detAndComputeSiftAtGivenScale(im, step, pixelScale, startKeyPoint =
                                  None):

    if startKeyPoint == None:
        startKeyPoint = pixelScale/2
    denseDetector = DenseDetector(step, pixelScale, startKeyPoint) 
    kp = denseDetector.detect(im)
    imKeyPoints = cv2.drawKeypoints(im,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("kpImage",imKeyPoints)
    cv2.waitKey(0)

    #compute sift
    sift=cv2.xfeatures2d.SIFT_create()
    kpOutput,denseFeat=sift.compute(im,kp)
    return kpOutput, denseFeat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-im1','--image1',required=True,help="input image 1")
    parser.add_argument('-im2','--image2',required=True,help='input image 2')
    args = parser.parse_args()

    #read and preprocess image
    im1,im1Uns = imagePreProcess(args.image1)
    im2,im2Uns = imagePreProcess(args.image2)


    #show the unscaled image
    cv2.imshow("im1", im1Uns)
    cv2.waitKey(0)
    cv2.imshow("im2", im2Uns)
    cv2.waitKey(0)
    #show the unscaled image
    cv2.imshow("im1", im1)
    cv2.waitKey(0)
    cv2.imshow("im2", im2)
    cv2.waitKey(0)

    procInputPath = "./processedInput"
    cv2.imwrite(procInputPath+"/"+args.image1,im1)
    cv2.imwrite(procInputPath+"/"+args.image2,im2)
    #compute d10enseSift in step of 4, for 10 neighborhood and 6 neighborhood
    #..separately
    imstep = 10 #step of keypoints
    im1kp10, im1denseFeat10 = detAndComputeSiftAtGivenScale(im1, imstep, 10)
    im1kp6, im1denseFeat6 = detAndComputeSiftAtGivenScale(im1, imstep, 6)
    print(im1denseFeat10.shape,len(im1kp10),im1denseFeat6.shape, len(im1kp6))

    #make the 10 and 6 scale sizes equal
    im1kp6=im1kp6[0:len(im1kp10)]
    im1denseFeat6 = im1denseFeat6[0:im1denseFeat10.shape[0],:]

    im1combinedDenseFeat = np.concatenate((im1denseFeat10, im1denseFeat6),axis=1)

    im2kp10, im2denseFeat10 = detAndComputeSiftAtGivenScale(im2, imstep, 10)
    im2kp6, im2denseFeat6 = detAndComputeSiftAtGivenScale(im2, imstep, 6)
    print(im2denseFeat10.shape,len(im2kp10),im2denseFeat6.shape, len(im2kp6))

    #make the 10 and 6 scale sizes equal
    im2kp6=im2kp6[0:len(im2kp10)]
    im2denseFeat6 = im2denseFeat6[0:im2denseFeat10.shape[0],:]

    im2combinedDenseFeat = np.concatenate((im2denseFeat10, im2denseFeat6),axis=1)

    print(im1combinedDenseFeat.shape,im2combinedDenseFeat.shape)

    #concatenate the two image graphs vertically.
    combinedImage =np.concatenate((im1combinedDenseFeat,im2combinedDenseFeat),axis=0)

    #create a laplacian matrix
    similarityMatrix = np.dot(combinedImage,combinedImage.T)
    #similarityMatrix = cdist(combinedImage,combinedImage, 'cosine')
    imageLaplacian = csgraph.laplacian(similarityMatrix, normed=True)

    #Eigen values and vectors.
    V,E = LA.eig(imageLaplacian)

    tempbestEigVec = E[:,np.where(V == np.partition(V,1)[1])[0][0]]
    fiveBestEigen = E[:,np.in1d(V, np.sort(np.partition(V,6))[1:6])][:,:3]
    #bestEigVec = fiveBestEigen[:,1]
    #sortedE = E[:,np.argsort(V)]

    #override five best to just 1 best
    #fiveBestEigen = tempbestEigVec.reshape(-1,1)

    #tu,ts,tv=np.linalg.svd(imageLaplacian)


    #First get shape of grid for given keypoints (so that eigen can be
    #...interpolated over it
    valim1Rows10 = np.unique(np.asarray([points.pt for points in
                                      im1kp10])[:,0])
    valim1Col10 = np.unique(np.asarray([points.pt for points in
                                     im1kp10])[:,1])




    countim1Rows10 = np.unique(np.asarray([points.pt for points in
                                      im1kp10])[:,0]).shape[0]
    countim1Col10 = np.unique(np.asarray([points.pt for points in
                                     im1kp10])[:,1]).shape[0]


    x1, y1 = np.meshgrid(valim1Rows10, valim1Col10)
    #x1, y1 = x1.T, y1.T #convert to 19*25 grid (from the 25*19 grid that meshgrid created)

    stretchim1Rows10 =  np.linspace(valim1Rows10[0].astype(int),valim1Rows10[-1].astype(int),valim1Rows10.shape[0]*100)
    stretchim1Col10 = np.linspace(valim1Col10[0].astype(int),valim1Col10[-1].astype(int),valim1Col10.shape[0]*100)
    sx1, sy1 = np.meshgrid( stretchim1Rows10 , stretchim1Col10)#for plotting purposes


    fig, ax = plt.subplots(nrows=3, ncols=2)

    for i, bestEigVec in enumerate(fiveBestEigen.T):
        bestEigVec = bestEigVec.reshape(-1,1)
        ind1 = np.round((bestEigVec.shape[0])/2).astype(int)
        eigen1 = bestEigVec[0:ind1].reshape(x1.T.shape).T
    
        interp1 = interpolate.interp2d(x1, y1, eigen1, kind = 'linear')
    
        eigenInterp1 = interp1(stretchim1Rows10,stretchim1Col10)
    
        eigen2 = bestEigVec[ind1:].reshape(x1.T.shape).T
    
        interp2 = interpolate.interp2d(x1, y1, eigen2, kind = 'linear')
    
        eigenInterp2 = interp2(stretchim1Rows10,stretchim1Col10)

        ax[i,0].axis('off')
        ax[i,0].imshow(eigenInterp1,cmap='jet')
        extent = ax[i,0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig('eig1Img_'+str(i)+".png", dpi=200, bbox_inches=extent, format='png', facecolor=fig.get_facecolor(), transparent=True)
        
        #colorEigenInterp1 = cv2.applyColorMap(eigenInterp1.astype('uint8'), colormap=cv2.COLORMAP_JET) 
        #cv2.imwrite("myFile.jpg", colorEigenInterp1)

        
        #################IMG2  END
        #ax[i,0].pcolormesh(sx1, sy1, eigenInterp1)
        #ax[i,0].invert_yaxis()
        #ax[i,0].set_title('EigenVector-'+str(i+1))
    
    
        ax[i,1].axis('off')
        ax[i,1].imshow(eigenInterp2,cmap='jet')
        extent2 = ax[i,1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig('eig2Img_'+str(i)+".png", dpi=200, bbox_inches=extent2, format='png', facecolor=fig.get_facecolor(), transparent=True)
        #ax[i,1].pcolormesh(sx1, sy1, eigenInterp2)
        #ax[i,1].invert_yaxis()
        #ax[i,1].set_title('EigenVector-'+str(i+1))


    mser = mserRelated()
    jetEigen1 = cv2.imread("eig1Img_2.png")
    jetEigen2 = cv2.imread("eig2Img_2.png")


    cv2.imshow("debugFrame",jetEigen2)
    cv2.waitKey(0)
    #crop and resize jetEigen
    h, w= im1.shape
    processJetEigen1 = mser.preprocessImages(jetEigen1, h, w) #shape of original scaled image.
    processJetEigen2 = mser.preprocessImages(jetEigen2, h, w) #shape of original scaled image.
    cv2.imshow("debugFrame2",processJetEigen2)
    cv2.waitKey(0)
    
    
    procOutputPath = "./processedOuput"
    cv2.imwrite(procOutputPath+"/e1_"+args.image1, processJetEigen1)
    cv2.imwrite(procOutputPath+"/e2_"+args.image2, processJetEigen2)

    kp1,kp2,imKP1,imKP2 = mser.detectKP(processJetEigen1, processJetEigen2,
                                        im1, im2, args.image1)

    cv2.imshow("kpImageMser",processJetEigen1)
    cv2.waitKey(0)
    cv2.imshow("kpImageMser",imKP1)
    cv2.imwrite(procOutputPath+"/kp_"+args.image1, imKP1)

    cv2.waitKey(0)


    cv2.imshow("kpImageMser",processJetEigen2)
    cv2.waitKey(0)
    cv2.imshow("kpImageMser",imKP2)
    cv2.imwrite(procOutputPath+"/kp_"+args.image2, imKP2)
    cv2.waitKey(0)



    plt.tight_layout()
    plt.show()


