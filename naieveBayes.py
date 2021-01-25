import numpy as np
import matplotlib.pyplot as plt
import cv2

def resizeImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dimensions = image.shape
    # height, width, number of channels in image
    height = dimensions[0]
    width = dimensions[1]
    y=0
    x=0
    h=0
    w=0
    if height > width:
        y = int((height-width)/2)
        h = width
        w = width
    else:
        x = int((width - height)/2)
        h = height
        w = height
    crop = image[y:y+h, x:x+w]
    dsize = (28, 28)
    resized = cv2.resize(crop, dsize, interpolation = cv2.INTER_AREA)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    output = cv2.filter2D(resized, -1, kernel)
    cv2.imwrite('Data/Individual/OutputSharpened.jpg', output)
    frame = np.asarray(output).reshape(-1)
    return frame

def seperate_classes(data,labels):
    sep_data = np.array([data[labels==i] for i in range(0,10)])
    return sep_data

def quantize(levels_no,data):
    return (data & 256-(256//levels_no))

def class_mean(classId, qId,train_qsp1):
    mean_image = np.sum(train_qsp1[classId][qId],axis=0)
    return mean_image/240

def total_mean(qId,train_qsp1):
    return np.array([class_mean(i,qId,train_qsp1) for i in range(0,10)])

def class_std(classId, qId, train_qsp1):
    avg = class_mean(classId, qId,train_qsp1)
    data = train_qsp1[classId][qId]
    variance = sum([(x-avg)**2 for x in data]) / float(len(data)-1)
    return np.sqrt(variance)

def total_std(qId,train_qsp1):
    return np.array([class_std(i,qId,train_qsp1) for i in range(0,10)])

def fullDatasetMean(qId,fullDS_train_q1):
    mean_image = np.sum(fullDS_train_q1[qId],axis=0)
    return mean_image/240

def fullDatasetStd(qId,fullDS_train_q1):
    avg = fullDatasetMean(qId,fullDS_train_q1)
    data = fullDS_train_q1[qId]
    variance = sum([(x-avg)**2 for x in data]) / float(len(data)-1)
    return np.sqrt(variance)


def trainModel():
    # Read Training Images, Testing Images, Training Labels, Testing Labels
    train = np.array([plt.imread('Data/Train/'+str(i)+'.jpg').reshape(-1) for i in range (1,2401)])
    train_labels = np.loadtxt('Data/Train/Training Labels.txt')
    train_sp=seperate_classes(train,train_labels)

    #Quantize Train images
    q_levels = [2, 4, 8, 16, 32, 64, 128, 256]
    train_qsp=np.array([np.array([quantize(i,train_sp[c]) for i in q_levels]) for c in range(0,10)])
    fullDS_train_q = np.array([quantize(i,train) for i in q_levels])
    mean_qsp=np.array([total_mean(i,train_qsp) for i in range(0,8)])
    std_qsp=np.array([total_std(i,train_qsp) for i in range(0,8)])

    save_mean = mean_qsp.reshape(8, -1)
    save_std = std_qsp.reshape(8, -1)

    np.savetxt('Data/MeanStd/mean.csv', save_mean)
    np.savetxt('Data/MeanStd/std.csv', save_std)

def calculate_probability(x, mean, stdev):
    exponent = np.exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

def naive_bayes(data):
    naive_probs=np.empty((10))
    for j in range(0,10):
        nz_values = data[j]
        c = np.nanprod(nz_values*100,dtype=np.float64)
        naive_probs[j]=c                
    return naive_probs

def testImage(im, qLevel, mean_qsp, std_qsp):

    #Adjusting image to match training dataset
    image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    dimensions = image.shape
    # height, width, number of channels in image
    height = dimensions[0]
    width = dimensions[1]
    y=0
    x=0
    h=0
    w=0
    if height > width:
        y = int((height-width)/2)
        h = width
        w = width
    else:
        x = int((width - height)/2)
        h = height
        w = height
    crop = image[y:y+h, x:x+w]
    dsize = (28, 28)
    resized = cv2.resize(crop, dsize, interpolation = cv2.INTER_AREA)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    output = cv2.filter2D(resized, -1, kernel)
    output = cv2.bitwise_not(output)
    cv2.imwrite('Data/Individual/OutputSharpened.jpg', output)
    test_sp = np.asarray(output).reshape(-1)

    #Testing
    probability_sp = [0 for i in range(10)]

    quantized_Test = quantize(qLevel,test_sp)
    
    for c in range(0,10):
        probability_sp[c]=calculate_probability(quantized_Test,mean_qsp[qLevel-2][c],std_qsp[qLevel-2][c])
    
    probability_sp=np.array([np.array([np.array(ji) for ji in xi]) for xi in probability_sp])

    nb = naive_bayes(probability_sp)

    prediction=np.array([np.argmax(nb)])  
    return prediction

trainModel()

loaded_mean = np.loadtxt('Data/MeanStd/mean.csv')
loaded_std = np.loadtxt('Data/MeanStd/std.csv')
mean = loaded_mean.reshape(loaded_mean.shape[0],loaded_mean.shape[1] // 784, 784)
std = loaded_std.reshape(loaded_std.shape[0],loaded_std.shape[1] // 784, 784)

prediction = testImage(cv2.imread('Data/Individual/T3.jpeg'), 2, mean, std)

print('The number in your image is: '+str(prediction[0]))