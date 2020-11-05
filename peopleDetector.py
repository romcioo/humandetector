##[Import]
import cv2 as cv
import pafy
##![import]


##[Declare]

##[Load HAAR classifier]
#haar_file_name = "pedestrian.xml"
haar_file_name = cv.data.haarcascades + "haarcascade_fullbody.xml"
haar = cv.CascadeClassifier()
if not haar.load(haar_file_name): # if the load fails
    print("Cannot load haar classifier")
    exit(1)
##![Load HAAR classifier]

##[Load HOG classifier]
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
##![Load HOG classifier]

##[Constants]
dim = (64,128)
minSize = (100,200)
maxSize = (300,600)
thresholdHaar = 1.4
##![Constants]

##![Declare]


##[Functions]

##[Draw detection]
def draw(img, haar_rect, hog_rect):
    x1,y1,w1,h1 = haar_rect
    x2,y2,w2,h2 = hog_rect
    cv.rectangle(img, (x1,y1), (x1+w1,y1+h1), (0,0,255), 3)
    cv.rectangle(img, (x2,y2), (x2+w2,y2+h2), (0,255,0), 3)
##![Draw detection]

##![Functions]



##[Program]

##[Video Input]
url = "https://www.youtube.com/watch?v=Axt_tvVtz1g"
video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv.VideoCapture(best.url) # cap is the video source
##![Video Input]

##[Loop]
while True:
    # capture frame by frame
    ret, frame = cap.read() # get next frame
    if not ret: # if there is nothing retived
        break # end loop
        
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # get a copy in grayscale of frame
    equalized = cv.equalizeHist(gray) # equalize histogram
    # haar_detect(equalized, frame)
    
    ##[HAAR detection]
    people_haar = haar.detectMultiScale3(equalized, 1.1, 3, 0, minSize,maxSize, True) # get the haar detection
    
    if len(people_haar[0]) > 0: # if there is at least 1 detection
        for i in range(len(people_haar[0])): # go through all detections
            weight = people_haar[2][i]
            if weight > thresholdHaar: # if the detection is over threshold
                rect1 = people_haar[0][i] # get rectangle
                x1,y1,w1,h1 = rect1
                ROI = equalized[y1:y1+h1,x1:x1+w1] # and make ROI
    ##![HAAR detection]
                
                ##[HOG detection]
                people_hog = hog.detectMultiScale(ROI) # get HOG detection
                if len(people_hog[0]) > 0: # if there is at least a detection
                    for j in range(len(people_hog[0])): # go through all detections
                        weight = people_hog[1][j]
                        if weight > thresholdHaar: # and if it's over threshold
                            x2,y2,w2,h2 = people_hog[0][j] # get rectangle
                            rect2 = (x1+x2, y1+y2, w2, h2) # convert to global coordinates
                            draw(frame, rect1, rect2)
                ##![HOG detection]
    
    cv.imshow("Detection", frame) # show image
    k = cv.waitKey(10) & 0xff
    if k == 27 or k == ord('q'): # if q or ESC pressed get out
        break
##![Loop]

##[Finsih Program]
cap.release() # release capture
cv.destroyAllWindows() # close windows
##![Finish Program]

##![Program]