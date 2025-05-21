import cv2 
import os
from flask import Flask, render_template, Response, request, send_file, redirect
import pickle
import numpy as np
import sqlite3

app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2() 
os.makedirs(os.path.join(app.instance_path, 'upload'), exist_ok=True)

@app.route('/')
@app.route('/index') 
def index():
    
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')


@app.route('/image') 
def image_page():
    return render_template('image.html')

@app.route('/upload', methods=('GET','POST'))
def upload():
    if request.method == 'POST':
        f = request.files['file']
        # f.save(os.path.join(app.instance_path, 'upload', 'file2.jpg'))
        f.save(r'C:\Users\hp\OneDrive\Desktop\major project code\heart rate\upload\file2')
        gen2()
    
    return redirect('/image')

@app.route('/hr')
def hr():
    return render_template('hr.html')

def __del__():
    cam.stop()


def gen():   
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    

    cam=cv2.VideoCapture(0)   
    while(cam.isOpened()):    

        ret, frame=cam.read()

        if not ret:
            frame= cv2.VideoCapture(1) 
            continue
        if ret:

            image= cv2.resize(frame, (0, 0), None, 1, 1)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:

                image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = image[y:y+h, x:x+w] 
        key = cv2.waitKey(20)
        if key == 27:
            break


def gen2(): 
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    
    frame = cv2.imread(r'C:\Users\hp\OneDrive\Desktop\major project code\heart rate\upload\file2')
        
    image= cv2.resize(frame, (0, 0), None, 1, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
     
    frame = cv2.imencode('.jpg', image)[1].tobytes()
    return (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gen3():
    cam=cv2.VideoCapture(0)   
    while(cam.isOpened()):    

        ret, frame=cam.read()

        if not ret:
            frame= cv2.VideoCapture(1) 
            continue
        if ret:
            
            
            img1= cv2.resize(frame, (0, 0), None, 1, 1)
            img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img3=cv2.mean(img2)
            img4= np.delete(img3,-1)
            
            model= pickle.load(open('notebook/model_hr.pkl','rb'))
            op=model.predict([img4])
            out=op.astype(np.int64)
            
            out1=str(out)
            
            cv2.putText(img1, out1 , (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
            
        
        frame = cv2.imencode('.jpg', img1)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
        key = cv2.waitKey(20)
        if key == 27:
            break

    

        

@app.route('/video_feed') 
def video_feed():
    return Response(gen3(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image_feed') 
def image_feed():    
    return Response(gen2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hr_feed') 
def hr_feed():
    return Response(gen3(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == "__main__":
	app.run(debug = True)