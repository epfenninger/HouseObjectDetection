import detector
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
MQTT_SERVER = "192.168.10.13"
MQTT_PATH = "ImageProcessor"
from PIL import Image
import io
import json
import datetime
import myTelegram
import requests
from time import sleep
import sys

recent = ''
#Dataset that you care about
dataset = ['person','bicycle','truck','motorcycle','car','cat','dog','bear','horse']
imgList = []
home = True

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MQTT_PATH)
    # The callback for when a PUBLISH message is received from the server.


def on_message(client, userdata, msg):
    global recent
    global imgList
    global home
    end = False
    # more callbacks, etc
    # Create a file with write byte permission

    #---Set Camera that is talkine
    try:
      message = msg.payload.decode("utf-8")
      print("Trigger from " + message)
    except:
      print(msg.payload)
      message = 'error'

    if (message == 'Away'):
      home = False
    elif (message == 'Home'):
      home = True
    #Parameters are CameraName , number of frames to grab, space between
    #Frames (in seconds), and T/F on splitting image up, and accuracy
    elif (message == 'Front'):
      detect(message,3,.75,False,.4)
    elif (message == 'Driveway'):
      detect(message,3,.75,True,.7)
    elif (message == 'Livingroom'):
      detect(message,1,.75,False,.7)
    elif (message == 'Garage'):
      detect(message,3,.75,True,.5)
    elif (message == 'Basement'):
      detect(message,3,.75,False,.7)
    elif (message == 'Washer'):
      detect(message,3,.75,False,.7)
    elif (message == 'Kitchen'):
      detect(message,1,.75,False,.7)
    elif (message == 'SideCam'):
      detect(message,3,.75,False,.4)


def detect(name,num,spacing,multi,acc):
  imgList = []
  acc = float(acc)


  for x in range(num):
    try:
      r = requests.get('http://192.168.10.6:3300/image/' + name) 
      img = Image.open(io.BytesIO(r.content)).convert('RGB')
      imgList.append(img.copy())
      sleep(float(spacing))
    except:
      print("Couldn't open image")

  try:
    if (multi):
      objects = detector.multiDetect(imgList,acc)
    else:
      objects = detector.singleDetect(imgList,acc)
  except:
    e = sys.exc_info()[0]
    myTelegram.sendMsg("Failed to process image " + str(e))
  
  try:
    handleObjects(name,objects,imgList[int(num/2)])
  except:
    print("Imagelist had nothin in it")

def handleObjects(name,objects,img):
  list = []
  filteredList = []
  send = False
  message = ''
  global dataset
  global home
  print("Handling pictures from " + name + "with pre-filtered list of length " + str(len(objects)))
  
  for object in objects:
      list.append(object.get('Name'))
  print(list)    
  for i in list:
    if i not in filteredList:
      filteredList.append(i)

  for item in filteredList:
    if item in dataset:
      message = message + str(item) + " and a "
      send = True
    
  message = message[:len(message)-7]
  print("Filtered List length " + str(len(filteredList)))
  if not send:
    print("nothing important seen")
  else:    
    message = "The " + name + " Camera spotted a " + message
    print(message)
    if ((name == 'Livingroom' or name == 'Kitchen') and home): 
      print("we're home, don't wanna see this shit")
    else:
      myTelegram.sendMsg(message)
      myTelegram.sendImg(img)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_SERVER, 1883, 60)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.



myTelegram.sendMsg("Image Server Up!")
try:
  client.loop_forever()
except:
  e = sys.exc_info()[0]
  myTelegram.sendMsg("Image Server down " + str(e))
