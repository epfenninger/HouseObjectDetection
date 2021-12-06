import paho.mqtt.publish as publish
MQTT_SERVER = ""  #Write Server IP Address
MQTT_PATH = "ImageProcessor"
port=1883


f=open("testdata/singleperson.jpg", "rb") #3.7kiB in same folder
fileContent = f.read()
byteArr = bytearray(fileContent)
x =  0
publish.single(MQTT_PATH, "Livingroom", hostname=MQTT_SERVER)
while (x < 5):
  publish.single(MQTT_PATH, byteArr, hostname=MQTT_SERVER)
  x = x + 1

publish.single(MQTT_PATH, "fin", hostname=MQTT_SERVER)
