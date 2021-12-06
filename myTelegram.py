import requests
from io import BytesIO

chat_id = 
token=
imgUrl = f"https://api.telegram.org/bot{token}/sendPhoto"
msgUrl = f'https://api.telegram.org/bot{token}/sendMessage'


def sendMsg(message):
  global chat_id
  global token
  global msgUrl
  data = {'chat_id': chat_id, 'text': message}
  requests.post(msgUrl, data).json()

def sendImg(img):
  global imgUrl
  global token
  global chat_id
  files = {}
  files["photo"] = processImage(img)
  requests.get(imgUrl, params={"chat_id": chat_id}, files=files)

def processImage(img):
  buf = BytesIO()

# Save the image as jpeg to the buffer
  img.save(buf, 'jpeg')

# Rewind the buffer's file pointer
  buf.seek(0)

# Read the bytes from the buffer
  image_bytes = buf.read()

# Close the buffer
  buf.close()
  return image_bytes
