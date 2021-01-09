import time
import requests
import cv2
import base64
f = open("test.txt", "r")
url = 'http://172.16.1.111:8080/v1/events'


def read_text_file(text_name):
    return_list = []
    f = open(text_name, "r")
    for x in f:
      return_list.append(x)
    return return_list

def send_image_to_base64(image_name):
    image = cv2.imread("../send_imgs/"+str(image_name)+".jpg")
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text

def post_noti(base64Image,timestamp,cameraid,vectorID):
    myobj = {'QueryImage': base64Image  , "Timestamp" : timestamp , "CameraID" : cameraid , "VectorIDs" : vectorID }
    x = requests.post(url, data = myobj)
    print(x.status_code, x.reason)

def check_recent(vectorID, delta_s_thres):
    global time_temp_id_list
    print(time_temp_id_list)
    if vectorID in time_temp_id_list:
        delta_time = time.time() - time_temp_id_list[vectorID]
        print(delta_time)
        if delta_time >= delta_s_thres:
            print("hihih")
            time_temp_id_list[vectorID] = time.time()
            return True
        else:
            return False
    else:
        time_temp_id_list[vectorID] = time.time()
        return True


old_len = 0
old_id = 0
time_temp_id_list = {}
RECENT_TIME_THRES = 5
while True:
    new_text_file = read_text_file("test.txt")
    if old_len != len(new_text_file):
        for text in new_text_file[old_len:]:
            print(text)
            split_text = text.split(" ")
            if old_id == int(split_text[0]) - 1 :
                print("Post ID",split_text[0])
                old_id = int(split_text[0])
                timestamp = split_text[1] + " " + split_text[2]
                cameraid = split_text[3]
                vectorID = split_text[4]
                if check_recent(vectorID.split(",")[0], RECENT_TIME_THRES ):
                    print("=========== send noti =======================")
                    base64Image = send_image_to_base64(split_text[0])
                    post_noti(base64Image,timestamp,cameraid,vectorID)
        old_len = len(new_text_file)
    else:
        time.sleep(0.01)
