import time
import requests

f = open("test.txt", "r")
url = 'http://192.168.0.111:8080/v1/events'


def read_text_file(text_name):
    return_list = []
    f = open(text_name, "r")
    for x in f:
      return_list.append(x)
    return return_list

old_len = 0
old_id = 0
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
                myobj = {'QueryImage': '0' , "Timestamp" : timestamp , "CameraID" : cameraid , "VectorIDs" : vectorID }
                x = requests.post(url, data = myobj)
                print(x.status_code, x.reason)
        old_len = len(new_text_file)
    else:
        time.sleep(0.01)
