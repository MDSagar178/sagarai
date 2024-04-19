#!/usr/bin/env python

import cv2
import os
import sys
import getopt
import signal
import time
from edge_impulse_linux.image import ImageImpulseRunner

import RPi.GPIO as GPIO 
from hx711 import HX711

import requests
import json
from requests.structures import CaseInsensitiveDict

runner = None
show_camera = True

c_value = 0
flag = 0
ratio = -1363.992

id_product = 1
list_label = []
list_weight = []
count = 0
final_weight = 0
taken = 0

a = 'Apple'
b = 'Monaco'
l = 'Lays'
c = 'Coke'

processed_flags = {}  # Flagging mechanism to avoid multiple posts

if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False

def now():
    return round(time.time() * 1000)

def get_webcams():
    port_ids = []
    for port in range(5):
        print("Looking for a camera in port %s:" % port)
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            ret = camera.read()[0]
            if ret:
                backendName = camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) found in port %s " % (backendName, h, w, port))
                port_ids.append(port)
            camera.release()
    return port_ids

def sigint_handler(sig, frame):
    print('Interrupted')
    if runner:
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def help():
    print('python classify.py <path_to_model.eim> <Camera port ID, only required when more than 1 camera is present>')

def find_weight():
    global c_value
    global hx
    if c_value == 0:
        print('Debug: Calibration starts')
        try:
            GPIO.setmode(GPIO.BCM)
            hx = HX711(dout_pin=20, pd_sck_pin=21)
            err = hx.zero()
            if err:
                raise ValueError('Tare is unsuccessful.')
            hx.set_scale_ratio(ratio)
            c_value = 1
        except (KeyboardInterrupt, SystemExit):
            print('Debug: Bye :)')
        print('Debug: Calibrate ends')  
    else:
        GPIO.setmode(GPIO.BCM)
        time.sleep(1)
        try:
            weight = int(hx.get_weight_mean(20))
            print("Debug: Weight:", weight)
            return weight
        except (KeyboardInterrupt, SystemExit):
            print('Debug: Bye :)')

def post(label, price, final_rate, taken):
    global id_product
    global list_label
    global list_weight
    global count
    global final_weight
    global taken
    
    url = "https://selfcheckout.onrender.com/product"
    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    data_dict = {"id": id_product, "name": label, "price": price, "units": "units", "taken": taken, "payable": final_rate}
    data = json.dumps(data_dict)
    resp = requests.post(url, headers=headers, data=data)
    print("Debug: Response status code:", resp.status_code)
    id_product += 1  
    time.sleep(1)
    list_label = []
    list_weight = []
    count = 0
    final_weight = 0
    taken = 0
    processed_flags.clear()  # Reset processed_flags dictionary after posting

def list_com(label, final_weight):
    global count
    global taken
    
    print('Debug: Detected label:', label)  # Debug print to check the detected label
    
    if final_weight > 2:    
        list_weight.append(final_weight)
        if count > 1 and list_weight[-1] > list_weight[-2]:
            taken += 1
    list_label.append(label)
    count += 1
    print('Debug: Count is', count)
    time.sleep(1)
    
    if count > 1 and list_label and list_weight:  # Check if lists are not empty
        if list_label[-1] != list_label[-2]:
            print("Debug: New Item detected")
            print("Debug: Final weight is", list_weight[-1])
            rate(list_weight[-2], list_label[-2], taken)

def rate(final_weight, label, taken):
    print("Debug: Calculating rate")
    if label not in processed_flags:
        print("Debug: Label not processed before")
        if label == a:
            print("Debug: Calculating rate of", label)
            final_rate_a = final_weight * 0.01  
            price = 10     
            post(label, price, final_rate_a, taken)
        elif label == b:
            print("Debug: Calculating rate of", label)
            final_rate_b = final_weight * 0.02
            price = 20
            post(label, price, final_rate_b, taken)
        elif label == l:
            print("Debug: Calculating rate of", label)
            final_rate_l = 1 
            price = 1
            post(label, price, final_rate_l, taken)
        else:
            print("Debug: Calculating rate of", label)
            final_rate_c = 2
            price = 2
            post(label, price, final_rate_c)
        processed_flags[label] = True  # Mark the label as processed
    else:
        print("Debug: Label already processed before")

def main(argv):
    global flag
    global final_weight
    global processed_flags
    if flag == 0 :
        find_weight()
        flag = 1
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) == 0:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('Debug: MODEL:', modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Debug: Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']
            if len(args) >= 2:
                videoCaptureDeviceId = int(args[1])
            else:
                port_ids = get_webcams()
                if len(port_ids) == 0:
                    raise Exception('Cannot find any webcams')
                if len(args) <= 1 and len(port_ids) > 1:
                    raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use to this script")
                videoCaptureDeviceId = int(port_ids[0])

            camera = cv2.VideoCapture(videoCaptureDeviceId)
            ret = camera.read()[0]
            if ret:
                backendName = camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Debug: Camera %s (%s x %s) in port %s selected." % (backendName, h, w, videoCaptureDeviceId))
                camera.release()
            else:
                raise Exception("Couldn't initialize selected camera.")

            next_frame = 0  # limit to ~10 fps here

            for res, img in runner.classifier(videoCaptureDeviceId):
                if (next_frame > now()):
                    time.sleep((next_frame - now()) / 1000)

                if "classification" in res["result"].keys():
                    print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')

                elif "bounding_boxes" in res["result"].keys():
                    print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                    for bb in res["result"]["bounding_boxes"]:
                        print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                        img = cv2.rectangle(img, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)
                        if bb['value'] > 0.9 :
                            label = bb['label']
                            final_weight = find_weight()
                            list_com(label, final_weight)
                            if label == a:
                                print('Debug: Apple detected')       
                            elif label == b:
                                print('Debug: Monaco detected')
                            elif label == l:
                                print('Debug: Lays detected')
                            else:
                                print('Debug: Coke detected')
                                print('%s: %.2f\t' % (label, bb['value']), end='')
                        print('', flush=True)

                if (show_camera):
                    cv2.imshow('edgeimpulse', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) == ord('q'):
                        break

                next_frame = now() + 100
        finally:
            if runner:
                runner.stop()

if __name__ == "__main__":
    main(sys.argv[1:])
