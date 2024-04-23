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
ratio = -154.0

id_product = 1
list_label = []
list_weight = []
count = 0
final_weight = 0
taken = 0

b = 'Monaco'
l = 'Lays'

processed_flags = {}

if sys.platform == 'linux' and not os.environ.get('DISPLAY'):
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
        print('Calibration starts')
        try:
            GPIO.setmode(GPIO.BCM)
            hx = HX711(dout_pin=20, pd_sck_pin=21)
            err = hx.zero()
            if err:
                raise ValueError('Tare is unsuccessful.')
            hx.set_scale_ratio(ratio)
            c_value = 1
        except (KeyboardInterrupt, SystemExit):
            print('Bye :)')
        print('Calibrate ends')  
    else:
        GPIO.setmode(GPIO.BCM)
        time.sleep(1)
        try:
            weight = int(hx.get_weight_mean(20))
            print(weight, 'g')
            return weight
        except (KeyboardInterrupt, SystemExit):
            print('Bye :)')

def post(label, price, final_rate, taken):
    global id_product
    global list_label
    global list_weight
    global count
    global final_weight
    taken
    
    url = "https://sagarai.onrender.com/product"
    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    data_dict = {"id": id_product, "name": label, "price": price, "units": "units", "taken": taken, "payable": final_rate}
    data = json.dumps(data_dict)
    
    try:
        resp = requests.post(url, headers=headers, data=data)
        resp.raise_for_status()  # Raise error for bad response status
        
        print("API response:", resp.status_code)
        print("API response content:", resp.text)  # Print API response content for debugging
        
        id_product += 1  
        time.sleep(1)
        list_label = []
        list_weight = []
        count = 0
        final_weight = 0
        taken = 0
    except requests.exceptions.RequestException as e:
        print("Error connecting to API:", e)

def list_com(label, final_weight):
    global count
    global taken
    
    if final_weight is not None and final_weight > 2:    
        list_weight.append(final_weight)
        if count > 1 and list_weight[-1] > list_weight[-2]:
            taken += 1
    else:
        return  # Skip further processing if final_weight is None or below the threshold
        
    list_label.append(label)
    count += 1
    print('count is', count)
    time.sleep(1)
    
    if count > 1 and list_label and list_weight:  # Check if lists are not empty
        if list_label[-1] != list_label[-2]:
            print("New Item detected")
            print("Final weight is", list_weight[-1])
            rate(list_weight[-2], list_label[-2], taken)

def rate(final_weight, label, taken):
    print("Calculating rate")
    if label == b:
        print("Calculating rate of", label)
        final_rate_b = final_weight * 0.02
        price = 20
        post(label, price, final_rate_b, taken)
    elif label == l:
        print("Calculating rate of", label)
        final_rate_l = 1 
        price = 1
        post(label, price, final_rate_l, taken)

def main(argv):
    global final_weight
    if not flag:
        find_weight()

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

    print('MODEL:', modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
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
                print("Camera %s (%s x %s) in port %s selected." % (backendName, h, w, videoCaptureDeviceId))
                camera.release()
            else:
                raise Exception("Couldn't initialize selected camera.")

            next_frame = 0

            for res, img in runner.classifier(videoCaptureDeviceId):
                if next_frame > now():
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
                            if label == b:
                                print('Monaco detected')
                            elif label == l:
                                print('Lays detected')
                            print('%s: %.2f\t' % (label, bb['value']), end='')
                        print('', flush=True)

                if show_camera:
                    cv2.imshow('edgeimpulse', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) == ord('q'):
                        break

                next_frame = now() + 100
        finally:
            if runner:
                runner.stop()

if __name__ == "__main__":
    main(sys.argv[1:])
