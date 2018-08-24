#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 23:33:28 2018

@author: ketanagrawal
"""

#!/usr/bin/env python

# Import required modules
import time
import RPi.GPIO as GPIO

def setup():
    # Declare the GPIO settings
    GPIO.setmode(GPIO.BOARD)
    
    # set up GPIO pins
    GPIO.setup(7, GPIO.OUT) # Connected to PWMA
    GPIO.setup(11, GPIO.OUT) # Connected to AIN2
    GPIO.setup(12, GPIO.OUT) # Connected to AIN1
    GPIO.setup(13, GPIO.OUT) # Connected to STBY
    
def stop():
    GPIO.output(12, GPIO.LOW) # Set AIN1
    GPIO.output(11, GPIO.LOW) # Set AIN2
    GPIO.output(7, GPIO.LOW) # Set PWMA
    GPIO.output(13, GPIO.LOW) # Set STBY

def go_outward(secs):
    # Drive the motor clockwise
    GPIO.output(12, GPIO.HIGH) # Set AIN1
    GPIO.output(11, GPIO.LOW) # Set AIN2
    
    # Set the motor speed
    GPIO.output(7, GPIO.HIGH) # Set PWMA
    
    # Disable STBY (standby)
    GPIO.output(13, GPIO.HIGH)
    
    # Wait 5 seconds
    time.sleep(secs)
    
    # Reset all the GPIO pins by setting them to LOW
    stop()
    
def go_inward(secs):
    # Drive the motor clockwise
    GPIO.output(12, GPIO.LOW) # Set AIN1
    GPIO.output(11, GPIO.HIGH) # Set AIN2
    
    # Set the motor speed
    GPIO.output(7, GPIO.HIGH) # Set PWMA
    
    # Disable STBY (standby)
    GPIO.output(13, GPIO.HIGH)
    
    # Wait 5 seconds
    time.sleep(secs)
    
    # Reset all the GPIO pins by setting them to LOW
    stop()