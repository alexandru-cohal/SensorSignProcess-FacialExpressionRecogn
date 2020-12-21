# -*- coding: utf-8 -*-
# vim:set et ts=4 sw=4:
#
## Copyright (C) 2012 Ozan Çağlayan <ocaglayan@gsu.edu.tr>
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program; if not, write to the Free Software
## Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

"""\
This module provides the EPOC class for accessing Emotiv EPOC
EEG headsets.
"""

import os

from Crypto.Cipher import AES

import usb.core
import usb.util

import numpy as np

import utils

import sys

import matplotlib.pyplot as plt

import time

import pickle

#Additional libraries needed by the classifier
from sklearn.neighbors import KNeighborsClassifier


"""Constants Declaration"""
N_CHANNELS = 14
WINDOW_SIZE = 25
OVERLAPPING_SIZE = 15
N_SAMPLES_IN_RECORDING = 100

label_dictionary = {"neutral":0, "bl":1, "br":2, "bb":3, "fm":4, "om":5, "eb":6, "m2l":7, "m2r":8, "n":9, "s":10}
label_dictionary_inverted = {0:"neutral", 1:"bl", 2:"br", 3:"bb", 4:"fm", 5:"om", 6:"eb", 7:"m2l", 8:"m2r", 9:"n", 10:"s"}

class EPOCError(Exception):
    """Base class for exceptions in this module."""
    pass


class EPOCTurnedOffError(EPOCError):
    """Exception raised when Emotiv EPOC is not turned on."""
    pass


class EPOCDeviceNodeNotFoundError(EPOCError):
    """Exception raised when /dev/emotiv_epoc is missing."""
    pass


class EPOCUSBError(EPOCError):
    """Exception raised when error occurs during I/O operations."""
    pass


class EPOCNotPluggedError(EPOCError):
    """Exception raised when EPOC dongle cannot be detected."""
    pass


class EPOCPermissionError(EPOCError):
    """Exception raised when EPOC dongle cannot be opened for I/O."""
    pass


class EPOC(object):
    """Class for accessing Emotiv EPOC headset devices."""

    # Device descriptions for USB
    INTERFACE_DESC = "Emotiv RAW DATA"
    MANUFACTURER_PREFIX = "Emotiv Systems"

    # Channel names
    channels = ["F3", "FC5", "AF3", "F7", "T7", "P7", "O1",
                "O2", "P8",  "T8",  "F8", "AF4", "FC6", "F4"]

    # Sampling rate: 128Hz (Internal: 2048Hz)
    sampling_rate = 128

    # Vertical resolution interval (0.51uV)
    vres = 0.51

    # Battery levels
    # github.com/openyou/emokit/blob/master/doc/emotiv_protocol.asciidoc
    battery_levels = {247: 99, 246: 97, 245: 93, 244: 89, 243: 85,
                      242: 82, 241: 77, 240: 72, 239: 66, 238: 62,
                      237: 55, 236: 46, 235: 32, 234: 20, 233: 12,
                      232: 6, 231: 4, 230: 3, 229: 2, 228: 1,
                      227: 1, 226: 1,
                      }
    # 100% for bit values between 248-255
    battery_levels.update(dict([(k, 100) for k in range(248, 256)]))
    # 0% for bit values between 128-225
    battery_levels.update(dict([(k, 0) for k in range(128, 226)]))

    # Define a contact quality ordering
    #   github.com/openyou/emokit/blob/master/doc/emotiv_protocol.asciidoc

    # For counter values between 0-15
    cq_order = ["F3", "FC5", "AF3", "F7", "T7",  "P7",  "O1",
                "O2", "P8",  "T8",  "F8", "AF4", "FC6", "F4",
                "F8", "AF4"]

    # 16-63 is currently unknown
    cq_order.extend([None, ] * 48)

    # Now the first 16 values repeat once more and ends with 'FC6'
    cq_order.extend(cq_order[:16])
    cq_order.append("FC6")

    # Finally pattern 77-80 repeats until 127
    cq_order.extend(cq_order[-4:] * 12)

    # emokit-style bit indexes to use with utils.get_level()
    bit_indexes = {
        'F3': [10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7],
        'FC5': [28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9],
        'AF3': [46, 47, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27],
        'F7': [48, 49, 50, 51, 52, 53, 54, 55, 40, 41, 42, 43, 44, 45],
        'T7': [66, 67, 68, 69, 70, 71, 56, 57, 58, 59, 60, 61, 62, 63],
        'P7': [84, 85, 86, 87, 72, 73, 74, 75, 76, 77, 78, 79, 64, 65],
        'O1': [102, 103, 88, 89, 90, 91, 92, 93, 94, 95, 80, 81, 82, 83],
        'O2': [140, 141, 142, 143, 128, 129, 130, 131, 132, 133, 134, 135, 120, 121],
        'P8': [158, 159, 144, 145, 146, 147, 148, 149, 150, 151, 136, 137, 138, 139],
        'T8': [160, 161, 162, 163, 164, 165, 166, 167, 152, 153, 154, 155, 156, 157],
        'F8': [178, 179, 180, 181, 182, 183, 168, 169, 170, 171, 172, 173, 174, 175],
        'AF4': [196, 197, 198, 199, 184, 185, 186, 187, 188, 189, 190, 191, 176, 177],
        'FC6': [214, 215, 200, 201, 202, 203, 204, 205, 206, 207, 192, 193, 194, 195],
        'F4': [216, 217, 218, 219, 220, 221, 222, 223, 208, 209, 210, 211, 212, 213],
        'QU': [99,100,101,102,103,104,105,106,107,108,109,110,111,112],
    }

    def __init__(self, method="libusb", serial_number=None, enable_gyro=True):
        self.vendor_id = None
        self.product_id = None
        self.decryption = None
        self.decryption_key = None
        self.headset_on = False
        self.headset_type = "research"  # Default to research, detect later
        self.enable_gyro = enable_gyro
        self.battery = 0
        self.counter = 0
        self.gyroX = 0
        self.gyroY = 0

        # Access method can be direct/libusb/dummy (Default: libusb)
        # If dummy is given the class behaves as a random signal generator
        self.method = method

        # One may like to specify the dongle with its serial
        self.serial_number = serial_number

        # libusb device and endpoint
        self.device = None
        self.endpoint = None

        # By default acquire from all channels
        self.channel_mask = self.channels

        # Dict for storing contact qualities
        self.quality = {
            "F3": 0, "FC5": 0, "AF3": 0, "F7": 0,
            "T7": 0, "P7": 0, "O1": 0, "O2": 0,
            "P8": 0, "T8": 0, "F8": 0, "AF4": 0,
            "FC6": 0, "F4": 0,
        }

        # Update __dict__ with convenience attributes for channels
        self.__dict__.update(dict((v, k) for k, v in enumerate(self.channels)))

        # Enumerate the bus to find EPOC devices
        self.enumerate()

    def _is_epoc(self, device):
        """Custom match function for libusb."""
        try:
            manu = usb.util.get_string(device, device.iManufacturer)
        except usb.core.USBError, usb_exception:
            # If the udev rule is installed, we shouldn't get an exception
            # for Emotiv device.
            print usb_exception
            return False
        else:
            if manu and manu.startswith(self.MANUFACTURER_PREFIX):
                print manu
                return True
                # FIXME: This may not be necessary at all Found a dongle, check for interface class 3
                for interf in device.get_active_configuration():
                    if_str = usb.util.get_string(device, interf.iInterface)
                    if if_str == self.INTERFACE_DESC:
                        return True

    def set_channel_mask(self, channel_mask):
        """Set channels from which to acquire."""
        self.channel_mask = channel_mask

    def enumerate(self):
        """Traverse through USB bus and enumerate EPOC devices."""
        if self.method == "dummy":
            self.endpoint = open("/dev/urandom")
            self.get_sample = self.__get_sample_dummy
            return

        devices = usb.core.find(find_all=True, custom_match=self._is_epoc)

        if not devices:
            raise EPOCNotPluggedError("Emotiv EPOC not found.")

        for dev in devices:
            serial = usb.util.get_string(dev, dev.iSerialNumber)
            if self.serial_number and self.serial_number != serial:
                # If a special S/N is given, look for it.
                continue

            # Record some attributes
            self.serial_number = serial
            self.vendor_id = "%x" % dev.idVendor
            self.product_id = "%x" % dev.idProduct

            if self.product_id == "0001":
                print "Consumer headset detected."
                self.headset_type = "consumer"

            if self.method == "libusb":
                # Last interface is the one we need
                for interface in dev.get_active_configuration():
                    if dev.is_kernel_driver_active(interface.bInterfaceNumber):
                        # Detach kernel drivers and claim through libusb
                        dev.detach_kernel_driver(interface.bInterfaceNumber)
                        usb.util.claim_interface(dev, interface.bInterfaceNumber)

                self.device = dev
                self.endpoint = usb.util.find_descriptor(
                    interface, bEndpointAddress=usb.ENDPOINT_IN | 2)
            elif self.method == "direct":
                if os.path.exists("/dev/emotiv_epoc"):
                    self.endpoint = open("/dev/emotiv_epoc")
                else:
                    raise EPOCDeviceNodeNotFoundError(
                        "/dev/emotiv_epoc doesn't exist.")

            # Return the first Emotiv headset by default
            break

        self.setup_encryption()
        # Attempt to see whether the headset is turned on
        try:
            self.endpoint.read(32, 100)
        except usb.USBError as ue:
            if ue.errno == 110:
                self.headset_on = False
                print "Setup is OK but make sure that headset is turned on."
        else:
            self.headset_on = True

    def setup_encryption(self):
        """Generate the encryption key and setup Crypto module.
        The key is based on the serial number of the device and the
        information whether it is a research or consumer device.
        """
        if self.headset_type == "research":
            self.decryption_key = ''.join([self.serial_number[15], '\x00',
                                           self.serial_number[14], '\x54',
                                           self.serial_number[13], '\x10',
                                           self.serial_number[12], '\x42',
                                           self.serial_number[15], '\x00',
                                           self.serial_number[14], '\x48',
                                           self.serial_number[13], '\x00',
                                           self.serial_number[12], '\x50'])
        elif self.headset_type == "consumer":
            self.decryption_key = ''.join([self.serial_number[15], '\x00',
                                           self.serial_number[14], '\x48',
                                           self.serial_number[13], '\x00',
                                           self.serial_number[12], '\x54',
                                           self.serial_number[15], '\x10',
                                           self.serial_number[14], '\x42',
                                           self.serial_number[13], '\x00',
                                           self.serial_number[12], '\x50'])

        self._cipher = AES.new(self.decryption_key)

    def set_external_decryption(self):
        """Use another process for concurrent decryption."""
        self.decryption = Process(target=decryptionProcess,
                                  args=[self._cipher,
                                        self.input_queue,
                                        self.output_queue, False])
        self.decryption.daemon = True
        self.decryption.start()

    def __get_sample_dummy(self):
        """Read random dummy samples."""
        raw_data = self.endpoint.read(32)
        return [utils.get_level(raw_data, self.bit_indexes[n]) for n in self.channel_mask]

    def get_sample(self):
        """Returns an array of EEG samples."""
        try:
            raw_data = self._cipher.decrypt(self.endpoint.read(32))
            # Parse counter
            ctr = ord(raw_data[0])
            # Update gyro's if requested
            if self.enable_gyro:
                self.gyroX = ((ord(raw_data[29]) << 4) | (ord(raw_data[31]) >> 4))
                self.gyroY = ((ord(raw_data[30]) << 4) | (ord(raw_data[31]) & 0x0F))
            if ctr < 128:
                self.counter = ctr
                # Contact qualities
                if self.cq_order[ctr]:
                    self.quality[self.cq_order[ctr]] = utils.get_level(raw_data, self.bit_indexes["QU"]) / 540.0
                # Finally EEG data
                return [0.51 * utils.get_level(raw_data, self.bit_indexes[n]) for n in self.channel_mask]
            else:
                # Set a synthetic counter for this special packet: 128
                self.counter = 128
                # Parse battery level
                self.battery = self.battery_levels[ctr]
                return []
        except usb.USBError as usb_exception:
            if usb_exception.errno == 110:
                self.headset_on = False
                raise EPOCTurnedOffError(
                        "Make sure that headset is turned on")
            else:
                raise EPOCUSBError("USB I/O error with errno = %d" %
                        usb_exception.errno)

    def acquire_data(self, duration):
        """Acquire data from the EPOC headset."""

        total_samples = duration * self.sampling_rate
        _buffer = np.ndarray((total_samples, len(self.channel_mask) + 1),
                dtype=np.uint16)
        ctr = 0
        while ctr < total_samples:
            # Fetch new data
            data = self.get_sample()
            if data:
                # Prepend sequence numbers
                _buffer[ctr] = np.insert(np.array(data), 0, self.counter)
                ctr += 1

        return _buffer

    def acquire_data_fast(self, duration, stop_callback=None, stop_callback_param=None):
        """A more optimized method to acquire data from the EPOC headset without calling get_sample()."""

        def get_level(raw_data, bits):
            """Returns signal level from raw_data frame."""
            level = 0
            for i in range(13, -1, -1):
                level <<= 1
                b, o = (bits[i] / 8) + 1, bits[i] % 8
                level |= (ord(raw_data[b]) >> o) & 1
            # Return level in uV (microVolts)
            return level

        bit_indexes = [self.bit_indexes[n] for n in self.channel_mask]
        # Packet idx to keep track of losses
        idx = []
        total_samples = duration * self.sampling_rate

        # Pre-allocated array
        _buffer = np.ndarray((total_samples, len(self.channel_mask)), dtype=np.float64)

        # Acquire in one read, this should be more robust against drops
        raw_data = self._cipher.decrypt(self.endpoint.read(32 * (total_samples + duration + 1), timeout=(duration+1)*1000))

        if stop_callback and stop_callback_param:
            stop_callback(stop_callback_param)

        # Split data back into 32-byte chunks, skipping 1st packet
        split_data = [raw_data[i:i + 32] for i in range(32, len(raw_data), 32)]

        # Loop ctr
        c = 0
        for block in split_data:
            if c == total_samples:
                break
            # Parse counter
            ctr = ord(block[0])
            # Skip battery
            if ctr < 128:
                idx.append(ctr)
                _buffer[c] = [0.51 * get_level(block, bi) for bi in bit_indexes]
                c += 1
                # Update qualities as well
                if self.cq_order[ctr] is not None:
                    self.quality[self.cq_order[ctr]] = utils.get_level(block, self.bit_indexes["QU"]) / 540.0
            else:
                # Parse battery level
                self.battery = self.battery_levels[ctr]

        return idx, _buffer

    def get_quality(self, electrode):
        "Return contact quality for the specified electrode."""
        return self.quality.get(electrode, None)

    def disconnect(self):
        """Release the claimed interface."""
        if self.method == "libusb":
            for interf in self.device.get_active_configuration():
                usb.util.release_interface(
                    self.device, interf.bInterfaceNumber)
        else:
            self.endpoint.close()

def init_KNN_classifier(k, features, labels):
    print " Initializing KNN classifier"

    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(features, labels)

    print "KNN classifier initialized"
    print "Initial accuracy after initializazion is ", classifier.score(features, labels)

    return classifier

def store_ratio_windowMean_neutralMean(data, mean_channels_neutral, classes_used_list, channels_used_list, n_recordings):
    # Calculate the ratios for the specified channels and classes
    # Create the features and labels matrices for training and testing

    features = []
    labels = []
	
    for file_index in range(n_recordings):		
        for class_label in classes_used_list:
            class_index = label_dictionary[class_label]
            for window_start_index in range(0, N_SAMPLES_IN_RECORDING, WINDOW_SIZE - OVERLAPPING_SIZE):
                if window_start_index + WINDOW_SIZE <= N_SAMPLES_IN_RECORDING:

                    recording_window = data[class_index][file_index][window_start_index : window_start_index + WINDOW_SIZE]
                    features_list = []

                    for channel_index in channels_used_list:
                        channel_ratio_current = features_extractor(recording_window[:,channel_index], mean_channels_neutral[0,channel_index])
                        features_list.append(channel_ratio_current)

                    features.append(np.asarray(features_list))
                    labels.append(class_index)

	features = np.asarray(features)
	labels = np.asarray(labels)

	return features, labels

def neutral_calibration(e):
    MAX_N_SAMPLES = 10 * WINDOW_SIZE
    neutral_mean_array = np.zeros((1, N_CHANNELS))
    n_samples = 0

    print "Calibration: Prepare to hold a stable facial expression in..."
    print "3"
    time.sleep(1)
    print "2"
    time.sleep(1)
    print "1"
    time.sleep(1)
    print "Now:"

    while(1):
        try:
            data = e.get_sample()
            if data:
                n_samples = n_samples + 1
                neutral_mean_array = neutral_mean_array + data
                
                if(n_samples == MAX_N_SAMPLES):
                    neutral_mean_array = neutral_mean_array / MAX_N_SAMPLES
                    print "Calibration done!"
                    print neutral_mean_array.shape
                    return neutral_mean_array

        except EPOCTurnedOffError, ete:
            print ete
        except KeyboardInterrupt, ki:
            e.disconnect()
            return 0

def data_train_acquisition(e, n_recording, class_label):
    MAX_N_SAMPLES = 100
    data_array = np.zeros((MAX_N_SAMPLES, N_CHANNELS))
    n_samples = 0
    
    data_array_list = []
    
    for index_recording in range(n_recording):
        print "Recording number " + str(index_recording) + " for class [" + class_label + "]"
        print "Prepare to perform the action in..." 
        print "3"
        time.sleep(1)
        print "2"
        time.sleep(1)
        print "1"
        time.sleep(1)
        print "Action!\n\n"

        while(1):
            try:
                data = e.get_sample()
                if data:
                    data_array[n_samples, :] = data
                    n_samples = n_samples + 1
                    
                    if(n_samples == MAX_N_SAMPLES):
                        data_array_list.append(data_array)
                        n_samples = 0
                        break

            except EPOCTurnedOffError, ete:
                print ete
            except KeyboardInterrupt, ki:
                e.disconnect()
                return 0

    return data_array_list
    
def data_train_generation(e, classes_used_list, n_recording):       
    data = [[] for x in range(len(label_dictionary))]    
    for class_label in classes_used_list:    
        data[label_dictionary[class_label]] = data_train_acquisition(e, n_recording, class_label)

    with open('outfile', 'wb') as fp:
        pickle.dump(data, fp)  

    return data  


def features_extractor(windows_data, neutral_mean):
     return np.average(windows_data) / neutral_mean
            
def classify_KNN_classifier(classifier, windows_data):
    result = classifier.predict(windows_data)
    print label_dictionary_inverted[result[0]] 

def main():
    print "----------------------------------------------------------------------------------"
    print "------------------- Real-time Facial Expressions Classifier ----------------------"
    print "----------------------------------------------------------------------------------"
    e = EPOC()

    NON_OVERLAPPING_SIZE = WINDOW_SIZE - OVERLAPPING_SIZE 

    k_neighbors = 11
    sample_index = 0 

    

    new_data = np.zeros((NON_OVERLAPPING_SIZE, N_CHANNELS))
    windows_data = np.zeros((WINDOW_SIZE, N_CHANNELS))

    neutral_mean = neutral_calibration(e)
    
    print "Generating data"
    classes_used_list = ["neutral", "bl", "bb", "br", "om"]
    channels_used_list = range(N_CHANNELS)
    n_recordings = 10
    data = data_train_generation(e, classes_used_list, n_recordings)
    features, labels = store_ratio_windowMean_neutralMean(data, neutral_mean, classes_used_list, channels_used_list, n_recordings)
    classifier = init_KNN_classifier(k_neighbors, features, labels)
    list_ratios = []    
    while(1):
        try:
            data = e.get_sample()
            if data:
                new_data[sample_index,:] = data            
                sample_index = sample_index + 1

                if sample_index == NON_OVERLAPPING_SIZE: 
                    windows_data = np.vstack((windows_data[NON_OVERLAPPING_SIZE : WINDOW_SIZE, :], new_data))
                    for channel_index in range(N_CHANNELS):
                        list_ratios.append(features_extractor(windows_data[:, channel_index], neutral_mean[0, channel_index]))
                    np.asarray(list_ratios)
                    windows_data = np.reshape(np.asarray(list_ratios), (1,14))
                    classify_KNN_classifier(classifier, windows_data)
                    list_ratios = []
                    sample_index = 0
		
        except EPOCTurnedOffError, ete:
            print ete
        except KeyboardInterrupt, ki:
            e.disconnect()
            return 0
			

           
if __name__ == "__main__":
    sys.exit(main())
