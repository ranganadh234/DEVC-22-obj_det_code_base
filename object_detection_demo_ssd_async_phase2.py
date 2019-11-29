#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import time
import logging as log
import numpy as np
import io
from openvino.inference_engine import IENetwork, IECore
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))
from demoTools.demoutils import *
import cv2
from openvino.inference_engine import IENetwork, IECore


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    #args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    parser.add_argument('-o', '--output_dir',
                        help='Location to store the results of the processing',
                        default=None,
                        required=True,
                        type=str)
    parser.add_argument('-nireq', '--number_infer_requests',
                        help='Number of parallel inference requests (default is 2).',
                        type=int,
                        required=False,
                        default=2)

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    log.info("Creating Inference Engine...")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    img_info_input_blob = None
    feed_dict = {}
    for blob_name in net.inputs:
        if len(net.inputs[blob_name].shape) == 4:
            input_blob = blob_name
        elif len(net.inputs[blob_name].shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.inputs[blob_name].shape), blob_name))

    assert len(net.outputs) == 1, "Demo supports only single output topologies"

    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=args.number_infer_requests, device_name=args.device)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    if img_info_input_blob:
        feed_dict[img_info_input_blob] = [h, w, 1]

    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
        out_file_name = os.path.splitext(os.path.basename(args.input))[0]
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None
        
    log.info("Starting inference in async mode, {} requests in parallel...".format(args.number_infer_requests))
    job_id = str(os.environ['PBS_JOBID'])
    result_file = open(os.path.join(args.output_dir, 'output_'+job_id+'.txt'), "w")
    infer_file = os.path.join(args.output_dir, 'i_progress_'+job_id+'.txt')

    cap = cv2.VideoCapture(input_stream)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cur_request_id = 0
    next_request_id = 1
    frame_count = 0
    current_inference = 0
    previous_inference = 1 - args.number_infer_requests
    infer_requests = exec_net.requests
    log.info("Starting inference in async mode...")
    #is_async_mode = True
    #render_time = 0
    ret, frame = cap.read()

    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between sync/async modes, press TAB key in the output window")
    
    infer_time_start = time.time()
    while frame_count<video_len:
        ret, next_frame = cap.read()
        initial_w = cap.get(3)
        initial_h = cap.get(4)
        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        #inf_start = time.time()
        if ret:
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            feed_dict[input_blob] = in_frame
            exec_net.start_async(request_id=current_inference, inputs=feed_dict)
        if previous_inference >= 0:
            status=infer_requests[previous_inference].wait()
            #if status is not 0:
            #    raise Exception("Infer request not completed successfully")
            #inf_end = time.time()
            #det_time = inf_end - inf_start

            # Parse detection results of the current request
            res = infer_requests[previous_inference].outputs[out_blob]
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > args.prob_threshold:
                    dims = "{frame_id} {xmin} {ymin} {xmax} {ymax} {class_id} {est} {time} \n".format(frame_id=frame_count, xmin=int(obj[3]                    * initial_w), ymin=int(obj[4] * initial_h), xmax=int(obj[5] * initial_w), ymax=int(obj[6] * initial_h),                                     class_id=int(obj[1]), est=round(obj[2]*100, 1), time='N/A')
                    result_file.write(dims)
            frame_count += 1
                    
        if frame_count % 10 == 0: 
            progressUpdate(infer_file, time.time()-infer_time_start, frame_count+1, video_len+1) 
            # Increment counter for the inference queue and roll them over if necessary 
        current_inference += 1
        if current_inference >= args.number_infer_requests:
            current_inference = 0

        previous_inference += 1
        if previous_inference >= args.number_infer_requests:
            previous_inference = 0

    #cv2.destroyAllWindows()
    # End while loop
    total_time = time.time() - infer_time_start
    with open(os.path.join(args.output_dir, 'stats_{}.txt'.format(job_id)), 'w') as f:
        f.write('{:.3g} \n'.format(total_time))
        f.write('{} \n'.format(frame_count))
    result_file.close()

if __name__ == '__main__':
    sys.exit(main() or 0)
