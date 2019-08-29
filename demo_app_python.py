import onnxruntime as rt
import string
import numpy as np
import os
import pandas as pd
import time

import warnings
warnings.filterwarnings("ignore")

onnx_folder = 'onnx_files'
iterations=1
i_shape = 12006

def preprocess(text):
   tokens = word_tokenize(text)
   # split into lower-case word tokens, in numpy array with shape of (seq, 1)
   words = np.asarray([w.lower() for w in tokens]).reshape(-1, 1)
   # split words into chars, in numpy array with shape of (seq, 1, 1, 16)
   chars = [[c for c in t][:16] for t in tokens]
   chars = [cs+['']*(16-len(cs)) for cs in chars]
   chars = np.asarray(chars).reshape(-1, 1, 1, 16)
   return words, chars

# input
def rt_bind_model(model_path):
    options = rt.SessionOptions()
    options.enable_profiling = True
    options.session_thread_pool_size=16
    options.enable_sequential_execution=True
    options.set_graph_optimization_level(2)
    #options.session_logid = os.path.basename(model_path) + 'model_path/'
    return rt.InferenceSession(model_path, options)

def run_model(sess, model_path, input, output=None):
    inputs = sess.get_inputs()
    ouputs = sess.get_outputs()
    model_input = dict()
    model_output = list()
    
    for i in inputs:
        if('string' in i.type):
            type = str
        elif('int64' in i.type):
            type = np.int64
        elif('int32' in i.type):
            type = int
        elif('float' in i.type):
            type = np.float32
        elif('float32' in i.type):
            type = np.float32
        elif('float64' or 'double' in i.type):
            type = np.float64

        model_input[i.name] = input
    
    for i in ouputs:
       model_output.append(i.name)

    for x in range(iterations):
        t0 = time.perf_counter()
        res = sess.run(None, model_input)
        sec = time.perf_counter() - t0
        prof_file = sess.end_profiling()
        print(model_path, ": done in:" + str(sec*1000) + "ms   fps:" + str(iterations/sec) )
        print(model_path, ":", prof_file)

    if output is not None:
        print("CHEKING OUTPUT ")
        np.testing.assert_allclose(output[0].numpy(), res[0], rtol=1e-02, atol=1e-05)
        for i in range(len(output[1])):
            np.testing.assert_allclose(output[1][i].numpy(), res[1+i], rtol=1e-03, atol=1e-05)
        print("DONE!! ")

def check_model(input, output):
    sess = rt.InferenceSession("gpt2.onnx")
    run_model(sess, "gpt2.onnx", input, output)
