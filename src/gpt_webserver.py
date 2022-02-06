import json
import os
import numpy as np
import tensorflow as tf
from urllib.parse import parse_qs
from http.server import BaseHTTPRequestHandler, HTTPServer
from cgi import parse_header, parse_multipart
import multiprocessing as mp
from ctypes import c_char_p
import fire

import model, sample, encoder

hostName = "localhost"
serverPort = 8080

def interact_model(
    result,
    pretext,
    model_name='1558M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=64,
    temperature=1,
    top_k=40,
    top_p=1,
    models_dir='models',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """

    if not pretext:
        result.value = -1
        return
    if len(pretext) > 500:
        model_name='774M'
    if len(pretext) > 4000:
        result.value = -1
        return

    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        result.value = -1
        return #raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        # # Actual Generation of Code # #
        raw_text = pretext
        while not raw_text:
            result.value = -1
            return
        context_tokens = enc.encode(raw_text)
        generated = 0
        
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                result.value = text
                return

class Server(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(405)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def parse_POST(self):
        ctype, pdict = parse_header(self.headers['content-type'])
        if ctype == 'multipart/form-data':
            pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
            postvars = parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers['content-length'])
            print(2)
            postvars = parse_qs(
                    self.rfile.read(length), 
                    keep_blank_values=1)
        else:
            postvars = {}
        return postvars

    def do_POST(self):
        postvars = self.parse_POST()
        try:
            s = str( postvars['string'][0])
        except ValueError:
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            return

        if not s:
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            return
        elif str.isspace(str.strip(s)):
            self.send_response(406)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            return

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        manager = mp.Manager()
        result = manager.Value(c_char_p, '')

        ai = mp.Process(target=interact_model, args=(result, s))
        ai.start()
        print('awaiting result')
        ai.join()
        print('result returned')

        output = result.value

        self.wfile.write(bytes(output, 'utf-8'))

def startServer (
    host='localhost',
    port=8080
): 
    webServer = HTTPServer((host, port), Server)
    print("Server started http://%s:%s" % (host, port))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")


if __name__ == '__main__':
    fire.Fire(startServer)