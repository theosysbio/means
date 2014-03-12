#!/usr/bin/env python
"""
simple example script for running and testing notebooks.

Forked from shoyer / ipnbdoctest.py (https://gist.github.com/shoyer/7497853),
who has forked from the original author minrk/ipnbdoctest.py (https://gist.github.com/minrk/2620735)

Usage: `ipnbdoctest.py foo.ipynb [bar.ipynb [...]]`

Each cell is submitted to the kernel, and the outputs are compared with those stored in the notebook.
"""

import os,sys,time
import base64
import hashlib
import traceback
import png
import random
import re
import argparse

from collections import defaultdict
from Queue import Empty
from cStringIO import StringIO

import numpy as np

try:
    from IPython.kernel import KernelManager
except ImportError:
    from IPython.zmq.blockingkernelmanager import BlockingKernelManager as KernelManager

from IPython.nbformat.current import reads, NotebookNode

PNG_DIFF_TOLERANCE = 1e-2

def png_b64_to_ndarray(a64):
    """convert PNG output into a np.ndarray using pypng"""
    pngdata = png.Reader(StringIO(base64.decodestring(a64))).asRGBA8()[2]
    return np.array(list(pngdata))

def print_base64_img(data):
    s = StringIO()
    png.from_array(data, mode='RGBA;8').save(s)
    contents = s.getvalue()
    s.close()
    base64_image = base64.b64encode(contents)

    print '<img src="data:image/png;base64,{0}"/>'.format(base64_image)



def diff_png(a64, b64, generate_diff_images=True):
    """compare the pixels of two PNGs"""
    a_data, b_data = map(png_b64_to_ndarray, (a64, b64))
    if a_data.shape != b_data.shape:
        diff = 255
    else:
        diff = np.mean(np.abs(a_data - b_data))
    diff /= 255.0
    if diff > PNG_DIFF_TOLERANCE:
        digest = hashlib.sha1(a64).digest()
        if generate_diff_images:
            if not os.path.exists('.diffs/'):
                os.mkdir('.diffs/')
            prefix = '.diffs/ipnbdoctest-%s-' % base64.urlsafe_b64encode(digest)[:4]
            print 'Images mismatch'
            print 'Expected:'
            print_base64_img(a_data)
            print 'Actual:'
            print_base64_img(b_data)
            if diff < 1:
                print 'Diff:'
                print_base64_img(255 - np.abs(b_data - a_data))

    return diff


def sanitize(s):
    """sanitize a string for comparison.

    fix universal newlines, strip trailing newlines, and normalize likely random values (memory addresses and UUIDs)
    """
    if not isinstance(s, basestring):
        return s
    # normalize newline:
    s = s.replace('\r\n', '\n')

    # ignore trailing newlines (but not space)
    s = s.rstrip('\n')

    # normalize hex addresses:
    s = re.sub(r'0x[a-f0-9]+', '0xFFFFFFFF', s)

    # normalize UUIDs:
    s = re.sub(r'[a-f0-9]{8}(\-[a-f0-9]{4}){3}\-[a-f0-9]{12}', 'U-U-I-D', s)

    # ignore outputs of %time and %timeit magics:
    s = re.sub(r'(CPU times|Wall time|\d+ loops, best of).+', 'TIMING', s)

    # Ignore the %%cache magic output
    s = re.sub(r'Skipped the cell\'s code and loaded variables problems from file.+', 'CACHING', s)

    return s


def consolidate_outputs(outputs):
    """consolidate outputs into a summary dict (incomplete)"""
    data = defaultdict(list)
    data['stdout'] = ''
    data['stderr'] = ''

    for out in outputs:
        if out.type == 'stream':
            data[out.stream] += out.text
        elif out.type == 'pyerr':
            data['pyerr'] = dict(ename=out.ename, evalue=out.evalue)
        else:
            for key in ('png', 'svg', 'latex', 'html', 'javascript', 'text', 'jpeg',):
                if key in out:
                    data[key].append(out[key])
    return data


def compare_outputs(test, ref,
                    skip_compare=('traceback', 'latex', 'prompt_number'),
                    generate_png_diffs=True):
    for key in ref:
        if key not in test:
            print "missing key: %s != %s" % (test.keys(), ref.keys())
            return False
        elif key == 'png':
            diff = diff_png(test[key], ref[key],
                            generate_diff_images=generate_png_diffs)
            if diff > PNG_DIFF_TOLERANCE:
                print "png mismatch %s" % key
                print "%2.6f%% disagree" % diff
                return False
        elif key not in skip_compare and sanitize(test[key]) != sanitize(ref[key]):
            print "mismatch %s:" % key
            print test[key]
            print '  !=  '
            print ref[key]
            return False
    return True


def run_cell(shell, iopub, cell):
    # print cell.input
    shell.execute(cell.input)
    # wait for finish, maximum 10min
    shell.get_msg(timeout=60*10)
    outs = []

    while True:
        try:
            msg = iopub.get_msg(timeout=0.2)
        except Empty:
            break
        msg_type = msg['msg_type']
        if msg_type in ('status', 'pyin'):
            continue
        elif msg_type == 'clear_output':
            outs = []
            continue

        content = msg['content']
        # print msg_type, content
        out = NotebookNode(output_type=msg_type)

        if msg_type == 'stream':
            out.stream = content['name']
            out.text = content['data']
        elif msg_type in ('display_data', 'pyout'):
            out['metadata'] = content['metadata']
            for mime, data in content['data'].iteritems():
                attr = mime.split('/')[-1].lower()
                # this gets most right, but fix svg+html, plain
                attr = attr.replace('+xml', '').replace('plain', 'text')
                setattr(out, attr, data)
            if msg_type == 'pyout':
                out.prompt_number = content['execution_count']
        elif msg_type == 'pyerr':
            out.ename = content['ename']
            out.evalue = content['evalue']
            out.traceback = content['traceback']
        else:
            print "unhandled iopub msg:", msg_type

        outs.append(out)
    return outs

def collapse_stream_outputs(outputs):
    """
    Collapse the stream outputs returned by ipynb as different buffering
    might change the end result
    """

    stream_outputs = {}
    collapsed_outputs = []
    for output in outputs:
        if output[u'output_type'] == u'stream':
            text = output[u'text']
            try:
                stream_outputs[output[u'stream']].append(text)
            except KeyError:
                stream_outputs[output[u'stream']] = [text]
        else:
            collapsed_outputs.append(output)
    sorted_streams = sorted(stream_outputs.iteritems())
    for stream, stream_texts in sorted_streams:

        collapsed_outputs.append({u"output_type" : u"stream",
                                  u"stream" : stream,
                                  u"text": u''.join(stream_texts)})

    return collapsed_outputs


def test_notebook(nb, generate_png_diffs=True):
    km = KernelManager()
    km.start_kernel(extra_arguments=['--pylab=inline'], stderr=open(os.devnull, 'w'))
    try:
        kc = km.client()
        kc.start_channels()
        iopub = kc.iopub_channel
    except AttributeError:
        # IPython 0.13
        kc = km
        kc.start_channels()
        iopub = kc.sub_channel
    shell = kc.shell_channel

    # run %pylab inline, because some notebooks assume this
    # even though they shouldn't
    shell.execute("pass")
    shell.get_msg()
    while True:
        try:
            iopub.get_msg(timeout=1)
        except Empty:
            break

    successes = 0
    failures = 0
    errors = 0
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type != 'code':
                continue
            try:
                outs = run_cell(shell, iopub, cell)
            except Exception as e:
                print
                print "Failed to run cell, got {0!r}".format(e)
                traceback.print_exc()
                print
                print "The cell is:"
                print '-' * 50
                print cell.input
                print '-' * 50
                errors += 1
                break

            failed = False
            outs = collapse_stream_outputs(outs)
            cell_outputs = collapse_stream_outputs(cell.outputs)
            for out, ref in zip(outs, cell_outputs):
                if not compare_outputs(out, ref, generate_png_diffs=generate_png_diffs):
                    failed = True
            if failed:
                failures += 1
            else:
                successes += 1
            sys.stdout.write('.')

    print
    print "tested notebook %s" % nb.metadata.name
    print "    %3i cells successfully replicated" % successes
    if failures:
        print "    %3i cells mismatched output" % failures
    if errors:
        print "    %3i cells failed to complete" % errors

    if failures or errors:
        print " If image tests failed, but images not displayed, copy and paste the line below to your address bar: "
        print "javascript:pres = document.querySelectorAll('pre'); for (var i = 0; i < pres.length; i++) { pres[i].innerHTML = pres[i].innerHTML.replace(/&lt;img/g, '<img').replace(/\/&gt;/g, '/>') };void 0;"
    kc.stop_channels()
    km.shutdown_kernel()
    del km

    return failures, errors

def arguments_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('notebooks', nargs='+', type=unicode, help='IPython notebooks to test')
    parser.add_argument('--no-png-output', dest='generate_png_images',
                        action="store_false", default=True,
                        help="Do not generate PNG images")
    return parser

if __name__ == '__main__':
    parser = arguments_parser()
    args = parser.parse_args()
    for ipynb in args.notebooks:
        print "testing %s" % ipynb
        with open(ipynb) as f:
            nb = reads(f.read(), 'json')
        test_notebook(nb, generate_png_diffs=args.generate_png_images)
