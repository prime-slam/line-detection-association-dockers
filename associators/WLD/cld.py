#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO implement network protocol
TODO varInt for MESSAGE_ID?
TODO more Elements
TODO varInt for COUNT?

 MSG             = MAGIC , PAYLOAD_SIZE , MESSAGE ;
 MESSAGE         = MESSAGE_ID , ':' , MESSAGE_CONTENT ;
 MESSAGE_CONTENT = TYPE_ID , MESSAGE_ELEMENT ;
 MESSAGE_ELEMENT = number
                  | string
                  | LIST
                  | MAT
                  | CUTOUTS ;
 LIST            = COUNT , TYPE_ID , MESSAGE_ELEMENT ;
 CUTOUTS         = WIDTH , HEIGHT , COUNT , MAT , { MAT } ;
 MAT             = WIDTH, HEIGHT, CHANNELS, MAT_TYPE , uint8 , { uint8 }
 MAT_TYPE        = "f"   // float/double/long double
                  | "i"  // int/char/short/long
                  | "u"  // unsigned int/char/short/long
                  | "b"  // boolean

 MAGIC           = uint32 ;
 PAYLOAD_SIZE    = uint32
 MESSAGE_ID      = uint8 , { uint8 } ;
 TYPE_ID         = uint8 ;
 WIDTH           = uint32 ;
 HEIGHT          = uint32 ;
 CHANNELS        = uint32 ;
 COUNT           = uint32 ;

- msg header : <messageID:uint?>:
"""
import os
import io # noqa

import cnn.modules.term as term

from cnn.modules.config import Conf
from cnn import config

import misc.logger as logger
_L = logger.getLogger("CNNmain")


def argslice(s):
    a = [int(e) if e.strip() else None for e in s.split(":")]
    return slice(*a)


def setup_argparser(conf=None):
    """Setup the ArgumentParser for command line args.

    Arguments:
    config - dict-like object with default values"""
    import argparse
    import textwrap
    if not conf:
        conf = Conf()
    common = argparse.ArgumentParser(add_help=False)

    common.add_argument("--conf",
                        type=argparse.FileType("r"),
                        help=("read settings from this conf file "
                              "(additional command line arguments "
                              "take precedence)"))
    common.add_argument("--out",
                        type=str, default="",
                        help=("subname to append to output folder"
                              " (eg. train.50.<out>)"))
    common.add_argument("--in",
                        type=str, default="", dest="input_folder",
                        help=("input folder with serialized data "
                              "to use instead of a network connection"))
    common.add_argument("--in_valid",
                        type=str, default=None, dest="validation_input_folder",
                        help=("input folder with serialized data"
                              "to use for validation. If not set no validation will be done"))
    common.add_argument("--npz_in",
                        type=str, default="", dest="npz_input_folder",
                        help=("input folder with serialized data in npz files "
                              "to use instead of a network connection"))
    common.add_argument("--npz_in_valid",
                        type=str, default=None, dest="npz_validation_input_folder",
                        help=("input folder with serialized data in npz files "
                              "to use for validation. If not set no validation will be done"))
    common.add_argument("--image_in",
                        type=str, default=None, dest="imagereader_image",
                        help=("single image "
                              "to use instead of a network connection"))
    common.add_argument("--keylines_in",
                        type=str, default=None, dest="imagereader_keylines",
                        help=("precomputed keylines "
                              "to use instead of LSDDetector"))
    common.add_argument("--keylines_use_second_image", "--keylines_use_right",
                        action="store_true", dest="imagereader_keylines_use_right",
                        help="use right keylines")
    common.add_argument("--cutout_width",
                        type=int, default=27, dest="imagereader_cutout_width",
                        help="width of cutouts")
    common.add_argument("--cutout_height",
                        type=int, default=100, dest="imagereader_cutout_height",
                        help="height of cutouts")
    common.add_argument("--range",
                        type=argslice, default=[slice(None)],
                        nargs="*",
                        help=textwrap.dedent("""(only if --in <folder> is given!)\
                    range of files in form of <start>:<end>:<step>.
                    `:` means 'all' (same as not specifying anything)
                    `10:` means 'all files from 10 upwards'
                    `10:100` means 'all files from 10 to 100 (exclusive)'
                    `10:100:5` means 'every 5th file from 10 to 100'
                    `10:100:-1` means 'every file from 100 to 10 (backwards)
                        """))
    common.add_argument("--rand",
                        action="store_true", dest="random_data",
                        help=("(only if --in <folder> is given!) "
                              "read data in random order"))
    common.add_argument("--batch_size",
                        type=int, default=128,
                        help="batch size for learning")
    common.add_argument("--ip",
                        type=str, default="0.0.0.0",
                        help="ip address of server to contact for Elements")
    common.add_argument("--port",
                        type=int, default=9645,
                        help="port of Server")
    common.add_argument("--gpu",
                        type=str, default="0",
                        help="comma separated list of GPU(s) to use.")
    common.add_argument("--debug",
                        action="store_true",
                        help="turn on debugging (single-threaded mode)")
    # common.add_argument("--debug_cutout",
    #                     action="store_true",
    #                     help=("turn on debugging for cutouts "
    #                           "(extra cutout with drawn in Linesegment)"))
    common.add_argument("--log_ask",
                        action="store_true",
                        help=("ask what to do with duplicate log dirs. "
                              "(Default is \"keep\")"))
    common.add_argument("--log_devices",
                        action="store_true",
                        help=("log device placement"))

    common.add_argument("--min_len",
                        type=int, default=5,
                        help=("min length of lines"))

    common.add_argument("--fixed_length",
                        action="store_true",
                        help=("min length of lines"))

    common.add_argument("-n", "--batch_num",
                       type=int, default=1,
                       help="how many batches should be tested")

    common.add_argument("--disable_bn",
                        action="store_true",
                        help="disable all batch normalization layers in resnet")

    common.add_argument("--const_lr",
                       type=float, default=None,
                       help="if set learning rate will be set constant to given value")

    parser = argparse.ArgumentParser(parents=[common])
    # set defaults of this parser to already parsed values
    cargs = common.parse_known_args()[0]
    parser.set_defaults(**dict(cargs._get_kwargs()))

    # setup conf object if arg was given
    if (cargs.conf):
        conf.read_file(cargs.conf)

    subparsers = parser.add_subparsers(dest="cmd")

    # ==== TRAINER
    trainp = subparsers.add_parser("train", conflict_handler="resolve",
                                   parents=[common])
    trainp.add_argument("--load",
                        type=str,
                        help="load model")

    # config options
    trainp.add_argument("--max_epoch",
                        type=int,
                        default=99999,  # this value comes from tensorpack..
                        help="max epoch")

    trainp.add_argument("--steps_per_epoch",
                        type=int, default=None,
                        help="steps per epoch")

    # resnet and densenet
    trainp.add_argument("-d", "--depth",
                        type=int, default=50,
                        help="resnet depth (one of [18, 34, 50, 101, 152])")

    trainp.add_argument("--growth_rate",
                        type=int, default=12,
                        help="growth rate, only used with dense net")

    trainp.add_argument("--use_dense_net",
                        action="store_true",
                        help="use_dense_net")

    trainp.add_argument("--dense_net_BC",
                        action="store_true",
                        help="dense net with bottleneck and compression")

    trainp.add_argument("--theta",
                        type=float, default=1.0,
                        help=("feature reduction in dense net transition "
                              "layer in [0, 1]"))
    trainp.add_argument("--prob",
                        type=float, default=0.,
                        help=("propability of brightness change "
                              "of line cutouts"))
    trainp.add_argument("--brightness_d",
                        type=float, default=1.5,
                        help=("delta of brightness change "
                              "of line cutouts"))

    # ==== TESTER
    testp = subparsers.add_parser("test", conflict_handler="resolve",
                                  parents=[trainp])

    testp.add_argument("model",
                       type=argparse.FileType("r"), nargs="*",
                       help="model to test")
    testp.add_argument("--load_frozen",
                       action="store_true",
                       help="load model from frozen file.")
    # testp.add_argument("--imgs",
    #                    type=str, nargs="*",
    #                    help="run the model for these images")

    testp.add_argument("--tp_imgs",
                       action="store_true",
                       help="generate image output for tensorboard")
    testp.add_argument("--tp_proj",
                       action="store_true",
                       help="generate projection output for tensorboard")
    testp.add_argument("--save_dists",
                       action="store_true",
                       help=("save distance matrices to files for"
                             " external ROC calculation"))
    testp.add_argument("--return_results",
                       action="store_true",
                       help=("answer server with cnn results"))
    testp.add_argument("--save_results",
                       action="store_true",
                       help=("save cnn results into a npz file next to image"))
    testp.add_argument("--time",
                       action="store_true",
                       help="time generation of descriptors and print at the end")

    # ==== SAVER
    savep = subparsers.add_parser("save", conflict_handler="resolve",
                                  parents=[trainp])

    savep.add_argument("model",
                       type=argparse.FileType("r"), nargs="+",
                       help="model to save")
    savep.add_argument("--to",
                       type=str, default="",
                       help=("target folder to save to"))
    savep.add_argument("--compact",
                       action="store_true",
                       help="prune and freeze model to files")

    # set defaults for every subparser
    for subp in subparsers._name_parser_map.values():
        for action in subp._actions:
            if action.dest in conf.keys() and action.option_strings:
                # this is an optional argument
                subp.set_defaults(**{action.dest: conf[action.dest]})

    return parser


if __name__ == '__main__':
    """
    This can be configured with a configuration file
    or per command line.
    On multiple CLI-Args the latter overwrites the former.

    argparse_defaults < config < cli_args
    """
    global config

    parser = setup_argparser(config)
    args = parser.parse_args()

    if args.input_folder and not os.path.exists(args.input_folder):
        _L.critical("There is no such folder {}".format(args.input_folder))
        exit(1)

    if args.validation_input_folder and not os.path.exists(args.validation_input_folder):
        _L.critical("There is no such validation folder {}".format(args.validation_input_folder))
        exit(1)

    if args.npz_input_folder and not os.path.exists(args.npz_input_folder):
        _L.critical("There is no such npz folder {}".format(args.npz_input_folder))
        exit(1)

    if args.npz_validation_input_folder and not os.path.exists(args.npz_validation_input_folder):
        _L.critical("There is no such npz folder {}".format(args.npz_validation_input_folder))
        exit(1)

    # extract file name from FileType
    for special_arg in ["model"]:
        arg = getattr(args, special_arg, None)
        if arg:
            setattr(args, special_arg, [el.name for el in arg])

    # couldn't find the model to load
    if (args.cmd == "train" and args.load and not os.path.exists(args.load)):
        _L.info("Couldn't find model `{}` to load. Should we ignore it? [Y/n] "
                .format(args.load))
        term.hide_cursor()
        key = term.getch()
        term.show_cursor()
        if key in ["y", "Y", "\r"]:
            _L.info("Ignoring model")
            args.load = ""

    # TODO load settings.conf from model directory on testing if present
    # update global config object
    config.set(**dict(args._get_kwargs()))

    # finally run the CNN
    from cnn import main
    main.run()
