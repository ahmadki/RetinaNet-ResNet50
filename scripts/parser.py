#!/usr/bin/env python3

import os
import re
import argparse
from string import Template
from pathlib import Path

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


SAVABLE_OBJECTS = ['train_df', 'val_df', 'loss_lr', 'map_lr']
DF_SAVE_FORMATS = ['pickle', 'csv']
PLOT_SAVE_FORMATS = ['html', 'jpg', 'png']

def parse_args():
    parser = argparse.ArgumentParser(description='Parse model logs and save them to different formats')

    # Model arguments
    parser.add_argument('log', type=str,
                        help='Input log file')
    parser.add_argument('--save', type=str.lower, nargs='+', choices=SAVABLE_OBJECTS, default=SAVABLE_OBJECTS,
                        help='The objects to save')
    parser.add_argument('--df-format', type=str.lower, choices=DF_SAVE_FORMATS, default='csv',
                        help='The format to save the DataFrames in')
    parser.add_argument('--plot-format', type=str.lower, choices=PLOT_SAVE_FORMATS, default='html',
                        help='The format to save the plots in')
    parser.add_argument('--prefix', type=str, default=None,
                        help='Prefix to the saved objects. Defaults to the log file name followed by a _')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output dir, default to same folder as the log')
    args = parser.parse_args()

    # Process default args
    args.prefix = args.prefix or (Path(args.log).stem + '_')
    args.output_dir = args.output_dir or Path(args.log).parent

    return args


def parse_log(fname):
    train_re = r"Epoch: \[(\d*)\]  \[\s*?(\d*)\/(\d*)\] .* lr: (\d*.\d*)  loss: (\d*.\d*)"
    val_re = f"Average Precision  \(AP\) \@\[ IoU=0\.50:0\.95 \| area=   all \| maxDets=100 \] = (\d*.\d*)"

    train_epoch = []
    train_epoch_iter = []
    train_iters_per_epoch = []
    train_lr = []
    train_loss = []

    val_map = []
    val_epoch = []
    val_iter = []

    with open(fname,"r") as fp:
        file_text = fp.read()

        train_matches = re.finditer(train_re, file_text, re.MULTILINE)
        for match in train_matches:
            groups = match.groups()
            train_epoch.append(int(groups[0]))
            train_epoch_iter.append(int(groups[1]))
            train_iters_per_epoch.append(int(groups[2]))
            train_lr.append(float(groups[3]))
            train_loss.append(float(groups[4]))

        iters_per_epoch = train_iters_per_epoch[-1]

        val_matches = re.finditer(val_re, file_text, re.MULTILINE)
        for i, match in enumerate(val_matches, start=1):
            groups = match.groups()
            val_map.append(float(groups[0]))
            val_epoch.append(i)
            val_iter.append(i*iters_per_epoch)

    train_df = pd.DataFrame(
        {
            'epoch': train_epoch,
            'iter': train_epoch_iter,
            'iters_per_epoch': train_iters_per_epoch,
            'loss': train_loss,
            'lr': train_lr,
        }
    )
    val_df = pd.DataFrame(
        {
            'map': val_map,
            'epoch': val_epoch,
            'iter': val_iter,
        }
    )

    return train_df, val_df


def get_loss_lr_fig(train_df):
    fractional_epoch = train_df["epoch"].values + train_df["iter"].values/train_df["iters_per_epoch"].values
    loss_lr_fig = make_subplots(specs=[[{"secondary_y": True}]])    
    loss_lr_fig.add_trace(
        go.Scatter(x=fractional_epoch, y=train_df["loss"], name="Training loss"),
        secondary_y=False,
    )
    loss_lr_fig.add_trace(
        go.Scatter(x=train_df["epoch"], y=train_df["lr"], name="lr"),
        secondary_y=True,
    )
    return loss_lr_fig


def get_map_lr_fig(train_df, val_df):
    map_lr_fig = make_subplots(specs=[[{"secondary_y": True}]])
    map_lr_fig.add_trace(
        go.Scatter(x=val_df["epoch"], y=val_df["map"], name="mAP"),
        secondary_y=False,
    )
    map_lr_fig.add_trace(
        go.Scatter(x=train_df["epoch"], y=train_df["lr"], name="lr"),
        secondary_y=True,
    )
    return map_lr_fig


def save_df(df, fname):
    ext = Path(fname).suffix
    if ext == '.csv':
        df.to_csv(fname)
    elif ext == '.pickle':
        df.to_pickle(fname)
    else:
        assert False, f'Unknown df save format {ext}'


def save_plotly(fig, fname):
    ext = Path(fname).suffix
    if ext in ['.jpg', '.png']:
        fig.write_image(fname)
    elif ext == '.html':
        fig.write_html(fname)
    else:
        assert False, f'Unknown plot save format {ext}'


def print_summary(train_df, val_df):
    map_idxmax = val_df["map"].idxmax()
    print(f"Best mAP= {val_df['map'][map_idxmax]} At epoch= {val_df['epoch'][map_idxmax]}")


if __name__ == "__main__":
    args = parse_args()
    train_df, val_df = parse_log(args.log)
    loss_lr_fig = get_loss_lr_fig(train_df=train_df)
    map_lr_fig = get_map_lr_fig(train_df=train_df, val_df=val_df)

    print_summary(train_df=train_df, val_df=val_df)

    fname_tempalte = Template(os.path.join(args.output_dir, args.prefix+'$fname.$ext'))
    if 'train_df' in args.save:
        save_df(train_df, fname_tempalte.substitute(fname='train_df', ext=args.df_format))
    if 'val_df' in args.save:
        save_df(val_df, fname_tempalte.substitute(fname='val_df', ext=args.df_format))
    if 'loss_lr' in args.save:
        save_plotly(loss_lr_fig, fname_tempalte.substitute(fname='loss_lr', ext=args.plot_format))
    if 'map_lr' in args.save:
        save_plotly(map_lr_fig, fname_tempalte.substitute(fname='map_lr', ext=args.plot_format))

