#! /bin/bash

stilts tpipe ifmt=csv DES_prefinal.csv \
             omode=out ofmt=fits out=DES_final.fits

stilts tpipe ifmt=csv PS_prefinal.csv \
             omode=out ofmt=fits out=PS_final.fits

stilts tpipe ifmt=csv SM_prefinal.csv \
             omode=out ofmt=fits out=SM_final.fits
