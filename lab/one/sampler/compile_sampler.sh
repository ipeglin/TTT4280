#! /bin/bash
gcc -Wall -lpthread -o adc_sampler adc_sampler.c -lpigpio -lm
