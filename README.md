# OSS Automatic Bug Assignment - Daniel Artchounin

This repository contains the source code used in the experiments of the Master's Thesis "Tuning of machine learning algorithms for automatic bug assignment" (Daniel Artchounin), conducted on Eclipse JDT and Mozilla Firefox. In this Master's Thesis, a systematic four-step method to find some of the best configurations of several machine learning algorithms intending to solve the automatic bug assignment problem has been introduced. This method has been evaluated on 66 066 bug reports of Ericsson, 24 450 bug reports of Eclipse JDT and 30 358 bug reports of Mozilla Firefox.

## Goal of the repository 

The purpose of this repository is to manage the code base related to
my work on automatic bug assignment (on open- source software
projects).

## Organization

Below, the organization of the repository is described:

  * *eclipse_jdt/*: contains the code base related to the Eclipse JDT
  project;

  * *mozilla_firefox/*: contains the code base related to the Mozilla
  Firefox project.

## Eclipse JDT

### Scrapy

Below, there are some useful command lines to get started with Scrapy:

  * `cd eclipse_jdt/scrap_eclipse_jdt/`: move to the relevant folder;

  * `scrapy crawl eclipse_jdt -o brs.json`: run the Eclipse JDT spider 
  and store the scrapped data into a JSON file.

## Mozilla Firefox

### Scrapy

Below, there are some useful command lines to get started with Scrapy:

  * `cd mozilla_firefox/scrap_mozilla_firefox/`: move to the relevant 
  folder;

  * `scrapy crawl mozilla_firefox -o brs.json`: run the Mozilla 
  Firefox spider and store the scrapped data into a JSON file.
