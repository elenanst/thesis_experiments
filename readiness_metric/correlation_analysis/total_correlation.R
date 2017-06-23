# This script computes the total readiness metric for each dataset and performs correlation analysis.

## set up
rm(list=ls())
setwd("thesis_experiments/readiness_metric")
library(tikzDevice)

## load data
distances_k <- read.csv("distances_k.csv",
                       header = TRUE, sep=",", stringsAsFactors=FALSE)
distances_k <- distances_k$distances

distances_cp <- read.csv("distances_cp.csv",
                        header = TRUE, sep=",", stringsAsFactors=FALSE)
distances_cp <- distances_cp$distances

distances_size <- read.csv("distances_size.csv",
                        header = TRUE, sep=",", stringsAsFactors=FALSE)
distances_size <- distances_size$distances

distances_decay <- read.csv("distances_decay.csv",
                        header = TRUE, sep=",", stringsAsFactors=FALSE)
distances_decay <- distances_decay$distances

distances_sigma <- read.csv("distances_sigma.csv",
                        header = TRUE, sep=",", stringsAsFactors=FALSE)
distances_sigma <- distances_sigma$distances

distances_C <- read.csv("distances_C.csv",
                        header = TRUE, sep=",", stringsAsFactors=FALSE)
distances_C <- distances_C$distances

total_distances <- cbind(distances_k, distances_cp, distances_size, distances_decay,
                         distances_sigma, distances_C)

mean_distances  <- apply(total_distances,1, mean)
dataset         <- data.frame(X= mean_distances)

tikz('distances_hist_total.tex', standAlone = TRUE, width=7, height=5)
ggplot(dataset, aes(x = X)) + geom_histogram(aes(y = ..density..)) + geom_density(adjust = 0.45,kernel = "gaussian")
dev.off()
density_estimate <- density(mean_distances,adjust=0.45,kernel = "gaussian")
first_q          <- 1.91984 
third_q          <- 5.59785
inter_range      <- third_q - first_q 
low_bound        <- first_q - 1.5 *inter_range
upper_bound      <- third_q+ 1.5 *inter_range