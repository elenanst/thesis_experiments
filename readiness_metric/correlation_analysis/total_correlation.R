# This script computes the total readiness metric for each dataset (based on all HPP-models) and performs correlation analysis.

# --- set up ---
rm(list=ls())
setwd("thesis_experiments/readiness_metric")

# --- load data ---
distances_k <- read.csv("data/distances_k.csv",
                       header = TRUE, sep=",", stringsAsFactors=FALSE)
distances_cp <- read.csv("data/distances_cp.csv",
                        header = TRUE, sep=",", stringsAsFactors=FALSE)

distances_size <- read.csv("data/distances_size.csv",
                        header = TRUE, sep=",", stringsAsFactors=FALSE)

distances_decay <- read.csv("data/distances_decay.csv",
                        header = TRUE, sep=",", stringsAsFactors=FALSE)

distances_sigma <- read.csv("data/distances_sigma.csv",
                        header = TRUE, sep=",", stringsAsFactors=FALSE)

distances_C <- read.csv("data/distances_C.csv",
                        header = TRUE, sep=",", stringsAsFactors=FALSE)

# --- compute combined readiness metric ---
total_distances <- cbind(distances_k$X, distances_cp$X, distances_size$X, distances_decay$X,
                         distances_sigma$X, distances_C$X)
mean_distances  <- apply(total_distances,1, mean)
dataset         <- data.frame(X= mean_distances)

ggplot(dataset, aes(x = X)) + geom_histogram(aes(y = ..density..)) + geom_density(adjust = 1,kernel = "gaussian")
density_estimate <- density(mean_distances,adjust=1,kernel = "gaussian")
first_q          <- 1.9054 
third_q          <- 6.3174
inter_range      <- third_q - first_q 
low_bound        <- first_q - 1.5 *inter_range
upper_bound      <- third_q + 1.5 *inter_range