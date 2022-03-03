# Code for the Sequential Deconfounder from: "Sequential Deconfounding for Causal Inference
with Unobserved Confounders"

# Abstract: 
Using observational data to estimate the effect of a treatment is a powerful tool for decision-making when randomized experiments are infeasible or costly. However, observational data often yields biased estimates of treatment effects, since treatment assignment can be confounded by unobserved variables. A remedy is offered by deconfounding methods that adjust for such unobserved confounders. In this paper, we develop the Sequential Deconfounder, a method that enables estimating individualized treatment effects over time in presence of unobserved confounders. This is the first deconfounding method that can be used in a general sequential setting (i.e., with one or more treatments assigned at each timestep). The Sequential Deconfounder uses a novel Gaussian process latent variable model to infer substitutes for the unobserved confounders, which are then used in conjunction with an outcome model to estimate treatment effects over time. We prove that using our method yields unbiased estimates of individualized treatment responses over time. Using simulated and real medical data, we demonstrate the efficacy of our method in deconfounding the estimation of treatment responses over time. 

Paper available: https://arxiv.org/abs/2104.09323
