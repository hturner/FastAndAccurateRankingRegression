library(PlackettLuce)
# read in sushi complete orderings of 10 items in .soc file downloaded from
# https://www.kamishima.net/sushi/
# N.B. does *not* match that on Preflib 2021-01-07
sushi <- read.soc("../data/sushi_dectet_data/ED-00014-00000001.soc")

# fit standard Plackett-Luce model
R <- as.rankings(sushi[,-1], "ordering")
mod <- PlackettLuce(R, npseudo = 0, weights = sushi$Freq)
round(coef(mod, log = FALSE), 3)
