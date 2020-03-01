# Version control maintained using MRAN

# Github packages can be versioned by checking out a specific commit rather than
#  the latest

## Alternative
# Aother solution is to use dev tools as follows
# require(devtools)
# install_version("dplyr", version = "0.8.4")
# Caveat - probably just installs the latest version of dependencies, so 
# would need to install each manually, possible using sessionInfo() to help

repos = "mran.microsoft.com/snapshot/2020-02-22"
options("repos" = repos)
install.packages("dplyr")
install.packages("ggplot2")
install.packages("rmarkdown")
install.packages("DT")
