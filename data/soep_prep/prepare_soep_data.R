rm(list = ls())

# load packages
library(haven)
library(dplyr)

# function to double-check variable coding (used below)
double_checker <- function(x1, x2){
  # double-check matching between ids and descriptions
  check1 <- sapply(gregexpr("\\]", x1), 
                   function(z) z[1])
  sel <- !is.na(check1)
  check2 <- rep(NA, length(x1))
  check2[sel] <- as.integer(substr(x1[sel], 2, check1[sel]-1))
  okay <- all.equal(x2, check2)
  if (isTRUE(okay)) print("Double check passed")
}

# data set is available in Stata format from 
# https://www.diw.de/de/diw_01.c.836543.de/soep-uebungsdatensatz.html
# (English version)
dat0 <- read_stata("practice_dataset_eng.dta") 
# Rename, select and transform variables
dat <- dat0 %>% 
  transmute(id = id, survey_year = syear, female = (sex == 1), 
            age = alter, n_persons = anz_pers, n_children = anz_kind, 
            years_educ = bildung,  
            income = einkommenj1 + einkommenj2, 
            employed = NA, employed_id = as.integer(erwerb),
            sector = NA, sector_id = as.integer(branche)) 

# Create new variables that contain text description
# (applies to employment status and sector)

# ids of all employment statuses
unique_employed_ids <- unique(dat$employed_id) %>% na.omit
# names of all statuses
employed_names <- names(attr(dat0$erwerb, "labels"))
# correct typos in original labels
employed_names[8] <- gsub("-1", "1", employed_names[8])
employed_names[9] <- gsub("-2", "2", employed_names[9])

# re-code employed variable to text format
for (jj in unique_employed_ids){
  which_name <- which(grepl(paste0("\\[", jj, "]"), employed_names)) 
  if (length(which_name) > 1){
    stop("")
  } else {
    sel <- which(dat$employed_id == jj)
    dat$employed[sel] <- employed_names[which_name]
  }
}

# check consistency between IDs and text variables (which contain IDs)
double_checker(dat$employed, dat$employed_id)

# ids of all sectors
unique_sector_ids <- unique(dat$sector_id) %>%
  na.omit
# names of all sectors
sector_names <- names(attr(dat0$branche, "labels"))
# recode sector variable to text format
for (jj in unique_sector_ids){
  which_name <- which(grepl(paste0("\\[", jj, "]"), sector_names)) 
  if (length(which_name) > 1){
    stop("")
  } else {
    sel <- which(dat$sector_id == jj)
    dat$sector[sel] <- sector_names[which_name]
  }
}

# check consistency between IDs and text variables (which contain IDs)
double_checker(dat$sector, dat$sector_id)

# look at some randomly selected observations
(dat[sample.int(nrow(dat), size = 20), c("employed_id", "employed",
                                         "sector_id", "sector")])

# verify that IDs < 1 do not occur
mean(dat$employed_id < 1, na.rm = TRUE)
mean(dat$sector_id < 1, na.rm = TRUE)

# save new data set
write.table(dat, "soep_data_Sep9_2024.csv", row.names = FALSE, 
            col.names = TRUE, sep = ",")