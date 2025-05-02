
# Reads in a downloaded eBird dataset for Maryland (SED and EBD file), as well 
# as land cover and elevation data for Maryland; filters the eBird data
# appropriately; combines the eBird and environmental data; and writes a csv
# with the resultant clean dataset, ready for input into the RF or MLP model.
#
# "Prediction Grid" section creates a grid over Maryland and populates
# it with values for all environmental variables, writing the output to .csv. 
# This file is then ready to be given to the RF or MLP model to make
# predictions and to make a predicted encounter rate map.
# 
# This file also has a function for making a point map of all observed presences
# and absences.

# Adapted from eBird Best Practices: 
# https://ebird.github.io/ebird-best-practices/

library(dplyr)
library(auk)
library(ggplot2)
library(lubridate)
library(gridExtra)
library(readr)
library(sf)
library(luna)
library(tigris)
library(units)
library(exactextractr)
library(landscapemetrics)
library(stringr)
library(terra)
library(tidyr)
library(viridis)

### FILE INPUTS ###

# SED (sampling event data) file: contains all checklist-level data for a given
# region and time period. Downloaded from https://ebird.org/data/download
sed_file <- "ebd_US-MD_daejun_smp_relFeb-2025_sampling.txt"
# EBD (eBird data) file: contains species observations for each checklist in 
# the SED file. Downloaded from https://ebird.org/data/download
ebd_file <- "ebd_US-MD_daejun_smp_relFeb-2025.txt"
# GIS data filepath (gpkg from eBird Best Practices guide)
gpkg_filepath <- "gis-data.gpkg"
# Filepath for MODIS MCD12Q1 v006 land cover raster, 2022, for the Maryland 
# area. Downloaded from https://lpdaac.usgs.gov/products/mcd12q1v006/ in two 
# parts and combined into one .tif file
modis_filepath <- "MODIS_SpatRaster_for_MD_combined.tif"
# .csv containing names for the 15 land cover classes in the MODIS raster
umd_lc_classes_csv <- "mcd12q1_umd_classes.csv"
# Filepath for EarthEnv elevation raster (median, GMTED2010, 1-km res.),
# downloaded from http://www.earthenv.org/topography and cropped to the 
# Mid-Atlantic region
elev_filepath <- "elevation_midatlantic.tif"



### INITIAL PROCESSING ###

# Read in SED and EBD files using auk package
checklists <- read_sampling(sed_file)
observations <- read_ebd(ebd_file)
checklists <- checklists |>
  # Filter to complete checklists only
  # Change year range and month as needed
  filter(all_species_reported, between(year(observation_date), 2014, 2023),
         month(observation_date) == 1)
checklists_sf <- checklists |>
  select(checklist_id, latitude, longitude) |>
  st_as_sf(coords = c("longitude", "latitude"), crs=4326)
study_region_buffered <- read_sf(gpkg_filepath, layer="ne_states") |>
  st_transform(crs=8857) |>
  filter(state_code == "US-MD") |>
  st_buffer(dist=1000) |>
  st_transform(crs=st_crs(checklists_sf))
in_region <- checklists_sf[study_region_buffered, ]
checklists <- semi_join(checklists, in_region, by="checklist_id")
observations <- semi_join(observations, in_region, by="checklist_id")

# Fill in zeros for absences
zf <- auk_zerofill(observations_nc, checklists, collapse=TRUE)

# Convert times to seconds
time_to_decimal <- function(x) {
  x <- hms(x, quiet = TRUE)
  hour(x) + minute(x) / 60 + second(x) / 3600
}

# Add modified time features and average speed (for effort estimation)
zf <- zf |>
  mutate(
    observation_count = as.integer(observation_count),
    effort_distance_km = if_else(protocol_type == "Stationary", 0, effort_distance_km),
    effort_hours = duration_minutes / 60,
    effort_speed_kmph = effort_distance_km / effort_hours,
    hours_of_day = time_to_decimal(time_observations_started),
    year = year(observation_date),
    day_of_year = yday(observation_date))

# Remove outliers in effort variables
zf_filtered <- zf |>
  filter(protocol_type %in% c("Stationary", "Traveling"),
         effort_hours <= 6,
         effort_distance_km <= 10,
         effort_speed_kmph <= 100,
         number_observers <= 10)

# Train-test split for modeling later
zf_filtered$type <- if_else(runif(nrow(zf_filtered)) <= 0.8, "train", "test")

checklists <- zf_filtered |>
  select(checklist_id, observer_id, type,
         observation_count, species_observed,
         state_code, locality_id, latitude, longitude,
         protocol_type, all_species_reported,
         observation_date, year, day_of_year,
         hours_of_day,
         effort_hours, effort_distance_km, effort_speed_kmph,
         number_observers)

# Write to csv
write_csv(checklists, "checklists_zf_md_deju_jan.csv", na = "")



### OBSERVATION PLOTTING ###

ne_land <- read_sf(gpkg_filepath, "ne_land") |>
  st_geometry()
ne_country_lines <- read_sf(gpkg_filepath, "ne_country_lines") |>
  st_geometry()
ne_state_lines <- read_sf(gpkg_filepath, "ne_state_lines") |>
  st_geometry()
study_region <- read_sf(gpkg_filepath, "ne_states") |>
  filter(state_code == "US-MD") |>
  st_geometry()

plot_obs <- function(processed_checklists, month_name) {
  checklists_sf2 <- processed_checklists |>
    st_as_sf(coords = c("longitude", "latitude"), crs = 4326) |>
    select(species_observed)
  par(mar = c(0.25, 0.25, 4, 0.25))
  plot(st_geometry(checklists_sf2),
       main = paste0("Maryland Dark-Eyed Junco Observations\n", month_name, 
                     " 2014-2023"),
       col = NA, border = NA)
  plot(ne_land, col = "#cfcfcf", border = "#888888", lwd = 0.5, add = TRUE)
  plot(study_region, col = "#e6e6e6", border = NA, add = TRUE)
  plot(ne_state_lines, col = "#ffffff", lwd = 0.75, add = TRUE)
  plot(ne_country_lines, col = "#ffffff", lwd = 1.5, add = TRUE)
  plot(filter(checklists_sf2, !species_observed),
       pch = 19, cex = 0.14, col = alpha("#555555", 0.4),
       add = TRUE)
  plot(filter(checklists_sf2, species_observed),
       pch = 19, cex = 0.14, col = alpha("#4daf4a", 0.4),
       add = TRUE)
  legend("bottomright", bty = "n",
         col = c("#555555", "#4daf4a"),
         legend = c("Absent", "Present"),
         pch = 19)
  box()
}

# # For plotting multiple months
# months <- c("January", "February", "March", "April", "May", "June", "July",
#             "August", "September", "October", "November", "December")
# for (mo in 1:12) {
#   chlists <- checklists |> filter(month(observation_date) == mo)
#   plot_obs(chlists, months[mo])
# }



### ADDING ENVIRONMENTAL VARIABLES ###

landcover <- rast(modis_filepath)

# Generate circular neighborhoods for all checklists
checklists_sf3 <- checklists |> 
  # Identify unique locations
  distinct(locality_id, latitude, longitude) |> 
  # Generate 3-km buffers
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326)
buffers <- st_buffer(checklists_sf3, dist = set_units(1.5, "km"))

lsm <- NULL
for (i in seq_len(nrow(buffers))) {
  buffer_i <- st_transform(buffers[i, ], crs = crs(landcover))
  
  # Crop and mask land cover raster so all values outside buffer are missing
  lsm[[i]] <- crop(landcover, buffer_i) |> 
    mask(buffer_i) |> 
    # Calculate landscape metrics
    calculate_lsm(level = "class", metric = c("pland", "ed")) |> 
    # Add variables to uniquely identify each point
    mutate(locality_id = buffer_i$locality_id) |> 
    select(locality_id, class, metric, value)
}
lsm <- bind_rows(lsm)

# Read in land cover class names
lc_classes <- read_csv(umd_lc_classes_csv)

lsm_wide <- lsm |> 
  # Fill missing classes with zeros
  complete(nesting(locality_id),
           class = lc_classes$class,
           metric = c("ed", "pland"),
           fill = list(value = 0)) |> 
  # Bring in more descriptive names
  inner_join(select(lc_classes, class, label), by = "class") |> 
  # Transform from long to wide format
  pivot_wider(values_from = value,
              names_from = c(class, label, metric),
              names_glue = "{metric}_c{str_pad(class, 2, pad = '0')}_{label}",
              names_sort = TRUE) |> 
  arrange(locality_id)

# Elevation raster
elevation <- rast(elev_filepath)

# Mean and standard deviation within each circular buffer
elev_buffer <- exact_extract(elevation, buffers, fun = c("mean", "stdev"),
                             progress = FALSE) |> 
  # Add variables to uniquely identify each point
  mutate(locality_id = buffers$locality_id) |> 
  select(locality_id, 
         elevation_mean = mean,
         elevation_sd = stdev)

# Combine elevation and land cover
env_variables <- inner_join(elev_buffer, lsm_wide,
                            by = "locality_id")

# Attach and expand to checklists
env_variables <- checklists |> 
  select(checklist_id, locality_id) |> 
  inner_join(env_variables, by = "locality_id") |> 
  select(-locality_id) |>
  drop_na()

# Save to csv
write_csv(env_variables, 
          "environmental_vars_checklists_md_jan.csv", 
          na = "")

# Combine zero-filled eBird data  with environmental data
checklists_env <- inner_join(checklists, env_variables, by = "checklist_id")

# Write to csv. This file is ready for input into the RF or MLP model
write_csv(checklists_env, "checklists_env_combined_md_deju_jan.csv")



### PREDICTION GRID ###

# Lambert's azimuthal equal area projection for Maryland
laea_crs <- st_crs("+proj=laea +lat_0=39.1 +lon_0=-76.8")

# Study region: Maryland
study_region_laea <- read_sf(gpkg_filepath, layer = "ne_states") |> 
  filter(state_code == "US-MD") |> 
  st_transform(crs = laea_crs)

# Create a raster template covering the region with 3-km resolution
r <- rast(study_region_laea, res = c(3000, 3000))

# Fill the raster with 1s inside the study region
r <- rasterize(study_region_laea, r, values = 1) |> 
  setNames("study_region")

# Save for later use
r <- writeRaster(r, "prediction_grid_md.tif",
                 overwrite = TRUE,
                 gdal = "COMPRESS=DEFLATE")

# Generate buffers for the prediction grid cell centers
buffers_pg <- as.data.frame(r, cells = TRUE, xy = TRUE) |> 
  select(cell_id = cell, x, y) |> 
  st_as_sf(coords = c("x", "y"), crs = laea_crs, remove = FALSE) |> 
  st_transform(crs = 4326) |> 
  st_buffer(set_units(3, "km"))

# Estimate landscape metrics for each cell in the prediction grid
lsm_pg <- NULL
for (i in seq_len(nrow(buffers_pg))) {
  buffer_i <- st_transform(buffers_pg[i, ], crs = crs(landcover))
  
  # Crop and mask land cover raster so all values outside buffer are missing
  lsm_pg[[i]] <- crop(landcover, buffer_i) |> 
    mask(buffer_i) |> 
    # Calculate landscape metrics
    calculate_lsm(level = "class", metric = c("pland", "ed")) |> 
    # Add variable to uniquely identify each point
    mutate(cell_id = buffer_i$cell_id) |> 
    select(cell_id, class, metric, value)
}
lsm_pg <- bind_rows(lsm_pg)

# Transform to wide format
lsm_wide_pg <- lsm_pg |> 
  # Fill missing classes with zeros
  complete(cell_id,
           class = lc_classes$class,
           metric = c("ed", "pland"),
           fill = list(value = 0)) |> 
  # Bring in more descriptive names
  inner_join(select(lc_classes, class, label), by = "class") |> 
  # Transform from long to wide format
  pivot_wider(values_from = value,
              names_from = c(class, label, metric),
              names_glue = "{metric}_c{str_pad(class, 2, pad = '0')}_{label}",
              names_sort = TRUE,
              values_fill = 0) |> 
  arrange(cell_id)

elev_buffer_pg <- exact_extract(elevation, buffers_pg, 
                                fun = c("mean", "stdev"),
                                progress = FALSE) |> 
  # Add variables to uniquely identify each point
  mutate(cell_id = buffers_pg$cell_id) |> 
  select(cell_id, elevation_mean = mean, elevation_sd = stdev)

# Combine land cover and elevation for prediction grid
env_variables_pg <- inner_join(elev_buffer_pg, lsm_wide_pg, by = "cell_id")

# Attach the xy coordinates of the cell centers
env_variables_pg <- buffers_pg |> 
  st_drop_geometry() |> 
  select(cell_id, x, y) |> 
  inner_join(env_variables_pg, by = "cell_id")

# Save as csv, dropping any rows with missing variables
write_csv(drop_na(env_variables_pg),
          "environmental_vars_pred_grid_md.csv", 
          na = "")

# Example map
savanna_cover <- env_variables_pg |> 
  st_as_sf(coords = c("x", "y"), crs = laea_crs) |> 
  rasterize(r, field = "pland_c9_savanna")
par(mar = c(0.25, 0.25, 2, 0.25))
plot(savanna_cover, 
     axes = FALSE, box = FALSE, col = viridis(10), 
     main = "Savanna (% cover)")