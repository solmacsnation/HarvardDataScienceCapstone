

library(kknn)
library(ranger)
library(tidyverse)
library(caret)
options(digits=5)

# ============================================================
# 0) Provided code for loading full dataset
# ============================================================

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# https://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- "ml-10M100K.zip"
if (!file.exists(dl)) {
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
}

ratings_file <- "ml-10M100K/ratings.dat"
if (!file.exists(ratings_file)) {
  unzip(dl, files = ratings_file)
}

movies_file <- "ml-10M100K/movies.dat"
if (!file.exists(movies_file)) {
  unzip(dl, files = movies_file)
}

ratings <- as.data.frame(
  str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
  stringsAsFactors = FALSE
)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")

ratings <- ratings %>%
  mutate(
    userId = as.integer(userId),
    movieId = as.integer(movieId),
    rating = as.numeric(rating),
    timestamp = as.integer(timestamp)
  )

movies <- as.data.frame(
  str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
  stringsAsFactors = FALSE
)
colnames(movies) <- c("movieId", "title", "genres")

movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind = "Rounding")  # for R 3.6+
# set.seed(1)                          # for R 3.5 or earlier

test_index <- createDataPartition(
  y = movielens$rating,
  times = 1,
  p = 0.1,
  list = FALSE
)

edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# ============================================================
# 1) Bootstrapping
# ============================================================

set.seed(1)

sample_n_rows <- 200000
boot_iters <- 25

dat <- edx %>%
  sample_n(sample_n_rows)

# ============================================================
# 2) Create sample training and test sets
# ============================================================

set.seed(2)
test_index <- createDataPartition(dat$rating, p = 0.10, list = FALSE)

train_dat <- dat[-test_index, ]
holdout_dat <- dat[test_index, ]

train_raw <- train_dat
holdout_raw <- holdout_dat
final_holdout_test_dat <- final_holdout_test
test_raw <- final_holdout_test_dat

# 3) Global mean and SD

global_mean <- mean(train_raw$rating, na.rm = TRUE)
global_sd   <- sd(train_raw$rating, na.rm = TRUE)

# PCA on genres
  
build_genre_matrix <- function(df) {
  
  df %>%
    
    mutate(row_id = row_number()) %>%
    
    separate_rows(genres, sep = "\\|") %>%
    
    mutate(value = 1) %>%
    
    pivot_wider(
      id_cols = row_id,
      names_from = genres,
      values_from = value,
      values_fill = 0
    ) %>%
    
    arrange(row_id) %>%
    
    select(-row_id)
  
}
  
  train_genre_raw    <- build_genre_matrix(train_raw)
  holdout_genre_raw  <- build_genre_matrix(holdout_raw)
  test_genre_raw     <- build_genre_matrix(test_raw)
  
  genre_cols <- colnames(train_genre_raw)
  
  # add missing columns
  missing_holdout <- setdiff(genre_cols, colnames(holdout_genre_raw))
  for (col in missing_holdout) holdout_genre_raw[[col]] <- 0
  
  missing_test <- setdiff(genre_cols, colnames(test_genre_raw))
  for (col in missing_test) test_genre_raw[[col]] <- 0
  
  # enforce identical ordering
  train_genre_raw   <- train_genre_raw[, genre_cols]
  holdout_genre_raw <- holdout_genre_raw[, genre_cols]
  test_genre_raw    <- test_genre_raw[, genre_cols]
  
  # remove useless genre column everywhere
  train_genre_raw   <- train_genre_raw %>% select(-any_of("(no genres listed)"))
  holdout_genre_raw <- holdout_genre_raw %>% select(-any_of("(no genres listed)"))
  test_genre_raw    <- test_genre_raw %>% select(-any_of("(no genres listed)"))
  
  train_genre_mat   <- as.matrix(train_genre_raw)
  holdout_genre_mat <- as.matrix(holdout_genre_raw)
  test_genre_mat    <- as.matrix(test_genre_raw)
  
  genre_pca <- prcomp(train_genre_mat, center = TRUE, scale. = TRUE)
  
  explained <- cumsum(genre_pca$sdev^2 / sum(genre_pca$sdev^2))
  k_85 <- which(explained >= 0.85)[1]
  
  genre_pc_train <- genre_pca$x[, 1:k_85, drop = FALSE]
  genre_pc_holdout <- predict(genre_pca, holdout_genre_mat)[, 1:k_85, drop = FALSE]
  genre_pc_test <- predict(genre_pca, test_genre_mat)[, 1:k_85, drop = FALSE]

  colnames(genre_pc_train)   <- paste0("genre_pc_", 1:k_85)
  colnames(genre_pc_holdout) <- paste0("genre_pc_", 1:k_85)
  colnames(genre_pc_test)    <- paste0("genre_pc_", 1:k_85)
  
  train_raw   <- cbind(train_raw, genre_pc_train)
  holdout_raw <- cbind(holdout_raw, genre_pc_holdout)
  test_raw    <- cbind(test_raw, genre_pc_test)
  
  colnames(train_raw)[(ncol(train_raw)-k_85+1):ncol(train_raw)] <- paste0("genre_pc_",1:k_85)
  colnames(holdout_raw)[(ncol(holdout_raw)-k_85+1):ncol(holdout_raw)] <- paste0("genre_pc_",1:k_85)
  colnames(test_raw)[(ncol(test_raw)-k_85+1):ncol(test_raw)] <- paste0("genre_pc_",1:k_85)
  
  movie_stats <- train_raw %>%
    group_by(movieId) %>%
    summarise(
      movie_n = n(),
      movie_mean_raw = mean(rating, na.rm = TRUE),
      movie_sd_raw   = sd(rating, na.rm = TRUE),
      .groups = "drop"
    )
  
  movie_stats <- movie_stats %>%
    mutate(
      movie_sd_raw = ifelse(
        is.na(movie_sd_raw) | movie_sd_raw == 0,
        global_sd,
        movie_sd_raw
      )
    )
  
  lambda_movie <- 10
  
  movie_stats <- movie_stats %>%
    mutate(
      movie_mean = (movie_n * movie_mean_raw + lambda_movie * global_mean) /
        (movie_n + lambda_movie),
      
      movie_sd = (movie_n * movie_sd_raw + lambda_movie * global_sd) /
        (movie_n + lambda_movie)
    )
  
  train_raw   <- left_join(train_raw, movie_stats, by = "movieId")
  holdout_raw <- left_join(holdout_raw, movie_stats, by = "movieId")
  test_raw    <- left_join(test_raw, movie_stats, by = "movieId")
  
  pc_cols <- grep("^genre_pc_", colnames(movie_stats), value = TRUE)
  
  pc_mean_model <- lm(
    movie_mean ~ .,
    data = movie_stats[, c("movie_mean", pc_cols)],
    weights = sqrt(movie_stats$movie_n)
  )
  
  pc_sd_model <- lm(
    movie_sd ~ .,
    data = movie_stats[, c("movie_sd", pc_cols)],
    weights = sqrt(movie_stats$movie_n)
  )
  
  movie_stats$movie_pc_mean <- predict(pc_mean_model, movie_stats)
  movie_stats$movie_pc_sd   <- predict(pc_sd_model, movie_stats)
  
  movie_pc_raw <- movie_stats %>%
    select(movieId, movie_pc_mean, movie_pc_sd)
  
  train_raw   <- left_join(train_raw, movie_pc_raw, by = "movieId")
  holdout_raw <- left_join(holdout_raw, movie_pc_raw, by = "movieId")
  test_raw    <- left_join(test_raw, movie_pc_raw, by = "movieId")

  user_stats <- train_raw %>%
    group_by(userId) %>%
    summarise(
      user_n       = n(),
      user_avg_raw = mean(rating, na.rm = TRUE),
      user_sd_raw  = sd(rating, na.rm = TRUE),
      .groups = "drop"
    )
  
  global_user_mean <- mean(user_stats$user_avg_raw, na.rm = TRUE)
  global_user_sd   <- mean(user_stats$user_sd_raw, na.rm = TRUE)
  
  user_stats <- user_stats %>%
    mutate(
      user_sd = ifelse(
        is.na(user_sd_raw) | user_sd_raw == 0 | is.infinite(user_sd_raw),
        global_user_sd,
        user_sd_raw
      )
    )
  
  lambda_user <- 10
  
  user_stats <- user_stats %>%
    mutate(
      user_avg = (user_n * user_avg_raw + lambda_user * global_user_mean) /
        (user_n + lambda_user),
      
      user_sd = (user_n * user_sd_raw + lambda_user * global_user_sd) /
        (user_n + lambda_user)
    )
  
  attach_user <- function(df) {
    df %>%
      left_join(user_stats, by = "userId") %>%
      mutate(
        user_n   = coalesce(user_n, 0),
        user_avg = coalesce(user_avg, global_user_mean),
        user_sd  = coalesce(user_sd, global_user_sd)
      )
  }
  
  train_raw <- attach_user(train_raw)
  holdout_raw <- attach_user(holdout_raw)
  test_raw <- attach_user(test_raw)
  
  user_pc_cols <- grep("^genre_pc_", colnames(user_stats), value = TRUE)
  
  user_pc_mean_model <- lm(
    user_avg ~ .,
    data = user_stats[, c("user_avg", pc_cols)],
    weights = sqrt(user_stats$user_n)
  )
  
  user_pc_sd_model <- lm(
    user_sd ~ .,
    data = user_stats[, c("user_sd", pc_cols)],
    weights = sqrt(user_stats$user_n)
  )
  
  user_stats$user_pc_mean <- predict(user_pc_mean_model, user_stats)
  user_stats$user_pc_sd   <- predict(user_pc_sd_model, user_stats)
  
  user_pc_raw <- user_stats %>%
    select(userId, user_pc_mean, user_pc_sd)
  
  train_raw   <- left_join(train_raw, user_pc_raw, by = "userId")
  holdout_raw <- left_join(holdout_raw, user_pc_raw, by = "userId")
  test_raw    <- left_join(test_raw, user_pc_raw, by = "userId")
  
  train_raw$user_z2 <- ((train_raw$user_avg - global_user_mean) / train_raw$user_sd)^2
  holdout_raw$user_z2 <- ((holdout_raw$user_avg - global_user_mean) / holdout_raw$user_sd)^2
  test_raw$user_z2 <- ((test_raw$user_avg - global_user_mean) / test_raw$user_sd)^2
  
  train_raw$movie_z2 <- ((train_raw$movie_mean - global_mean) / train_raw$movie_sd)^2
  holdout_raw$movie_z2 <- ((holdout_raw$movie_mean - global_mean) / holdout_raw$movie_sd)^2
  test_raw$movie_z2 <- ((test_raw$movie_mean - global_mean) / test_raw$movie_sd)^2
  
  train_raw$user_movie_volatility <- train_raw$user_z2 * train_raw$movie_z2
  holdout_raw$user_movie_volatility <- holdout_raw$user_z2 * holdout_raw$movie_z2
  test_raw$user_movie_volatility <- test_raw$user_z2 * test_raw$movie_z2
  
  train_raw$user_movie_volatility <- log1p(train_raw$user_movie_volatility)
  holdout_raw$user_movie_volatility <- log1p(holdout_raw$user_movie_volatility)
  test_raw$user_movie_volatility <- log1p(test_raw$user_movie_volatility)
  
  train_raw$user_n_log <- log1p(train_raw$user_n)
  holdout_raw$user_n_log <- log1p(holdout_raw$user_n)
  test_raw$user_n_log <- log1p(test_raw$user_n)
  
  train_raw$usern_log_movie_sd <- train_raw$user_n_log * train_raw$movie_sd
  holdout_raw$usern_log_movie_sd <- holdout_raw$user_n_log * holdout_raw$movie_sd
  test_raw$usern_log_movie_sd <- test_raw$user_n_log * test_raw$movie_sd
  
  train_raw$usern_log_movie_mean <- train_raw$user_n_log * train_raw$movie_mean
  holdout_raw$usern_log_movie_mean <- holdout_raw$user_n_log * holdout_raw$movie_mean
  test_raw$usern_log_movie_mean <- test_raw$user_n_log * test_raw$movie_mean
  
  train_raw$movie_year <- as.numeric(str_extract(train_raw$title, "\\d{4}"))
  holdout_raw$movie_year <- as.numeric(str_extract(holdout_raw$title, "\\d{4}"))
  test_raw$movie_year <- as.numeric(str_extract(test_raw$title, "\\d{4}"))
  
  train_raw$release_date <- as.Date(paste0(train_raw$movie_year, "-01-01"))
  holdout_raw$release_date <- as.Date(paste0(holdout_raw$movie_year, "-01-01"))
  test_raw$release_date <- as.Date(paste0(test_raw$movie_year, "-01-01"))
  
  train_raw$rating_date <- as.Date(as.POSIXct(train_raw$timestamp, origin = "1970-01-01"))
  holdout_raw$rating_date <- as.Date(as.POSIXct(holdout_raw$timestamp, origin = "1970-01-01"))
  test_raw$rating_date <- as.Date(as.POSIXct(test_raw$timestamp, origin = "1970-01-01"))
  
  train_raw$time_since_release <- as.numeric(train_raw$rating_date - train_raw$release_date)
  holdout_raw$time_since_release <- as.numeric(holdout_raw$rating_date - holdout_raw$release_date)
  test_raw$time_since_release <- as.numeric(test_raw$rating_date - test_raw$release_date)
  
  movie_time_sd <- train_raw %>%
    group_by(movieId) %>%
    summarise(
      time_since_release_sd = sd(time_since_release, na.rm = TRUE),
      .groups = "drop"
    )
  
  train_raw <- left_join(train_raw, movie_time_sd, by = "movieId")
  holdout_raw <- left_join(holdout_raw, movie_time_sd, by = "movieId")
  test_raw <- left_join(test_raw, movie_time_sd, by = "movieId")
  
  train_raw$time_since_release_sd <- coalesce(train_raw$time_since_release_sd, 0)
  holdout_raw$time_since_release_sd <- coalesce(holdout_raw$time_since_release_sd, 0)
  test_raw$time_since_release_sd <- coalesce(test_raw$time_since_release_sd, 0)
  
  train_raw$tsd_x_movie_mean <- train_raw$time_since_release_sd * train_raw$movie_mean
  holdout_raw$tsd_x_movie_mean <- holdout_raw$time_since_release_sd * holdout_raw$movie_mean
  test_raw$tsd_x_movie_mean <- test_raw$time_since_release_sd * test_raw$movie_mean
  
  train_raw$tsd_x_movie_sd <- train_raw$time_since_release_sd * train_raw$movie_sd
  holdout_raw$tsd_x_movie_sd <- holdout_raw$time_since_release_sd * holdout_raw$movie_sd
  test_raw$tsd_x_movie_sd <- test_raw$time_since_release_sd * test_raw$movie_sd
  
  train_raw$tsd_x_user_n_log <- train_raw$time_since_release_sd * train_raw$user_n_log
  holdout_raw$tsd_x_user_n_log <- holdout_raw$time_since_release_sd * holdout_raw$user_n_log
  test_raw$tsd_x_user_n_log <- test_raw$time_since_release_sd * test_raw$user_n_log
  
  rmse <- function(a, p) sqrt(mean((a - p)^2, na.rm = TRUE))
  
  colSums(is.na(train_raw[, c(
    "tsd_x_user_n_log",
    "tsd_x_movie_sd",
    "tsd_x_movie_mean",
    "usern_log_movie_mean",
    "usern_log_movie_sd",
    "user_movie_volatility",
    "movie_pc_sd",
    "movie_pc_mean",
    "user_pc_mean",
    "user_pc_sd"
  )]))
  
  colSums(is.infinite(as.matrix(train_raw[, c(
    "tsd_x_user_n_log",
    "tsd_x_movie_sd",
    "tsd_x_movie_mean",
    "usern_log_movie_mean",
    "usern_log_movie_sd",
    "user_movie_volatility",
    "movie_pc_sd",
    "movie_pc_mean",
    "user_pc_sd",
    "user_pc_mean"
  )])))
  
  lm_fit <- lm(
    rating ~
      tsd_x_user_n_log +
      tsd_x_movie_sd +
      tsd_x_movie_mean +
      usern_log_movie_mean +
      usern_log_movie_sd +
      user_movie_volatility +
      movie_pc_sd +
      movie_pc_mean +
      user_pc_mean +
      user_pc_sd,
    data = train_raw
  )
  
  lm_pred_holdout <- predict(lm_fit, newdata = holdout_raw)
  lm_holdout_rating <- lm_pred_holdout
  
  lm_pred_test <- predict(lm_fit, newdata = test_raw)
  lm_test_rating <- lm_pred_test

  rf_features <- c(
    "tsd_x_user_n_log",
    "tsd_x_movie_sd",
    "tsd_x_movie_mean",
    "usern_log_movie_mean",
    "usern_log_movie_sd",
    "user_movie_volatility",
    "movie_pc_sd",
    "movie_pc_mean",
    "user_pc_mean",
    "user_pc_sd"
  )
  
  mtry_grid <- expand.grid(
    mtry = c(3,4,5),
    min.node.size = c(3,4)
  )
  
  rf_formula <- as.formula(
    paste("rating ~", paste(rf_features, collapse = " + "))
  )
  
  rf_models <- lapply(seq_len(nrow(mtry_grid)), function(i) {
    
    ranger(
      formula = rf_formula,
      data = train_raw,
      num.trees = 200,
      mtry = mtry_grid$mtry[i],
      min.node.size = mtry_grid$min.node.size[i],
      sample.fraction = 0.8,
      importance = "impurity"
    )
    
  })
  
  rf_preds <- lapply(rf_models, function(model) {
    predict(model, data = holdout_raw)$predictions
  })
  
  rf_rmse <- sapply(rf_preds, function(p) {
    rmse(holdout_raw$rating, p)
  })
  
  names(rf_rmse) <- paste0("mtry_", mtry_grid)
  
  rf_rmse
  
  best_index <- which.min(rf_rmse)
  best_rf <- rf_models[[best_index]]
  best_mtry <- mtry_grid[best_index]
  
  rf_test_rating <- predict(best_rf, data = test_raw)$predictions
  
  holdout_lm_rmse <- rmse(holdout_raw$rating, lm_holdout_rating)
  holdout_rf_rmse <- rmse(holdout_raw$rating, rf_preds[[best_index]])
  
  holdout_rmse_table <- data.frame(
    model = c("lm", "rf"),
    rmse  = c(holdout_lm_rmse, holdout_rf_rmse)
  )
  
  holdout_rmse_table
  
  w_lm  <- 0.50
  w_rf  <- 0.50
  
  ensemble5_holdout <- 
    w_lm  * lm_holdout_rating +
    w_rf  * rf_preds[[best_index]]
  
  ensemble5_holdout_rmse <- rmse(holdout_raw$rating, ensemble5_holdout)
  
  final_holdout_rmse_table <- data.frame(
    model = c("lm", "rf", "ensemble"),
    rmse  = c(
      holdout_lm_rmse,
      holdout_rf_rmse,
      ensemble5_holdout_rmse
    )
  ) %>%
    arrange(rmse)
  
  final_holdout_rmse_table
  
  ensemble5_test <- 
    w_lm  * lm_test_rating +
    w_rf  * rf_test_rating
  
  
  test_rmse_table <- data.frame(
    model = c("lm", "rf", "ensemble"),
    rmse = c(
      rmse(test_raw$rating, lm_test_rating),
      rmse(test_raw$rating, rf_test_rating),
      rmse(test_raw$rating, ensemble5_test)
    )
  )
  
  test_rmse_table[order(test_rmse_table$rmse), ]