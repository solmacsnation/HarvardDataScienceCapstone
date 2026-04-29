# HarvardDataScienceCapstone
Movielens data (think Netflix Prize) subjected to feature engineering (summary of features in R Markdown file), as well as linear and random forest modeling.

# Feature Engineering Report

This report summarizes the purpose of the engineered features used to predict movie ratings in the MovieLens 10M dataset. Each feature is designed to capture systematic variation in user preferences, movie characteristics, and temporal dynamics that may influence observed ratings.

The dimensionality of the genre feature space is reduced using principal component analysis (PCA). The number of retained components corresponds to **85% of explained variance**, which represents a deliberately **low-octane choice** intended to capture most genre information while avoiding aggressive dimensionality expansion.

---

# Genre Principal Components (`genre_pc_k`)

The original movie genres are converted into a binary indicator matrix and reduced using principal component analysis to produce a smaller set of orthogonal genre features. These components summarize patterns of co-occurrence among genres and allow the model to capture broad stylistic similarities between films without relying on a large number of correlated genre indicators.

The number of components retained is chosen to explain **85% of the variance in the genre matrix**, providing a compressed but informative representation of genre structure. This choice represents a **low-octane dimensionality reduction strategy**, preserving most information while avoiding unnecessary complexity.

---

# Movie Mean Rating (`movie_mean`)

This feature represents the average rating received by each movie in the training data. It captures the collective evaluation of a film by the user population and serves as a strong baseline indicator of movie quality.

To reduce noise from sparsely rated movies, the mean is **regularized toward the global mean rating** using a shrinkage parameter. This stabilization prevents movies with few ratings from appearing artificially extreme.

---

# Movie Rating Standard Deviation (`movie_sd`)

The movie rating standard deviation measures the dispersion of ratings for each film. It captures disagreement among viewers and identifies movies that generate polarized reactions.

The standard deviation is also **regularized toward the global rating standard deviation** to stabilize estimates when the number of ratings is small. This ensures that films with very few ratings do not produce unreliable volatility estimates.

---

# Movie Genre-Predicted Mean (`movie_pc_mean`)

This feature is a predicted movie rating mean derived from a regression of movie mean ratings on genre principal components. It represents the **expected rating of a film based purely on its genre composition**.

The model is weighted by the square root of the number of ratings per movie so that more frequently rated films contribute more heavily to the estimation. This feature captures systematic genre-based rating tendencies that may generalize to new or sparsely rated movies.

---

# Movie Genre-Predicted Rating Variability (`movie_pc_sd`)

This variable predicts the standard deviation of movie ratings using the same genre principal components used for the predicted mean. It represents the **expected level of disagreement among viewers given the genre profile of a film**.

By modeling variance as a function of genre, the feature captures the idea that certain genres—such as experimental or niche categories—may naturally produce more varied audience reactions.

---

# User Average Rating (`user_avg`)

The user average rating measures the typical rating level given by each user. This captures systematic differences in rating behavior, such as whether a user tends to rate movies generously or critically.

Like the movie statistics, this value is **regularized toward the global user mean** to stabilize estimates for users with few ratings. This prevents sparse user histories from producing unstable behavioral estimates.

---

# User Rating Standard Deviation (`user_sd`)

The user rating standard deviation measures how variable a user's ratings are across movies. Some users rate consistently within a narrow range, while others exhibit large variation depending on the film.

Regularization toward the global user standard deviation ensures that volatility estimates remain stable for users with limited rating histories.

---

# User Genre-Predicted Mean (`user_pc_mean`)

This feature predicts a user's average rating level based on the genre principal component structure. It captures how the **genre mix of movies a user typically watches relates to their overall rating tendencies**.

By linking user behavior to genre structure, the feature helps approximate how a user's preferences interact with the genre space represented by the PCA components.

---

# User Genre-Predicted Rating Variability (`user_pc_sd`)

This variable predicts how variable a user's ratings are expected to be based on genre composition. It reflects the idea that users who engage with a wider or more experimental set of genres may display more volatile rating patterns.

Including this feature allows the model to account for systematic differences in rating dispersion across different types of users.

---

# User Deviation Index (`user_z2`)

The user deviation index measures the squared standardized distance between a user's average rating and the global user mean. It quantifies how strongly a user deviates from the typical rating behavior of the population.

Squaring the standardized value emphasizes larger deviations and ensures that both positive and negative deviations contribute equally to the measure.

---

# Movie Deviation Index (`movie_z2`)

The movie deviation index measures the squared standardized distance between a movie’s mean rating and the global mean rating. It identifies films whose reception differs substantially from the overall rating baseline.

Like the user deviation measure, squaring the standardized value magnifies extreme differences and produces a non-directional measure of rating extremeness.

---

# User–Movie Volatility (`user_movie_volatility`)

This feature is the product of the squared user and movie deviation indices, measuring the interaction between extreme users and extreme movies. It captures situations in which strong rating tendencies from both sides may produce unusually large prediction uncertainty.

A logarithmic transformation is applied to compress extreme values and stabilize the distribution. This prevents rare but extreme interactions from dominating the model.

---

# Log User Rating Count (`user_n_log`)

The logarithm of the number of ratings submitted by a user measures their level of activity on the platform. Highly active users typically exhibit more stable preference patterns than occasional users.

The log transformation reduces skew in the distribution of rating counts and ensures that extremely active users do not dominate the scale of the feature.

---

# Interaction: User Activity × Movie Mean (`usern_log_movie_mean`)

This feature interacts user activity with the average movie rating. It captures the possibility that highly active users may evaluate popular or highly rated films differently from casual users.

The interaction allows the model to account for differences in how experienced users respond to widely liked movies.

---

# Interaction: User Activity × Movie Rating Variability (`usern_log_movie_sd`)

This interaction combines user activity with the variability of movie ratings. It captures how experienced users respond to films that generate disagreement among viewers.

Active users may develop stronger opinions about controversial or polarizing movies, making this interaction potentially informative.

---

# Time Since Release (`time_since_release`)

This variable measures the number of days between a movie’s release date and the time when a rating was submitted. It captures temporal dynamics such as nostalgia effects or shifting audience perceptions over time.

Older films may accumulate ratings from viewers who have different expectations than those watching newly released films.

---

# Time-Since-Release Dispersion (`time_since_release_sd`)

This feature measures the variability in the time elapsed between release and ratings for each movie. It captures whether ratings for a film are concentrated near its release period or spread across a long time horizon.

Movies with long temporal engagement may reflect enduring cultural relevance or rediscovery by later audiences.

---

# Temporal Interaction: Dispersion × Movie Mean (`tsd_x_movie_mean`)

This feature interacts the dispersion of rating times with the average movie rating. It captures whether films with sustained viewing interest also tend to have stronger or weaker average reception.

The interaction allows the model to detect long-lived movies that remain highly rated over time.

---

# Temporal Interaction: Dispersion × Movie Rating Variability (`tsd_x_movie_sd`)

This feature measures how the dispersion of rating times interacts with rating variability. It captures whether films that remain visible over long periods also generate more polarized reactions.

Such patterns may occur with cult classics or controversial films that continue to attract discussion.

---

# Temporal Interaction: Dispersion × User Activity (`tsd_x_user_n_log`)

This variable interacts the dispersion of rating times with user activity levels. It captures whether highly active users disproportionately rate films with long viewing lifetimes.

This interaction may reflect experienced users revisiting older films or engaging with historically significant titles.

---

# Summary

The engineered features combine **movie-level statistics, user behavior metrics, genre structure, interaction terms, and temporal dynamics** to capture multiple dimensions of rating behavior. Together, these variables create a structured representation of the recommendation environment that can be exploited by both linear models and tree-based machine learning algorithms.

The use of PCA to retain **85% of genre variance** represents a **low-octane dimensionality reduction strategy**, balancing information retention with model simplicity.

