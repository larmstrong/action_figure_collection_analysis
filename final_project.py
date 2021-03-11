
# %% PROBLEM STATEMENT ---------------------------------------------------------
# Final Project: 1:6 Scale Action Figure Analysis
#        Author: Leonard Armstrong
#           Log: 2021-Mar-04 (LA) Copied from HW01 for final project expansion.
#                2021-Feb-04 (LA) Original version.


# %% LOAD LIBRARIES ------------------------------------------------------------

# Partial library imports
from mizani.formatters import dollar_format		# Format axes with money values
from mizani.formatters import percent_format	# Format axes with pct values
from numpy import float64, int64

import credentials								# Account ids, pwds, etc.
import matplotlib as mpl						# For figure resizing
import matplotlib.pyplot as plt					# Ubiquitous Python plotting lib
import matplotlib.ticker as ticker				# For formatting plt axes
import numpy as np								# Numeric classes/functions
import pathlib as pl
import pandas as pd
import plotnine as gg							# "gg" refers to ggplot.
import seaborn as sns							# Seaborn plotting package			
import squarify									# Generate treemaps.
import statsmodels.api as sm					# Regression models
import tweepy									# Twitter API

# %% DEFINE GLOBAL CONSTANTS ---------------------------------------------------

DATA_PATH = "."									# Relative path to data file
DATA_FNAME = "action_figures_a.csv"				# Name of data file

# This map is used to rename a subset of columns. Not all columns are renamed.
COLUMN_RENAME_MAP = {
    "action_figure_description" : "af_descr",
	"FigureId"                  : "figure_id",
	"Manufacturer"              : "manufacturer",
	"Product"                   : "product_name",
	"ProductId"                 : "product_id",
    "Product Description"       : "product_descr",
	"Purchased From"            : "seller",
	"purchase_price"            : "price",
	"Release Year"              : "year",
}

# Define a set of one-hot-endcoded field for the genres.
GENRES = {
	"Adventure", "Air Force", "Armor", "Army", "Astronaut", "Avengers",
	"Celebrity", "Civilian", "Coast Guard", "Comics", "DC Comics", "Fashion",
	"Fire Fighter", "Foreign", "Horror", "Knight", "Marines", "Martial Arts", 
	"Marvel Comics", "Navy", "Police", "RAH/Cobra", "Sci-Fi", "Sports", "Spy", 
	"TV/Film", "Warrior", "Western", "World Leader", "X-Men" }

# Value to use when a year value is unknown
UNKNOWN_YEAR = 0

# Define a high-contrast color palette to assist those of us who are colorblind
# The palette is a subset of the colors defined in a palette published in
# https://jxnblk.com/blog/color-palette-documentation-for-living-style-guides/
MY_COLORS = [
	'#0074D9', '#2ECC40', '#FFDC00', '#FF4136', '#AAAAAA',
	'#DDDDDD', '#7FDBFF', '#39CCCC', '#3D9970', '#01ff70', '#FF851B' ]

# Twitter accounts are not (currently) part of joebase so we create a dictionary
# of key Twitter accounts here.
# TODO: Add Twitter accounts to joebase manufacturer table.
TWITTER_ACCTS = {
	"Hasbro" : "Hasbro",
	"Sideshow" : "collectsideshow",
	"Hot Toys" : "hottoysofficial",
	"DC Direct/DC Collectibles" : "DCCollectibles",
	"G.I. Joe Collector's Club" : "The_GIJoe_Club" }

MAX_TWEET_HISTORY = 3000

# %% TWITTER_OAUTH_LOGIN -------------------------------------------------------

def oauth_login () :
	"""
	Get authorization from Twitter. Enable Tweepy API with said authorization

	Returns:
		tweepy.api.API: Tweepy API interface object.
	"""	
	# Authenticate to Twitter.
	auth = tweepy.OAuthHandler(
		credentials.TWITTER_CONSUMER_KEY,
		credentials.TWITTER_CONSUMER_SECRET)
	auth.set_access_token(
		credentials.TWITTER_OAUTH_TOKEN,
		credentials.TWITTER_OAUTH_SECRET)
	# Enable Tweep API with Twitter credentials.
	api = tweepy.API(auth)
	# Guarantee a valid API is available.
	assert api
	# Return the API object.
	return api


# %% GET_TWEET_HISTORY ---------------------------------------------------------

def get_tweet_history (qstr : str) -> pd.DataFrame :
	"""
	Get all recent tweet history for an input query string. "All" is also
	limited to not more than MAX_TWEET_HISTORY tweets to ensure popular query
	strings to not cause excessive output.

	Args:
		qstr (str): Twitter query string

	Returns:
		pd.DataFrame: Dataframe representation of all returned tweets.
	"""	
	# Use a very large number up front so we get the most recent tweet.
	last_id = 99999999999999999999

	# Initialize total tweet counter
	total_tweets = 0

	# Initialize a dataframe for the result
	df = pd.DataFrame()

	# Loop until no more tweets are available.
	more_tweets = True
	while more_tweets :
		tweets = tapi.search(			# Search for tweets
			q=qstr,						# Use the given query string
			count=100,					# Get up to 100 (the max) per request
			lang="en",					# Look for only English tweets
			result_type = "recent",		# Order from most to least recent
			max_id = last_id,			# The starting point for a new batch
			tweet_mode='extended')		# Get full text
		
		# Update the tweet counts for the batch and the total
		num_tweets_in_batch = len(tweets)
		total_tweets += num_tweets_in_batch

		# Add the tweets to the dataframe
		new_tweet_list = [t._json for t in tweets]
		df=df.append(new_tweet_list, ignore_index=True)

		if num_tweets_in_batch > 0 :
			# If any tweets were returned use the ID of the last tweet as the
			# starting point for the next batch.
			last_id = tweets[num_tweets_in_batch-1].id
		if num_tweets_in_batch < 100 or total_tweets >= MAX_TWEET_HISTORY :
			# If we recieved less than the requested number of tweets, that's 
			# all there is. No more tweets.
			more_tweets = False
		print(f'{qstr}:: Total: {total_tweets}; Batch: {len(tweets)}; Last Id: {last_id}')

	return (df)


# %% FROM_PICKLE_OR_TWITTER ----------------------------------------------------

def from_pickle_or_twitter (qstr : str, pkl_fpath: str) -> pd.DataFrame :
	"""
	Generate a dataframe from either (a) a pickle file if one exists or
	(b) an API read from Twitter. If Twitter is used as the source then a 
	pickle file is written out to disk immediately after so fast-access data is 
	subsequently available. 

	Args:
		qstr (str): Twitter query string, if needed. 
		pkl_fpath (str): Pickle file path for reading, if available, or writing,
		if not initially available.

	Returns:
		pd.DataFrame: Dataframe representation of tweets
	"""	
	pkl_fpath = pl.Path(pkl_fpath) 
	if pkl_fpath.exists() :
		df = pd.read_pickle(pkl_fpath)
	else :
		df = get_tweet_history(qstr)
		df.to_pickle(path=pkl_fpath)
	return df


# %% READ DATA -----------------------------------------------------------------

# Create a proper file path.
data_fpath = pl.Path(DATA_PATH, DATA_FNAME)
assert data_fpath.exists()

# Read data into a pandas dataframe with default data types.
with open(data_fpath, "r") as datafile :
	fig_data = pd.read_csv(datafile, sep=',')
print(
	f'\nTHE DATA SET CONTAINS {fig_data.shape[0]} ROWS '
	f'AND {fig_data.shape[1]} COLUMNS.\n')


# %% CLEAN THE DATA ------------------------------------------------------------

# Rename some of the columns, as desired.
fig_data.rename(columns=COLUMN_RENAME_MAP, inplace=True)

# Remove records with a null or NaN figure id. These represent figure costume
# sets, furniture, vehicles, etc, not figures.
non_figure_rows = fig_data[fig_data['figure_id'].isnull()].index
fig_data.drop(labels=non_figure_rows, axis=0, inplace=True)

# Set any unknown year to -1
fig_data["year"].fillna(UNKNOWN_YEAR, inplace=True)

# Update the data types as desired/required. Due to NA values these fields will
# be orginally read as float64. We can convert to int64 now that the offending
# NA values have been removed.
fig_data["figure_id"]=fig_data["figure_id"].astype('int64')
fig_data["year"]=fig_data["year"].astype('int64')

# Add a half-decade field as a string type. This works out better for graphing.
half_decade = fig_data["year"] - fig_data["year"].mod(5)
fig_data["Half Decade"] = [str(x) for x in half_decade]


# %% PROVIDE BASIC DESCRIPTIVE STATISTICS SUMMARIES ----------------------------

# What is the shape of the data?
data_shape = fig_data.shape
print(
	f'\nTHE CLEANSED DATA SET CONTAINS {data_shape[0]} ROWS '
	f'AND {data_shape[1]} COLUMNS.')

# What data types are in the data set?
print('\nTHE DATA SET HAS THE FOLLOWING COLUMNS AND DATA TYPES:')
print(fig_data.dtypes)

# Provide descriptive statistics on manufacturers and sellers
print('\nTHE MANUFACTURER AND SELLER ATTRIBUTES HAS THE FOLLOWING STATISTCS:')
print(fig_data.describe(include='all')[{'manufacturer', 'seller'}])

# Provide descriptive statistics on figure prices.
print('\nTHE PRICE ATTRIBUTE HAS THE FOLLOWING STATISTCS:')
print(fig_data['price'].describe())


# %% ---------------------------------------------------------------------------
# QUESTION 1: WHAT IS THE AVERAGE FIGURE PRICE PAID PER YEAR?

# Get a slice of the data that has a valid year and price. The price data is
# placed into a DataFrame.
year_price_data=fig_data[fig_data['year']>0][["year", "price"]].dropna().copy()

# Graph price distribution as a single boxplot.
gg.options.figure_size=(4, 6)
g = (
	gg.ggplot(data=year_price_data) # prices_df)
	+ gg.geom_boxplot(mapping=gg.aes(x=['']*len(year_price_data), y='price'))
	+ gg.theme_bw()
	+ gg.ggtitle('Ranges of Prices Paid Across All Figures')
	+ gg.xlab('')
	+ gg.ylab('Price Paid')
	+ gg.scale_y_continuous(labels=dollar_format(digits=0)))
g.draw()
plt.show()

# Group figures by year and get counts per year.
year_gb = fig_data.groupby('year')
volume_per_year = year_gb.aggregate('count')
volume_per_year.drop(0, inplace=True)

# Plot a histogram of count of figures per year.
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 100 
fig, ax = plt.subplots()									# Create a graph
plt.bar(x=volume_per_year.index, height=volume_per_year["figure_id"])
ax.set_title('Year of Production of Figures in Collection')	# Title the chart
ax.set_xlabel('Year of Release')							# Title the x-axis
ax.set_ylabel('Volume of Figures')							# Title the y-axis
plt.show()
mpl.rcParams.update(mpl.rcParamsDefault)


# %% ---------------------------------------------------------------------------
# QUESTION 1: WHAT IS THE AVERAGE FIGURE PRICE PAID PER YEAR?

# First get a subset of figures with a valid year and price, filtering out
# figures with no price or an unknown year.
figs_with_prices = fig_data[(~fig_data["price"].isna()) & (fig_data["year"]>0)]

# Group figures by year.
year_gb = figs_with_prices.groupby(by="year", axis=0)

# Report the inter-quartile ranges for each year.
iqr = (
	year_gb.quantile(q=0.75)['price'] - year_gb.quantile(q=0.25)['price']
	).reset_index()										# Turns year into a col
iqr.columns = ('Year', 'Price IQR')						# Name columns in DF
iqr["0.25"] = year_gb.quantile(q=0.25)['price'].values	# Give explicit Q2 value
iqr["0.75"] = year_gb.quantile(q=0.75)['price'].values	# Give explicit Q4 value

print('\nTHE INTERQUARTILE RANGES FOR PRICE ARE:')
print(
	f"\n{'Year'}\t{'2nd Q':>5s}\t{'4th Q':>5s}\t{'IQR':>5s}",
	f"\n{'----'}\t{'-----':>5s}\t{'-----':>5s}\t{'---':>5s}")
for idx, row in iqr.iterrows() :
	print(
		f"{row['Year']:.0f}\t"
		f"${row['0.25']:4.0f}\t"
		f"${row['0.75']:4.0f}\t"
		f"${row['Price IQR']:4.0f}")
iqr.to_csv('price_iqr.csv')

# Get the number of figures aquired for each production year. (One row per
# year.) Create a dataframe from the results. Reset the index to year back as a
# column instead of the index (from group-by).)
num_aquired = year_gb.aggregate(len)["figure_id"]
avg_price = year_gb.aggregate('mean')["price"]
vol_price = pd.DataFrame(
	{
		"volume" : year_gb.aggregate(len)["figure_id"],
		"avg_price" : year_gb.aggregate('mean')["price"]
	}).reset_index()


# %% Q1 GRAPH: ACTUAL AND AVERAGE PRICES PAID ----------------------------------

# Plot actual prices and average annual prices on one graph.
gg.options.figure_size=(14, 11)
g_prices = (
	gg.ggplot(
		data=figs_with_prices,
		mapping=gg.aes(x="year", y="price"))
	+ gg.labs(								# Define all labels on graph
		title = "Actual and Average Price Paid Per Year",
		subtitle = "x-es are actual price; blue dots are averages", 
		x = "Year",
		y = "Price")
	+ gg.theme_light() 						# Plot an overall "light" theme
	+ gg.theme(
		plot_title=gg.element_text(			# Update the title font
			size=14,						# 	14-point font
			face="bold"),					# 	Bold text
		axis_title=gg.element_text(			# Update the axis font
			face="bold"))					# 	Bold text
	+ gg.geom_jitter(						# Plot data with some jitter
		shape="x", 							# 	Use "x"-shaped markers
		width=0.35,							# 	Horiz. jitter = 35% of resol'n
		height = 0,							# 	No vertical jitter
		color = "#555555", 					# 	Plot markers in a gray tone
		size=2)								# 	Slightly increase marker size
	+ gg.geom_point(
		data=vol_price,						# Use an average prices data set
		mapping=gg.aes(
			x="year",
			y="avg_price"),
		size=4,								# 	Increase marker size
		color='blue')						#	 Color markers blue
	+ gg.geom_smooth(						# Plot a regression line
		data=vol_price,						# 	Again, use avg prices data set
		mapping=gg.aes(
			x="year", 
			y="avg_price"),
		stat=gg.stat_smooth(method='lm'),	# 	Use linear-method smoothing	
		linetype = 'dashed')				# 	Use a dashed line
	+ gg.scale_y_continuous(
		labels=dollar_format(digits=0)))	# $-fmt y-axis
g_prices.draw()
plt.show()


# %% ---------------------------------------------------------------------------

# Get a list of manufacturers to focus on. First, count number of products from
# each manufecturer.
manu_prod_counts = (
	fig_data[['manufacturer', 'product_id']]
	.groupby('manufacturer')
	.aggregate('count')
	.reset_index()
	.rename(columns={"product_id":"volume"}))
top_manus = manu_prod_counts[manu_prod_counts["volume"] > 10]["manufacturer"]

m = (
	fig_data[fig_data['manufacturer'].isin(top_manus)]
	.groupby('manufacturer')
	.aggregate('sum')
	.drop('product_id', axis=1)
	.drop('figure_id', axis=1)
	.drop('year', axis=1)
	.drop('price', axis=1)
	.drop('exclusive_to_retailer_id', axis=1))
	#.reset_index())
	#.melt(id_vars=['manufacturer']))

# gg.options.figure_size=(10, 10)
# g = (
# 	gg.ggplot(
# 		data=m, 
# 		mapping=gg.aes(x='manufacturer', y='variable'))
# 	+ gg.geom_)

# %% ---------------------------------------------------------------------------



fig, ax = plt.subplots(figsize=(16, 9))
g1 = sns.heatmap(ax=ax, data=m, cmap="Greys")
g1.axes.set_title("Manufacturer/Genre Heatmap\n",fontsize=20)
g1.set_xlabel(None)
g1.set_ylabel(None)
_ = plt.xticks(rotation=70)

fig, ax = plt.subplots(figsize=(16, 9))
g2 = sns.heatmap(ax=ax, data=m, vmax=25, cmap="Greys")
g2.axes.set_title("Manufacturer/Genre Heatmap (Max=25)\n",fontsize=20)
g2.set_xlabel(None)
g2.set_ylabel(None)
_ = plt.xticks(rotation=70)

# %% ---------------------------------------------------------------------------
# vol_price["cal_year"] = vol_price.index

# figs_with_prices["sqrt_price"] = figs_with_prices["price"].pow(0.5)
# figs_with_prices["ln_price"] = np.log(figs_with_prices["price"])


# gg.options.figure_size=(14, 11)
# g = (
# 	gg.ggplot(
# 		data=figs_with_prices,
# 		mapping=gg.aes(x="year", y="ln_price"))
# 	+ gg.theme_light() 
# 	+ gg.geom_point(shape="x", color = "#777777")
# 	# + gg.geom_point(
# 	# 	data=vol_price,
# 	# 	mapping=gg.aes(x="cal_year", y="avg_price"),
# 	# 	size=3,
# 	# 	color='blue')
# 	+ gg.geom_smooth(stat=gg.stat_smooth(method='lm'))
# )
# g.draw()
# plt.show()

# %% ---------------------------------------------------------------------------

# Create addition value transforms to be able to perform linear, 
# piecewise-linear, quadratic, and log regressions.
vol_price["piece"] = [ (1 if y <= 2008 else 2) for y in vol_price["year"] ]
vol_price["sqrt_price"] = np.sqrt(vol_price["avg_price"])
vol_price["log_price"] = np.log(vol_price["avg_price"])

# Fit #1: Simple linear regression
print("REGULAR LINEAR REGRESSION")
X_linear = sm.add_constant(vol_price["year"])
linear_model = sm.OLS(vol_price['avg_price'], X_linear)
linear_results = linear_model.fit()
print(linear_results.summary())
print(f"\n{'-' * 90}\n\n")

# Fit #2: Quadratic, using SQRT of avg_price
print("QUADRATIC REGRESSION")
X_quad = sm.add_constant(vol_price["year"])
quad_model = sm.OLS(vol_price['sqrt_price'], X_quad)
quad_results = quad_model.fit()
print(quad_results.summary())
print(f"\n{'-' * 90}\n\n")

# Fit #3: Exponential, using LOG of avg_price
# (math.e**(-198.5069))*(math.e**(0.101069*2019))
print("EXPONENTIAL REGRESSION")
X_exp = sm.add_constant(vol_price["year"])
exp_model = sm.OLS(vol_price['log_price'], X_exp)
exp_results = exp_model.fit()
print(exp_results.summary())
print(f"\n{'-' * 90}\n\n")

# Fit #3: Piecewise linear, piece #1
vol_price_piece1 = vol_price[vol_price['piece']==1]
vol_price_piece2 = vol_price[vol_price['piece']==2]

print("PIECEWISE REGRESSION - PIECE #1")
X_piece1 = sm.add_constant(vol_price_piece1["year"])
piece1_model = sm.OLS(vol_price_piece1['avg_price'], X_piece1)
piece1_results = piece1_model.fit()
print(piece1_results.summary())
print("PIECEWISE REGRESSION - PIECE #2")
X_piece2 = sm.add_constant(vol_price_piece2["year"])
piece2_model = sm.OLS(vol_price_piece2['avg_price'], X_piece2)
piece2_results = piece2_model.fit()
print(piece2_results.summary())
print(f"\n{'-' * 90}\n\n")

# %% ---------------------------------------------------------------------------


# Then graph the price ranges as a boxplot
gg.options.figure_size=(14, 11)
g = (
	gg.ggplot(figs_with_prices)
	+ gg.geom_boxplot(mapping=gg.aes(x='factor(year)', y='price'))
	+ gg.theme_bw()
	+ gg.ggtitle('Ranges of Prices Paid for Figures Per Year')
	+ gg.xlab('Year of Acquisition')
	+ gg.ylab('Price Paid')
	+ gg.scale_y_continuous(labels=dollar_format(digits=0)))
g.draw()
plt.show()

# Sve figs_with_prices
figs_with_prices.to_csv('figs_with_prices.csv')

# %% ---------------------------------------------------------------------------
# QUESTION 2: HOW DID GENRE COLLECTING CHANGE OVER THE YEARS?

# Create volume summaries of the 30 genres
genre_volumes = pd.DataFrame(
	[ { "Genre" : g, "Volume" : sum(fig_data[g]) } for g in GENRES ] )
genre_volumes.sort_values('Volume', ascending=False, inplace=True)
print(genre_volumes)

# Set the figure's size and DPI.
mpl.rcParams['figure.figsize'] = [12.0, 9.0]
mpl.rcParams['figure.dpi'] = 100 

# Create a treemap
fig, ax = plt.subplots()
squarify.plot(
	sizes=genre_volumes["Volume"],
	label=genre_volumes['Genre'][:24], 
	color=MY_COLORS, alpha=0.7, ax=ax)
fig.suptitle('Tree Map of the Volume of Action Figures Within Genres',
	fontsize='x-large', fontweight='bold')
ax.set_title('Note: An action figure may be in more than one genre.')
plt.axis('off')
plt.tight_layout()
plt.show()

# Reset to default figure size
mpl.rcParams.update(mpl.rcParamsDefault)

# Get the sums of figures in all genres
genre_sums = fig_data.xs(key=GENRES, axis=1).sum()

# Get the genres with less than 10 figures
top_genres = set(genre_sums[genre_sums >= 35].index)

# Configure data for a relative percent bar plot.
year_gb = fig_data[fig_data["year"]>=1995].groupby(by="Half Decade", axis=0)
genre_yr_sums = year_gb.sum()

# Drop all columns except those needed for the plot and then reshape the data
# into a long form.
drop_cols = set(genre_yr_sums.columns) - top_genres
genre_yr_sums.drop(columns=drop_cols, inplace=True)
genre_yr_sums["Half Decade"] = genre_yr_sums.index
genre_df = genre_yr_sums.melt(id_vars=['Half Decade'])

# Generate the plot
g = (gg.ggplot(data=genre_df)
	+ gg.geom_col(
		mapping=gg.aes(x='Half Decade', y="value", fill="variable"),
		position=gg.position_fill())
	+ gg.ggtitle('Relative Percentages of Top Six Genres Over Time')
	+ gg.ylab('Relative Percent Across These Genres')
	+ gg.scale_fill_manual(values=MY_COLORS)
	+ gg.scale_y_continuous(labels=percent_format()))
g.draw()
plt.show()

# Save genre_df
genre_df.to_csv('genre_df.csv')


# %% PART II: SEMI-STRUCTUED DATA ----------------------------------------------

# Authenticate to Twitter
tapi = oauth_login()


# %% READ ALL TWEEETS AND PICKLE THEM IF NECESSARY -----------------------------

hasbro_id = TWITTER_ACCTS.get("Hasbro")
hasbro_tweet_df = from_pickle_or_twitter(hasbro_id, hasbro_id + ".pickle")

sideshow_id = TWITTER_ACCTS.get("Sideshow")
sideshow_tweet_df = from_pickle_or_twitter(sideshow_id, sideshow_id + '.pickle')

hottoys_id = TWITTER_ACCTS.get("Hot Toys")
hottoys_tweet_df = from_pickle_or_twitter(hottoys_id, hottoys_id + '.pickle')

dccoll_id = TWITTER_ACCTS.get("DC Direct/DC Collectibles")
dccoll_tweet_df = from_pickle_or_twitter(dccoll_id, dccoll_id + '.pickle')

gijoecc_id = TWITTER_ACCTS.get("G.I. Joe Collector's Club")
gijoecc_tweet_df = from_pickle_or_twitter(gijoecc_id, gijoecc_id + '.pickle')



# QED!


# %%
