
# %% PROBLEM STATEMENT ---------------------------------------------------------
# Final Project: 1:6 Scale Action Figure Analysis
#        Author: Leonard Armstrong
#           Log: 2021-Mar-04 (LA) Copied from HW01 for final project expansion.
#                2021-Feb-04 (LA) Original version.


# %% LOAD LIBRARIES ------------------------------------------------------------

# Partial library imports
from mizani.formatters import dollar_format		# Format axes with money values
from mizani.formatters import percent_format	# Format axes with pct values

import credentials								# Account ids, pwds, etc.
import datetime as dt							# Dates and time, like it says
import matplotlib as mpl						# For figure resizing
import matplotlib.pyplot as plt					# Ubiquitous Python plotting lib
import matplotlib.ticker as ticker				# For formatting plt axes
import nltk										# Natural language processing
import numpy as np								# Numeric classes/functions
import pathlib as pl							# File path management
import pandas as pd								# C'mon, who doesn't know pandas?
import plotnine as gg							# "gg" refers to ggplot.
import re										# Regular expressions
import seaborn as sns							# Seaborn plotting package			
import squarify									# Generate treemaps.
import statsmodels.api as sm					# Regression models
import tweepy									# Twitter API
import wordcloud as wc							# Wordcloud generator

# %% DEFINE GLOBAL CONSTANTS ---------------------------------------------------

# Path to structured data file
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

# Define tweet limit for a single batch, ±100.
MAX_TWEET_HISTORY = 3000

# Month abbreviation to number conversion
monthnum = {
	"Jan" : 1, "Feb" : 2, "Mar" : 3, "Apr" :  4, "May" :  5, "Jun" :  6,
	"Jul" : 7, "Aug" : 8, "Sep" : 9, "Oct" : 10, "Nov" : 11, "Dec" : 12 }

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

def get_tweet_history (qstr : str):
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
			last_id = tweets[num_tweets_in_batch-1].id - 1
		if (num_tweets_in_batch == 0) or (total_tweets >= MAX_TWEET_HISTORY) :
			# If we recieved less than the requested number of tweets, that's 
			# all there is. No more tweets.
			more_tweets = False
		print(
			f'{qstr}:: Total: {total_tweets}; '
			f'Batch: {len(tweets)}; Last Id: {last_id+1}')

	return (df)


# %% FROM_PICKLE_OR_TWITTER ----------------------------------------------------

def from_pickle_or_twitter (qstr : str, pkl_fpath: str) :
	"""
	Generate a dataframe from either (a) a pickle file if one exists or
	(b) an API read from Twitter. If Twitter is used as the source then a 
	pickle file is written out to disk immediately after, so fast-read data is 
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


# %% REPORT_TIMESTAMP ----------------------------------------------------------

def report_timestamp (year, month, day, hour) :
	"""
	Generate a timestamp for the timeseries report process. The timeseries 
	reports summary values by 1/2 day, separated by either AM (00:00-11:59) or
	PM (12:00-23:59).

	Args:
		year (int): Year in integer format.
		month (int): Month in integre format (1-12).
		day (int): Day of month, in integer format (0-31)
		hour (int): Hour of day in integer format per a 24-hour clock. 

	Returns:
		datetime.datetime: Datetime value for the input date with either 00:00
		or 12:00 as the time portion, representing an AM or PM stamp. 
	"""
	# Get the AM or PM hour using modulus arithmetic.
	ampm_hour = (hour//12) * 12
	# Create and return the result datetime.
	result = dt.datetime(
		year=year, month=month, day=day, hour=ampm_hour, minute=0)
	return result


# %% UPDATE_TWEET_TIMESTAMP ----------------------------------------------------

def update_tweet_timestamp (df : pd.DataFrame) :
	"""
	Add extended datetime details to a tweet dataframe. This is needed becuase
	the tweet `created_at` datetime timestamp is a character string. 

	Args:
		df (pd.DataFrame): Input tweet dataframe.

	Returns:
		pd.DataFrame: Input dataframe that has been updated with extended
		datetime details.
	"""
	# TODO: This function is slow and can likey be improved by comprehensions.

	# Define a regular expression to pull apart the individual pieces from a
	# tweet timestamp string.
	dt_re_str = r"([\w]{3}) ([\w]{3}) ([\d]{2}) ([\d]{2}):([\d]{2}):([\d]{2}) \+([\d]{4}) ([\d]{4})"
	dt_re = re.compile(dt_re_str)

	# Initialize a dataframe to hold the extended date/time details.
	tweet_dates = pd.DataFrame (
		columns=[
			"day_of_week", "month", "day_of_month", "hour", "minutes",
			"seconds", "tz_offset", "year", "datetime", "report_timestamp" ],
		dtype = int)

	# Loop through the tweets...
	for _, tw in df.iterrows() :
		# Gather individual datetime elements
		match = dt_re.match(tw["created_at"])

		# Convert certain fields to other data types
		imonth = monthnum.get(match[2])
		iday_of_month = int(match[3])
		ihour = int(match[4])
		iminutes = int(match[5])
		iseconds = int(match[6])
		iyear = int(match[8])

		# Append the tweet's datatime info to the datatime dataframe.
		tweet_dates = tweet_dates.append(
			{
				# "match" : match[0],
				"day_of_week" : match[1],
				"month" : imonth,
				"day_of_month" : iday_of_month,
				"hour" : ihour,
				"minutes" : iminutes,
				"seconds" : iseconds,
				"tz_offset" : match[7],
				"year" : iyear,
				"datetime" : dt.datetime(
					year=iyear, month=imonth, day=iday_of_month,
					minute = iminutes, second = iseconds),
				"report_timestamp" : report_timestamp(
					iyear, imonth, iday_of_month, ihour)
			},
			ignore_index = True)

	# Add columns of the datetime dataframe onto the right side of the input 
	# dataframe and return the result.
	result_df = df.join(tweet_dates)
	result_df.sort_values(by="datetime", ascending=False, inplace=True)
	return result_df


# %% GENERATE_SPIKE_WORDCLOUD --------------------------------------------------

def generate_spike_workcloud (
	spike_df: pd.DataFrame,
	spike_dt: dt.datetime,
	color: str = 'black') :
	"""
	Generates a wordcloud from tweets. Intended to show tweet activity driving
	a spike in tweet volume. 

	Args:
		spike_df (pd.DataFrame): Dataframe of Tweepy tweet statuses.
		spike_dt (dt.datetime): Datetime code when spike activity occurred.
		color (str, optional): Desired base color for the wordcloud.
		Defaults to 'black'.

	Returns:
		None
	"""
	# Select the full text field from thee spike period
	spike_text = (
		spike_df[spike_df["report_timestamp"] == spike_dt]["full_text"])

	# Create a tweet tokenizer and stopword list
	tknzr = nltk.TweetTokenizer(strip_handles=False, reduce_len=False)
	my_stopwords = wc.STOPWORDS.union({"rt", "!", "…", ",", ".", ":", "#"})

	# Generate the token list and frequency count.
	all_tokens = [ 
		str.lower(tok) 
			for msg in spike_text 
				for tok in tknzr.tokenize(msg)
					if str.lower(tok) not in my_stopwords ]
	freq = nltk.FreqDist(all_tokens)

	# Generate a wordcloud from the frequency data
	cloud = (
		wc.WordCloud(
			max_words=100,
			stopwords=wc.STOPWORDS,
			background_color="white",
			color_func=wc.get_single_color_func(color))
		.generate_from_frequencies(freq))

	# Display the cloud
	plt.figure(figsize=(10, 8))
	plt.clf()
	plt.imshow(cloud)
	plt.axis('off')
	plt.show()

	return None


################################################################################
#    START OF MAIN CODE BODY                                                   #
################################################################################

# %% READ STRUCTURED DATA FROM CSV ---------------------------------------------

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
# WHAT WHAT IS THE PRICE RANGE PAID FOR PRODUCTS?

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

# %% ---------------------------------------------------------------------------
# HOW MANY FIGURES WERE PURCHASED PER YEAR?

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
# WHAT ARE THE RANGES AND AVERAGE PRODUCT PRICE PAID PER YEAR?

# First get a subset of figures with a valid year and price, filtering out
# figures with no price or an unknown year.
figs_with_prices = fig_data[(~fig_data["price"].isna()) & (fig_data["year"]>0)]

# Group figures by year.
year_gb = figs_with_prices.groupby(by="year", axis=0)

# Compute inter-quartile ranges for each year.
iqr = (
	year_gb.quantile(q=0.75)['price'] - year_gb.quantile(q=0.25)['price']
	).reset_index()										# Turns year into a col
iqr.columns = ('Year', 'Price IQR')						# Name columns in DF
iqr["0.25"] = year_gb.quantile(q=0.25)['price'].values	# Give explicit Q2 value
iqr["0.75"] = year_gb.quantile(q=0.75)['price'].values	# Give explicit Q4 value

# Provide a table report of annual IQRs
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

# Then graph the annual price ranges as boxplots
gg.options.figure_size=(14, 11)
g = (
	gg.ggplot(figs_with_prices)
	+ gg.geom_boxplot(mapping=gg.aes(x='factor(year)', y='price'))
	+ gg.theme_bw()
	+ gg.ggtitle('Ranges of Prices Paid for Products Per Year')
	+ gg.xlab('Year of Acquisition')
	+ gg.ylab('Price Paid')
	+ gg.scale_y_continuous(labels=dollar_format(digits=0)))
g.draw()
plt.show()


# %% WHAT ARE THE ACTUAL, AVERAGE, AND LINEAR REGRESSION OF PRICES PAID? -------

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


# %% CEATE A MANUFACTURER SUMMARY DATA STRUCTURE -------------------------------

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


# %% PLOT A HISTOGRAM OF TOP MANUFACTURERS -------------------------------------

plotdata = manu_prod_counts.sort_values(
	by='volume',
	ascending=False)[:16]
g = (
	gg.ggplot(
		data=plotdata, 
		mapping=gg.aes(x="manufacturer", y='volume', fill='volume'))
	+ gg.geom_col()
	+ gg.theme_light()
	+ gg.scale_x_discrete(
		limits=plotdata['manufacturer'])			# Sort x
	+ gg.labs(
		title="Top Manufacturers by Product Volume",
		x = "",
		y = "Product Volume",
		fill = "Product Volume") 
	+ gg.theme(
		axis_text_x = gg.element_text(angle = 55),
		plot_title=gg.element_text(					# Update the title font
			size=14,								# 	14-point font
			face="bold"),							# 	Bold text
		axis_title=gg.element_text(					# Update the axis font
			face="bold"))
	+ gg.scale_y_continuous(limits=(0, 101), breaks=range(0,101,10))
)
g.draw()
plt.show()

print('\nTOP FIVE MANUFACTURERS')
print(plotdata[:5])


# %% PLOT AN HEATMAP OF MANUFACTURERS ⨉ GENRE ----------------------------------

fig, ax = plt.subplots(figsize=(16, 9))
g1 = sns.heatmap(ax=ax, data=m, cmap="Greys")
g1.axes.set_title("Manufacturer/Genre Heatmap\n",fontsize=20)
g1.set_xlabel(None)
g1.set_ylabel(None)
_ = plt.xticks(rotation=70)


# %% PLOT AN ALTERNATE HEATMAP COMBINING ALL MILITARY SERVICES -----------------

# Create a copy of the manufacturer dataframe and sum all military
# service-related counts into a single, new column: Military.
m2 = m.copy()
m2['Military'] = (
	m2['Army'] + 
	m2['Navy'] + 
	m2['Air Force'] + 
	m2['Marines'] + 
	m2['Coast Guard'])
# Drop the old individual service volumes
m2.drop(
	columns=['Army', 'Navy', 'Air Force', 'Marines', 'Coast Guard'],
	inplace=True)
# Resort the columns to put 'Military' in the right spot, alphabetically.
m2.sort_index(axis=1, inplace=True)

# Plot the alternate heatmap/
fig, ax = plt.subplots(figsize=(16, 9))
g2 = sns.heatmap(ax=ax, data=m2, cmap="Greys")
g2.axes.set_title("Manufacturer/Genre Heatmap (Combined Military)\n",fontsize=20)
g2.set_xlabel(None)
g2.set_ylabel(None)
_ = plt.xticks(rotation=70)


# %% PERFORM LINEAR REGRESSION ON ANNUAL AVERAGE PRICES ------------------------

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


# %% PERFORM QUADRATIC REGRESSION ON ANNUAL AVERAGE PRICES ---------------------

# Fit #2: Quadratic, using SQRT of avg_price
print("QUADRATIC REGRESSION")
X_quad = sm.add_constant(vol_price["year"])
quad_model = sm.OLS(vol_price['sqrt_price'], X_quad)
quad_results = quad_model.fit()
print(quad_results.summary())
print(f"\n{'-' * 90}\n\n")

# %% PERFORM EXPONENTIAL REGRESSION ON ANNUAL AVERAGE PRICES -------------------


# Fit #3: Exponential, using LOG of avg_price
# (math.e**(-198.5069))*(math.e**(0.101069*2019))
print("EXPONENTIAL REGRESSION")
X_exp = sm.add_constant(vol_price["year"])
exp_model = sm.OLS(vol_price['log_price'], X_exp)
exp_results = exp_model.fit()
print(exp_results.summary())
print(f"\n{'-' * 90}\n\n")


# %% ---------------------------------------------------------------------------
# QUESTION: HOW DID GENRE COLLECTING CHANGE OVER THE YEARS?
# THIS SECTION REMAINS FROM PRIOR RESEARCH

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


# %% UPDATE TIMESTAMPS; OBTAIN A 1/2 DAY REPORT TIMESTAMP ----------------------
# This section take a little while - about 30 seconds on an 2020, iMac, Core i5

hasbro_tweet_df = update_tweet_timestamp(hasbro_tweet_df)
sideshow_tweet_df = update_tweet_timestamp(sideshow_tweet_df)
hottoys_tweet_df = update_tweet_timestamp(hottoys_tweet_df)
dccoll_tweet_df = update_tweet_timestamp(dccoll_tweet_df)
gijoecc_tweet_df = update_tweet_timestamp(gijoecc_tweet_df)


# %% COMBINE MANUFACTURER DATAFRAMES -------------------------------------------

sideshow_tweet_df["group"] = "Sideshow"
hottoys_tweet_df["group"] = "Hot Toys"
dccoll_tweet_df["group"] = "DC Collectibles"
hasbro_tweet_df["group"] = "Hasbro"
gijoecc_tweet_df["group"] = "G.I. Joe Collectors Club"
all = (
	sideshow_tweet_df
	.append(hottoys_tweet_df)
	.append(dccoll_tweet_df)
	.append(hasbro_tweet_df)
	# .append(gijoecc_tweet_df)		# Left out: 1 data point causes plot error
	.sort_values(by="report_timestamp"))


# %% CREATE A TIMESERIES REPORT ------------------------------------------------

g = (
	gg.ggplot(										# Use combined dataframe
		data=all,
		mapping=gg.aes(
			x="report_timestamp",					# 	x-axis: report_timestamp
			group="group",							# 	1 line per manufacturer
			color="group",
			linetype="group"))
	+ gg.geom_line(size = 1, stat="count")			# Line plots
	+ gg.theme_light()								# Lighter plot theme
	+ gg.labs(										# Labels
		title = "Tweet Timeseries for Four Manufacturers",
		x = "",										# 	No x-axis label
		y = "Tweet Volumes per AM/PM, Daily",		# 	y-axis label
		linetype = "Manufacturer",					# 	Legend label
		color = "Manufacturer")						# 	Legend label
	+ gg.theme(
		plot_title=gg.element_text(					# Update the title font
			size=14,								# 	14-point font
			face="bold"),							# 	Bold text
		axis_title=gg.element_text(face="bold"))	# Axis font : bold text
	+ gg.theme(
		axis_text_x = gg.element_text(angle = 55)))	# Angle to x-axis labels

g.draw()
plt.show()


# %% GENERATE WORD CLOUDS FOR SPIKE PERIODS ------------------------------------

sideshow_spike = dt.datetime(year=2021, month=3, day=10, hour=0, minute=0)
generate_spike_workcloud(sideshow_tweet_df, sideshow_spike, 'mediumblue')

hottoys_spike = dt.datetime(year=2021, month=3, day=11, hour=0, minute=0)
generate_spike_workcloud(hottoys_tweet_df, sideshow_spike, "tomato")

hasbro_spike = dt.datetime(year=2021, month=3, day=12, hour=12, minute=0)
generate_spike_workcloud(hasbro_tweet_df, hasbro_spike, "darkgreen")


# %%
