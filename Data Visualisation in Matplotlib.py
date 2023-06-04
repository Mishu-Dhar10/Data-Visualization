#!/usr/bin/env python
# coding: utf-8

# In[14]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.show()


# In[15]:


import pandas as pd
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
seattle_weather = pd.read_csv('/Users/mishudhar/Downloads/matplotlib and statistics course/seattle_weather.csv')


# In[16]:


seattle_weather.head(15)


# In[17]:


temp = list(np.random.randint(42, 70, size = 12))


# In[18]:


Seatt_weather = pd.DataFrame({'Month':['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                'Month_avg_temp':temp})
Seatt_weather.head()


# In[19]:


np.random.seed(42)
random_temp = np.random.randint(42, 70, size = 12)
temp_list = list(random_temp)
austin_weather = pd.DataFrame({'Month':['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                 'month_avg_temp': temp_list})
austin_weather.head()


# In[20]:


fig, ax = plt.subplots()
#ax.plot(Seatt_weather["Month"], Seatt_weather["Month_avg_temp"], linestyle = "None", marker = 'v')
ax.plot(austin_weather['Month'], austin_weather['month_avg_temp'])
plt.show()


# In[21]:


type(Seatt_weather)


# In[22]:


fig = px.line(Seatt_weather, x = "Month", y = "Month_avg_temp")
fig.show()


# In[23]:


import plotly.graph_objects as go
import numpy as np

# Generate some random data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create trace objects for both lines
trace1 = go.Scatter(x=x, y=y1, mode='lines', name='Sine')
trace2 = go.Scatter(x=x, y=y2, mode='lines', name='Cosine')

# Add both trace objects to a data list
data = [trace1, trace2]

# Set the layout of the plot
layout = go.Layout(title='Sine and Cosine Waves', xaxis_title='x', yaxis_title='y')

# Create a Figure object that contains the data and layout
fig = go.Figure(data=data, layout=layout)

# Show the plot
fig.show()


# In[24]:


seattle_weather.shape


# In[27]:


Austin_weather = pd.read_csv('/Users/mishudhar/Downloads/matplotlib and statistics course/austin_weather.csv')
Austin_weather.shape


# In[28]:


# Slicing the dataset for 12 month
d_1 = seattle_weather.head(12)


# In[ ]:





# # adding new column as a month
# month_list =  ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# new_series = pd.Series(month_list)
# #d_1['Month'] = new_series
# new_series

# In[29]:


d_1


# In[30]:


Austin_weather


# In[31]:


fig, ax = plt.subplots()
ax.plot(d_1["Month"], d_1["MLY-PRCP-NORMAL"])
ax.plot(Austin_weather["Month"], Austin_weather["MLY-PRCP-NORMAL"])
plt.show()


# # Customizing the plots

# In[32]:


fig, ax = plt.subplots()
ax.plot(Seatt_weather["Month"], Seatt_weather["Month_avg_temp"], marker = "o") # circle markere
ax.plot(austin_weather['Month'], austin_weather['month_avg_temp'], marker = "v", linestyle = "--",
       color = 'r') # Triangle mamrker
plt.show() # Notice the line stylef


# In[33]:


fig, ax = plt.subplots()
ax.plot(Seatt_weather["Month"], Seatt_weather["Month_avg_temp"], marker = "o") # circle markere
ax.plot(austin_weather['Month'], austin_weather['month_avg_temp'], marker = "v") # Triangle mamrker
ax.set_xlabel("Time(months)")
ax.set_ylabel("Temperature(Farenhite)")
ax.set_title("Weather in Seattle and Austin")
plt.show()


# # Samll multiples

# In[38]:


fig, ax = plt.subplots(3,2)
plt.show()
ax.shape


# In[39]:


fig, ax = plt.subplots(3,2, sharey = True)

ax[0, 0].plot(d_1["Month"], d_1["MLY-PRCP-NORMAL"])
ax[1,1].plot(Austin_weather["Month"], Austin_weather["MLY-PRCP-NORMAL"])

plt.show()


# In[36]:


fig, ax = plt.subplots(2,1, sharey = True) # for same range of y axes value
ax[0].plot(d_1["Month"], d_1["MLY-PRCP-NORMAL"], color = 'b')
ax[0].plot(d_1["Month"], d_1["MLY-PRCP-25PCTL"], linestyle = "--", color = 'b')
ax[0].plot(d_1["Month"], d_1["MLY-PRCP-75PCTL"], linestyle = "--", color = 'b')
ax[1].plot(Austin_weather["Month"], Austin_weather["MLY-PRCP-NORMAL"], color = 'r')
ax[1].plot(Austin_weather["Month"], Austin_weather["MLY-PRCP-25PCTL"], linestyle = "--", color = 'r')
ax[1].plot(Austin_weather["Month"], Austin_weather["MLY-PRCP-75PCTL"], linestyle = "--", color = 'r')
ax[0].set_ylabel("Percipitation (inches)")
ax[1].set_ylabel("Percipitation (inches)")

ax[1].set_xlabel("Time (months)")

plt.show()


# In[139]:


climate = pd.read_csv('/Users/mishudhar/Downloads/climate_change.csv')
climate.head()


# In[140]:


climate.index


# In[143]:


fig, ax = plt.subplots()
ax.plot(climate.index, climate['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()


# In[142]:


fig = px.line(climate, x = 'date', y = 'co2')
fig.show()


# In[147]:


Climate_change = pd.read_csv('/Users/mishudhar/Downloads/climate_change.csv', index_col = [0])
Climate_change.head()


# In[148]:


Climate_change.index


# In[150]:


fig, ax = plt.subplots()
ax.plot(Climate_change.index, Climate_change['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()


# In[161]:


# Converting date into datettime format from object
klimat = pd.read_csv('/Users/mishudhar/Downloads/climate_change.csv')
klimat['date'] = pd.to_datetime(klimat['date'])

klimat.head()


# In[162]:


klimat.set_index('date', inplace = True)
klimat.head()


# In[163]:


klimat.index


# In[164]:


fig, ax = plt.subplots()
ax.plot(klimat.index, klimat['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()


# In[166]:


# Slicing the dataset for one decade
sixties = klimat["1960-01-01":"1969-12-31"]
fig, ax = plt.subplots()
ax.plot(sixties.index, sixties['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()


# In[167]:


sixty_nine = klimat["1969-01-01":"1969-12-31"]
fig, ax = plt.subplots()
ax.plot(sixty_nine.index, sixty_nine['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()


# In[170]:


# A better approach to importing data and converting date to datetime
Climate = pd.read_csv('/Users/mishudhar/Downloads/climate_change.csv', parse_dates = True, index_col = 'date')
Climate.head()


# In[171]:


Climate.index


# In[179]:


fig = px.line(Climate, x = Climate.index, y = 'relative_temp', title = 'Relative Temp. Over The Years')
fig.show()


# In[185]:


Ninties = Climate["1990-01-01":"1999-12-31"]
fig = px.line(data_frame = Ninties, x = Ninties.index, y='co2')
fig.show()


# # Plotting time series with different variable

# In[186]:


fig, ax = plt.subplots()
ax.plot(Climate.index, Climate['co2'])
ax.plot(Climate.index, Climate['relative_temp'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm) / Relative temperature')
plt.show()


# # Comment- because the unit of the measurments are not the same. To overcome this situation

# In[193]:


fig, ax = plt.subplots()
ax.plot(Climate.index, Climate['co2'], color = 'blue')
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)', color = 'blue')

#  Creating a twin
ax2 = ax.twinx()
ax2.plot(Climate.index, Climate['relative_temp'], color = 'red')
ax2.set_ylabel('Relative Temperature (Celsius)', color = 'red')
plt.show()


# # More interative
# tick_params

# In[194]:


fig, ax = plt.subplots()
ax.plot(Climate.index, Climate['co2'], color = 'blue')
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)', color = 'blue')
ax.tick_params('y', colors = 'blue')

#  Creating a twin
ax2 = ax.twinx()
ax2.plot(Climate.index, Climate['relative_temp'], color = 'red')
ax2.set_ylabel('Relative Temperature (Celsius)', color = 'red')
ax2.tick_params('y', colors = 'red')
plt.show()


# # A function that plots time series

# In[195]:


def plot_timeseries(axes, x, y, color, xlabel, ylabel):
    axes.plot(x, y, color = color)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel, color = color)
    axes.tick_params('y', colors = color)


# In[201]:


fig, ax = plt.subplots()
plot_timeseries(ax, Climate.index, Climate['co2'], 'blue', 'Time', 'CO2 (ppm)')

ax2 = ax.twinx()
plot_timeseries(ax2, Climate.index, Climate['relative_temp'], 'red', 'Time', 'Relative Temperature (Celsius)')
plt.show()


# In[202]:


import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.DataFrame({
    'Date': pd.date_range('2022-01-01', periods=10, freq='D'),
    'Sales': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
    'Profit': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
})

df.set_index('Date', inplace=True)

print(df)


# In[203]:


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df.index, y=df['Sales'], name='Sales'), secondary_y=False)
fig.add_trace(go.Scatter(x=df.index, y=df['Profit'], name='Profit'), secondary_y=True)
fig.update_layout(title='Sales and Profit by Date')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Sales', secondary_y=False)
fig.update_yaxes(title_text='Profit', secondary_y=True)
fig.show()


# In[13]:


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x= Climate.index, y= Climate['co2'], name='CO2'), secondary_y=False)
fig.add_trace(go.Scatter(x= Climate.index, y= Climate['relative_temp'], name= ' Temp (Celsius)'), secondary_y=True)
fig.update_layout(title='CO2 and Temperature Over The Years')
fig.update_xaxes(title_text='Year')
fig.update_yaxes(title_text='CO2', secondary_y=False)
fig.update_yaxes(title_text='Temp', secondary_y=True)
fig.show()


# # Annotating time-series data

# In[208]:


fig, ax = plt.subplots()
plot_timeseries(ax, Climate.index, Climate['co2'], 'blue', 'Time', 'CO2 (ppm)')

ax2 = ax.twinx()
plot_timeseries(ax2, Climate.index, Climate['relative_temp'], 'red', 'Time', 'Relative Temperature (Celsius)')
ax2.annotate(">1 degree", xy = (pd.Timestamp("2015-10-06"), 1))
plt.show()


# # It is a mess

# In[209]:


fig, ax = plt.subplots()
plot_timeseries(ax, Climate.index, Climate['co2'], 'blue', 'Time', 'CO2 (ppm)')

ax2 = ax.twinx()
plot_timeseries(ax2, Climate.index, Climate['relative_temp'], 'red', 'Time', 'Relative Temperature (Celsius)')
ax2.annotate(">1 degree", xy = (pd.Timestamp("2015-10-06"), 1),
            xytext = (pd.Timestamp('2008-10-06'), -0.2))
plt.show()


# Now it is not showing which data is indicating

# In[210]:


# Adding ana arrow
fig, ax = plt.subplots()
plot_timeseries(ax, Climate.index, Climate['co2'], 'blue', 'Time', 'CO2 (ppm)')

ax2 = ax.twinx()
plot_timeseries(ax2, Climate.index, Climate['relative_temp'], 'red', 'Time', 'Relative Temperature (Celsius)')
ax2.annotate(">1 degree", xy = (pd.Timestamp("2015-10-06"), 1),
            xytext = (pd.Timestamp('2008-10-06'), -0.2),
            arrowprops = {})
plt.show()


# In[211]:


fig, ax = plt.subplots()
plot_timeseries(ax, Climate.index, Climate['co2'], 'blue', 'Time', 'CO2 (ppm)')

ax2 = ax.twinx()
plot_timeseries(ax2, Climate.index, Climate['relative_temp'], 'red', 'Time', 'Relative Temperature (Celsius)')
ax2.annotate(">1 degree", xy = (pd.Timestamp("2015-10-06"), 1),
            xytext = (pd.Timestamp('2008-10-06'), -0.2),
            arrowprops = {'arrowstyle':'->', 'color':'gray'})
plt.show()


# In[213]:


fig, ax = plt.subplots()
ax.plot(Climate.index, Climate['relative_temp'])
ax.annotate('>1 degree', xy = (pd.Timestamp('2015-10-06'), 1))
plt.show()


# # Quantitative comparisons

# In[216]:


medals = pd.read_csv('/Users/mishudhar/Downloads/medals_by_country_2016.csv', index_col = 0)
medals.head()


# In[218]:


fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
plt.show()


# # Names are overlapping

# In[222]:


fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.set_xticklabels(medals.index, rotation = 90)
ax.set_ylabel('Number of Medals')
plt.show()


# In[12]:


fig = px.bar(medals, x = medals.index, y = 'Gold', color = medals.index)
fig.show()


# # Stacked bar charts

# In[227]:


fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.bar(medals.index, medals['Silver'], bottom = medals['Gold'])
ax.bar(medals.index, medals['Bronze'], bottom = medals['Gold'] + medals['Silver'])
ax.set_xticklabels(medals.index, rotation = 90)
ax.set_ylabel('Number of medals')
ax.legend()
plt.show()


# In[229]:


fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'], label = 'Gold')
ax.bar(medals.index, medals['Silver'], bottom = medals['Gold'], label = 'Silver')
ax.bar(medals.index, medals['Bronze'], bottom = medals['Gold'] + medals['Silver'], label = 'Bronze')
ax.set_xticklabels(medals.index, rotation = 90)
ax.set_ylabel('Number of medals')
ax.legend()
plt.show()


# In[231]:


import plotly.graph_objects as go
traces = []
for column in ['Gold', 'Bronze', 'Silver']:
    trace = go.Bar(x = medals.index, y = medals[column], name = column)
    traces.append(trace)
layout = go.Layout(title = 'Medals by Countries', barmode = 'stack')
fig = go.Figure(data = traces, layout = layout)
fig.show()


# # Quantitative Comparison

# In[41]:


summer_olympic = pd.read_csv('/Users/mishudhar/Downloads/matplotlib and statistics course/summer2016.csv')
summer_olympic.head()


# In[42]:


mens_rowing = summer_olympic[summer_olympic['Sport'] == 'Rowing']


# In[43]:


mens_rowing.head()


# In[44]:


mens_gymnastics = summer_olympic[summer_olympic['Sport'] == 'Gymnastics']
mens_gymnastics.head()


# In[45]:


fig, ax = plt.subplots()
ax.bar('Rowing(Olympic)', mens_rowing['Height'].mean())
ax.bar('Gymnastics(Olympic)', mens_gymnastics['Height'].mean())
ax.set_ylabel('Height (cm)')
plt.show()


# In[46]:


fig, ax = plt.subplots()
ax.hist(mens_rowing['Height'], label = 'Rowing')
ax.hist(mens_gymnastics['Height'], label = 'Gymnastics')
ax.set_xlabel('Height (cm)')
ax.set_ylabel('# of observations')
ax.legend()
plt.show()


# In[ ]:





# In[47]:


fig, ax = plt.subplots()
ax.hist(mens_rowing['Height'], label = 'Rowing', bins = [150, 160, 170, 180, 190, 200, 210])
ax.hist(mens_gymnastics['Height'], label = 'Gymnastics', bins = [150, 160, 170, 180, 190, 200, 210])
ax.set_xlabel('Height (cm)')
ax.set_ylabel('# of observations')
ax.legend()
plt.show()
print('By default binsize is 10')


# In[48]:


fig, ax = plt.subplots()
ax.hist(mens_rowing['Height'], label = 'Rowing', bins = [150, 160, 170, 180, 190, 200, 210], histtype = 'step')
ax.hist(mens_gymnastics['Height'], label = 'Gymnastics', bins = [150, 160, 170, 180, 190, 200, 210], histtype = 'step')
ax.set_xlabel('Height (cm)')
ax.set_ylabel('# of observations')
ax.legend()
plt.show()


# In[49]:


import plotly.express as px

fig = px.histogram(mens_rowing, x = 'Height', nbins = 20, opacity = 0.5, barmode = 'overlay')
fig.add_histogram( x = mens_gymnastics['Height'])

fig.show()


# # Adding error bars in plot

# In[50]:


fig, ax = plt.subplots()
ax.bar('Rowing', mens_rowing['Height'].mean(),
      yerr = mens_rowing['Height'].std())
ax.bar('Gymnastics', mens_gymnastics['Height'].mean(),
      yerr = mens_gymnastics['Height'].std())
ax.set_xlabel('Height (cm)')
plt.show()


# In[51]:


Seat_weather = d_1


# In[52]:


Seat_weather.head()


# In[53]:


Austin_weather.head()


# In[54]:


fig, ax = plt.subplots()
ax.errorbar(Seat_weather['Month'], Seat_weather['MLY-TAVG-NORMAL'],
           yerr = Seat_weather['MLY-TAVG-STDDEV'])
ax.errorbar(Austin_weather['Month'], Austin_weather['MLY-TAVG-NORMAL'],
           yerr = Austin_weather['MLY-TAVG-STDDEV'])
ax.set_ylabel('Temperature (Farenhit)')
plt.show()


# In[ ]:





# In[55]:


fig, ax = plt.subplots()
ax.boxplot([mens_rowing['Height'], mens_gymnastics['Height']])

ax.set_xticklabels(["Rowing", "Gymnastics"])

ax.set_ylabel('Height (cm)')

plt.show()


# In[56]:


fig, ax = plt.subplots()
ax.scatter(Climate['co2'], Climate['relative_temp'])
ax.set_xlabel('CO2 (ppm)')
ax.set_ylabel('Relative temperature (celsius)')
plt.show()


# In[288]:


Eighties = Climate["1980-01-01":"1989-12-31"]
Eighties.head()


# In[290]:


Nineties = Climate["1990-01-01":"1999-12-31"]
Nineties.head()


# In[291]:


fig, ax = plt.subplots()
ax.scatter(Eighties['co2'], Eighties['relative_temp'],
          color = 'red', label = 'eighties')
ax.scatter(Nineties['co2'], Nineties['relative_temp'],
           color = 'blue', label = 'nineties')
ax.legend()

ax.set_xlabel("CO2 (ppm)")
ax.set_ylabel("Relative temperature (Celsius)")
plt.show()


# In[293]:


fig, ax = plt.subplots()
ax.scatter(Climate['co2'], Climate['relative_temp'], c = Climate.index)
ax.set_xlabel('CO2 (ppm)')
ax.set_ylabel('Relative temperature (Celsius)')
plt.show()


# # Preparing your plots to share with others

# In[296]:


plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.errorbar(Seat_weather['Month'], Seat_weather['MLY-TAVG-NORMAL'],
           yerr = Seat_weather['MLY-TAVG-STDDEV'])
ax.errorbar(Austin_weather['Month'], Austin_weather['MLY-TAVG-NORMAL'],
           yerr = Austin_weather['MLY-TAVG-STDDEV'])
ax.set_ylabel('Temperature (Farenhit)')
plt.show()


# In[297]:


plt.style.use('seaborn-colorblind')
fig, ax = plt.subplots()
ax.errorbar(Seat_weather['Month'], Seat_weather['MLY-TAVG-NORMAL'],
           yerr = Seat_weather['MLY-TAVG-STDDEV'])
ax.errorbar(Austin_weather['Month'], Austin_weather['MLY-TAVG-NORMAL'],
           yerr = Austin_weather['MLY-TAVG-STDDEV'])
ax.set_ylabel('Temperature (Farenhit)')
plt.show()


# In[299]:


plt.style.use('tableau-colorblind10')
fig, ax = plt.subplots()
ax.errorbar(Seat_weather['Month'], Seat_weather['MLY-TAVG-NORMAL'],
           yerr = Seat_weather['MLY-TAVG-STDDEV'])
ax.errorbar(Austin_weather['Month'], Austin_weather['MLY-TAVG-NORMAL'],
           yerr = Austin_weather['MLY-TAVG-STDDEV'])
ax.set_ylabel('Temperature (Farenhit)')
plt.show()


# various styles are such as Solarize_Light2, grayscale(for black and white printer)

# # Saving Figures
# 

# In[302]:


fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.set_xticklabels(medals.index, rotation = 90)
ax.set_ylabel('Number of Medals')
#ax.legend()
fig.savefig('gold_medals.png')
#fig.savefig('gold_medals.jpg') for JPG file


# In[304]:


fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.set_xticklabels(medals.index, rotation = 90)
ax.set_ylabel('Number of Medals')
#ax.legend()
#fig.savefig('gold_medals.png')
fig.savefig('gold_medals.jpg', quality = 50) 


# In[305]:


fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.set_xticklabels(medals.index, rotation = 90)
ax.set_ylabel('Number of Medals')
#ax.legend()
#fig.savefig('gold_medals.png')
fig.savefig('gold_medals.svg') 


# In[307]:


fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.set_xticklabels(medals.index, rotation = 90)
ax.set_ylabel('Number of Medals')
#ax.legend()
#fig.savefig('gold_medals.png')
fig.savefig('gold_medals.png', dpi = 300) # dot per inch


# In[308]:


fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.set_xticklabels(medals.index, rotation = 90)
ax.set_ylabel('Number of Medals')
#ax.legend()
#fig.savefig('gold_medals.png
fig.set_size_inches([5,3] # First number is the width and second number is the height
fig.savefig('gold_medals.png', dpi = 300)


# In[309]:


ls


# # Automating figures from Data
# # Why automating?
# 1. Ease and speed
# 2. Flexibility
# 3. Robustness
# 4. Reproducibility

# In[310]:


summer_olympic.head()


# In[312]:


# Getting the unique values from a column
sports = summer_olympic['Sport'].unique()
sports


# # Bar charts for heights of all sports

# In[314]:


fig, ax = plt.subplots()
for sport in sports:
    sport_df = summer_olympic[summer_olympic['Sport'] == sport]# if the value of Sport column is one of this unique column
    ax.bar(sport, sport_df['Height'].mean(),
          yerr = sport_df['Height'].std())
ax.set_ylabel('Height (cm)')
ax.set_xticklabels(sports, rotation = 90)
plt.show()


# In[321]:


fig = px.bar(summer_olympic, x = 'Sport', y = summer_olympic['Height'], color = 'Sport')
fig.show()


# In[322]:


summer_olympic.columns


# # An alternative approach by using plotly

# In[57]:


alternative = summer_olympic.groupby(['Sport']).agg({'Height':'mean'}).reset_index()
alternative.head()


# In[328]:


fig = px.bar(data_frame = alternative, x = 'Sport', y = 'Height', color = 'Sport')
fig.show()


# # To learn more
# visit
# https://matplotlib.org/stable/gallery/index.html
# https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html#sphx-glr-gallery-mplot3d-scatter3d-py
# https://matplotlib.org/stable/tutorials/introductory/images.html
# https://matplotlib.org/stable/api/animation_api.html
# https://scitools.org.uk/cartopy/docs/latest/getting_started/index.html #for Geospatial data
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




